import collections
import datetime
import gin
import numpy as np
import os
import pprint
import re
import shutil
import tensorflow as tf

from absl import flags
from absl import logging

from thin.estimator_specs import TrainSpec
from thin.estimator_specs import EvalSpec
from thin.estimator_specs import PredictSpec
from thin.estimator_specs import RunConfig
from thin.utils import ndarray_feature

tf_v1 = tf.compat.v1

flags.DEFINE_string('do', 'train', '')
flags.DEFINE_string('root_dir', None, 'Root directory.')
flags.DEFINE_string('checkpoint_dir', None, 'Directory to load checkpoint/gin config.')
flags.DEFINE_string('checkpoint_path', None, 'Path to load checkpoint.')
flags.DEFINE_string('gin_file', None, 'Gin config file.')
flags.DEFINE_multi_string('gin_param', None, 'Gin config parameters.')
FLAGS = flags.FLAGS

_CONFIG_GIN = 'operative_config-0.gin'


def main(_):
    if FLAGS.gin_file:
        gin_paths = [FLAGS.gin_file]

    elif (FLAGS.checkpoint_dir or FLAGS.checkpoint_path):
        checkpoint_dir = FLAGS.checkpoint_dir
        if checkpoint_dir is None:
            checkpoint_dir = os.path.dirname(FLAGS.checkpoint_path)

        gin_paths = [os.path.join(checkpoint_dir, _CONFIG_GIN)]

    else:
        gin_paths = []

    gin.parse_config_files_and_bindings(gin_paths, FLAGS.gin_param)
    estimator = Estimator()
    getattr(estimator, FLAGS.do)()


class InputFn(object):
    @staticmethod
    def create_dir(base_dir):
        dir_path = os.path.join(
            base_dir,
            datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S-%f'))
        os.makedirs(dir_path)
        return dir_path

    @property
    def root_dir(self):
        return FLAGS.root_dir

    @property
    def data_dir(self):
        return os.path.join(self.root_dir, 'data')

    @property
    def model_dir_root(self):
        return os.path.join(self.root_dir, 'models')

    @property
    def result_dir_root(self):
        return os.path.join(self.root_dir, 'results')

    @property
    def split_dir_root(self):
        return os.path.join(self.data_dir, 'splits')

    @property
    def tfrecord_dir_root(self):
        return os.path.join(self.data_dir, 'tfrecords')

    def _write_tfrecord(self, tfrecord, tfrecord_path):
        writer = tf.io.TFRecordWriter(tfrecord_path)
        feature = {
            key: ndarray_feature(value)
            for (key, value) in tfrecord.items()}

        logging.info(f'Caching features {list(feature)} to {tfrecord_path}')

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    def _dataset_from_df(self, df, training):
        dataset = tf.data.Dataset.range(len(df))

        if training:
            dataset = dataset.shuffle(len(df))
            dataset = dataset.repeat()

        def _get_item(i):
            return df.iloc[i].to_list()

        dtypes = [
            tf.as_dtype(np.asarray(df[column].iloc[0]))
            for column in df.columns]

        dataset = dataset.map(
            lambda i: tf.numpy_function(_get_item, [i], dtypes),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            lambda *values: dict(zip(df.columns, values)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return dataset

    def _example_to_features_and_labels(self, example, label_keys):
        features = example.copy()
        labels = {}
        for key in label_keys:
            value = features.pop(key, None)
            if value is not None:
                labels[key] = value

        return (features, labels)

    def _input_fn_train_or_eval(self, training, batch_size):
        pass

    def _input_fn_predict(self, batch_size):
        pass

    def input_fn(self, batch_size, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            return self._input_fn_train_or_eval(
                training=True, batch_size=batch_size)

        elif mode == tf.estimator.ModeKeys.EVAL:
            return self._input_fn_train_or_eval(
                training=False, batch_size=batch_size)

        elif mode == tf.estimator.ModeKeys.PREDICT:
            return self._input_fn_predict(
                batch_size=batch_size)


class ModelFn(object):
    def _get_global_step(self):
        return tf_v1.train.get_global_step()

    def _register_model_updates(self, model, features):
        update_ops = model.get_updates_for(None) + model.get_updates_for(features.values())
        for update_op in update_ops:
            tf_v1.add_to_collection(tf_v1.GraphKeys.UPDATE_OPS, update_op)

    def _regularization_loss(self):
        with tf.name_scope('reg_loss'):
            return (
                self._train_spec.weight_decay *
                tf.add_n([
                    tf.nn.l2_loss(var)
                    for var in tf_v1.trainable_variables()]))

    def model_fn(self, features, labels, mode):
        pass

    def _build_estimator_spec(self,
                              mode,
                              predictions,
                              model_loss=None,
                              metrics=None,
                              print_tensors=None):

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions)

        if print_tensors:
            print_ops = [
                tf.print(f'{key} =', value, summarize=256)
                for (key, value) in print_tensors.items()]

            with tf.control_dependencies(print_ops):
                model_loss = tf.identity(model_loss)

        total_loss = model_loss
        metrics = metrics or {}

        metrics['loss/model_loss'] = model_loss

        if mode == tf.estimator.ModeKeys.TRAIN:
            if self._train_spec.weight_decay:
                reg_loss = self._regularization_loss()

                total_loss = total_loss + reg_loss
                metrics['loss/reg_loss'] = reg_loss

            global_step = self._get_global_step()
            if self._train_spec.lr_decay_steps:
                learning_rate = tf_v1.train.exponential_decay(
                    self._train_spec.lr,
                    global_step=global_step,
                    decay_steps=self._train_spec.lr_decay_steps,
                    decay_rate=self._train_spec.lr_decay_rate,
                    staircase=True)
            else:
                learning_rate = tf.constant(self._train_spec.lr)

            metrics['learning_rate'] = learning_rate

            trainable_variables = tf_v1.get_collection(tf_v1.GraphKeys.TRAINABLE_VARIABLES)
            lr_groups = self._train_spec.lr_groups

            lr_group_to_variables = collections.defaultdict(list)
            for variable in trainable_variables:
                has_match = False
                for (lr_group_name, lr_scaling) in lr_groups.items():
                    if (not has_match) and re.match(lr_group_name, variable.name):
                        has_match = True
                        if lr_scaling:
                            lr_group_to_variables[lr_group_name].append(variable)

                if not has_match:
                    lr_group_to_variables['__all__'].append(variable)

            logging.info('Learning rate groups')
            logging.info(pprint.pformat(lr_group_to_variables))

            variables = sum(list(lr_group_to_variables.values()), [])
            gradients = tf.gradients(total_loss, variables)

            if self._train_spec.gradient_clip:
                (gradients, _) = tf.clip_by_global_norm(
                    gradients, self._train_spec.gradient_clip)

            variable_to_gradient = dict(zip(variables, gradients))

            apply_ops = []
            for lr_group_name in lr_group_to_variables.keys():
                lr_scaling = lr_groups.get(lr_group_name, 1.0)  # for __all__

                if not lr_scaling:
                    continue

                optimizer = self._train_spec.optimizer_cls(
                    learning_rate=lr_scaling * learning_rate)
                optimizer_variables = lr_group_to_variables[lr_group_name]

                gradient_variables = [
                    (variable_to_gradient[variable], variable)
                    for variable in optimizer_variables]

                apply_op = optimizer.apply_gradients(
                    gradient_variables, global_step=global_step)
                apply_ops.append(apply_op)

            update_ops = tf_v1.get_collection(tf_v1.GraphKeys.UPDATE_OPS)
            train_op = tf.group(update_ops, *apply_ops)

            for (key, value) in metrics.items():
                if isinstance(value, tuple):  # global metric
                    scalar = value[0]
                else:  # local metric
                    scalar = value

                tf_v1.summary.scalar(key, scalar)

            return tf.estimator.EstimatorSpec(
                mode=mode, loss=total_loss, train_op=train_op)

        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {}
            for (key, value) in metrics.items():
                if isinstance(value, tuple):  # global metric
                    eval_metric_op = value
                else:  # local metric
                    eval_metric_op = tf_v1.metrics.mean(value)

                eval_metric_ops[key] = eval_metric_op

            evaluation_hooks = []
            summary_op = tf_v1.summary.merge_all()
            if summary_op is not None:
                summary_hook = tf.estimator.SummarySaverHook(
                    save_steps=self._eval_spec.save_summary_per_steps,
                    output_dir=self._estimator.eval_dir(self._eval_spec.name),
                    summary_op=summary_op)
                evaluation_hooks.append(summary_hook)

            return tf.estimator.EstimatorSpec(
                mode=mode, loss=total_loss, eval_metric_ops=eval_metric_ops,
                evaluation_hooks=evaluation_hooks)


@gin.configurable
class Estimator(InputFn, ModelFn):
    def __new__(cls, estimator_cls=None):
        if estimator_cls is None:
            return super(Estimator, cls).__new__(cls)
        elif issubclass(estimator_cls, Estimator):
            return super(Estimator, cls).__new__(estimator_cls)
        else:
            obj = super(estimator_cls, estimator_cls).__new__(estimator_cls)
            obj.__init__()
            return obj

    def train_eval(self):
        model_dir = self.create_dir(self.model_dir_root)
        shutil.copy(FLAGS.gin_file, os.path.join(model_dir, _CONFIG_GIN))

        self._train_spec = TrainSpec(
            input_fn=self.input_fn,
            hooks=[gin.tf.GinConfigSaverHook(model_dir)])
        self._eval_spec = EvalSpec(input_fn=self.input_fn)

        if self._train_spec.checkpoint_path is None:
            warm_start_from = None
        else:
            warm_start_from = tf.estimator.WarmStartSettings(
                ckpt_to_initialize_from=os.path.join(
                    self.root_dir, self._train_spec.checkpoint_path),
                vars_to_warm_start=self._train_spec.checkpoint_vars)

        self._estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            model_dir=model_dir,
            config=RunConfig(),
            warm_start_from=warm_start_from)

        tf.estimator.train_and_evaluate(
            self._estimator, self._train_spec, self._eval_spec)

    def predict(self, predict_keys=None):
        self._predict_spec = PredictSpec(input_fn=self.input_fn)
        self._estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            config=RunConfig())

        checkpoint_path = FLAGS.checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

        return self._estimator.predict(
            input_fn=self._predict_spec.input_fn,
            predict_keys=predict_keys,
            checkpoint_path=checkpoint_path)

    def eval(self):
        pass
