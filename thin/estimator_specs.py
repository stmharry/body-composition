import functools
import gin
import gin.tf.external_configurables
import tensorflow as tf

gin.config.external_configurable(
    tf.distribute.MirroredStrategy, module='tf.distribute')

RunConfig = gin.external_configurable(tf.estimator.RunConfig)


@gin.configurable
class TrainSpec(tf.estimator.TrainSpec):
    def __new__(cls,
                input_fn,
                checkpoint_path=None,
                checkpoint_vars='.*',
                lr=gin.REQUIRED,
                lr_groups={},
                lr_decay_rate=None,
                lr_decay_steps=None,
                optimizer_cls=gin.REQUIRED,
                weight_decay=None,
                gradient_clip=None,
                batch_size=gin.REQUIRED,
                max_steps=gin.REQUIRED,
                hooks=None):

        self = super(TrainSpec, cls).__new__(
            cls,
            input_fn=functools.partial(
                input_fn,
                batch_size=batch_size,
                mode=tf.estimator.ModeKeys.TRAIN),
            max_steps=max_steps,
            hooks=hooks)

        self.checkpoint_path = checkpoint_path
        self.checkpoint_vars = checkpoint_vars
        self.lr = lr
        self.lr_groups = lr_groups
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps
        self.optimizer_cls = optimizer_cls
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip

        return self


@gin.configurable
class EvalSpec(tf.estimator.EvalSpec):
    def __new__(cls,
                input_fn,
                batch_size=gin.REQUIRED,
                steps=None,
                save_summary_per_steps=1,
                name=None,
                hooks=None,
                exporters=None,
                start_delay_secs=120,
                throttle_secs=0):

        self = super(EvalSpec, cls).__new__(
            cls,
            input_fn=functools.partial(
                input_fn,
                batch_size=batch_size,
                mode=tf.estimator.ModeKeys.EVAL),
            steps=steps,
            name=name,
            hooks=hooks,
            exporters=exporters,
            start_delay_secs=start_delay_secs,
            throttle_secs=throttle_secs)

        self.save_summary_per_steps = save_summary_per_steps

        return self


@gin.configurable
class PredictSpec(object):
    def __new__(cls,
                input_fn,
                batch_size=gin.REQUIRED):

        self = super(PredictSpec, cls).__new__(cls)
        self.input_fn = functools.partial(
            input_fn,
            batch_size=batch_size,
            mode=tf.estimator.ModeKeys.PREDICT)

        return self
