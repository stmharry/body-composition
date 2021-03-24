import functools
import gin
import glob
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import nibabel as nib
import numpy as np
import os
import pandas as pd
import pdfcombine
import scipy.interpolate
import scipy.ndimage
import scipy.optimize
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
import tqdm
import warnings

from absl import app
from absl import flags
from absl import logging
from attr import attrs
from attr import attrib

from thin import estimator
from thin import main
from thin.scripts import as_numpy_func
from thin.scripts import as_tensorflow_func
from thin.scripts import with_input
from thin.scripts import with_output
from thin.utils import maybe_decode

flags.DEFINE_string(
    'predict_path', None, 'Path to find studies for prediction.')
flags.DEFINE_string(
    'result_dir', None, 'Directory to output results. Blank to use default.')
FLAGS = flags.FLAGS


def sigmoid(z, z0=0, t=1, clip=10):
    exponent = np.clip(-(z - z0) / (t + 1e-5), -clip, clip)
    return 1 / (1 + np.exp(exponent))


@attrs
class InputFn(estimator.InputFn):
    resample_key = attrib(default=None)
    train_augmentation = attrib(default=None)
    test_augmentation = attrib(default=None)
    summarize_image = attrib(default=False)

    parse_study_as_batch = None
    image_shape = (512, 512)
    image_channels = 1

    @property
    def split_dir(self):
        return os.path.join(self.split_dir_root, self.split)

    def split_csv_path(self, training):
        name = {True: 'train', False: 'eval'}[training]
        return os.path.join(self.split_dir, f'{name}.txt')

    def image_path(self, study_name):
        return os.path.join(self.data_dir, 'studies', f'{study_name}.nii.gz')

    def mask_path(self, study_name):
        return os.path.join(self.data_dir, 'masks', f'{study_name}.nii.gz')

    @property
    def tfrecord_dir(self):
        return os.path.join(self.tfrecord_dir_root, self.split)

    def tfrecord_path(self, study_name):
        return os.path.join(self.tfrecord_dir, f'{study_name}.tfrecord')

    def _class_to_color(self, class_):
        if self.num_classes <= 10:
            colormap = cm.tab10(
                np.arange(self.num_classes))
        else:
            colormap = cm.gist_stern(
                np.arange(self.num_classes) / self.num_classes)

        return tf.gather(colormap, class_, axis=0)

    @with_input('image_path')
    @with_input('mask_path', dtype=tf.string, default='')
    @with_output('name', dtype=tf.string, shape=())
    @with_output('image',
                 dtype=tf.float32,
                 shape=np.r_[None, image_shape, image_channels])
    @with_output('mask', dtype=tf.int64, shape=np.r_[None, image_shape])
    @with_output('dim', dtype=tf.float32, shape=(3,))
    @with_output('shape', dtype=tf.int64, shape=(3,))
    @with_output('affine', dtype=tf.float32, shape=(4, 4))
    @with_output('index_l3', dtype=tf.int64, shape=())
    def _parse_study(self, image_path, mask_path):
        name = os.path.basename(maybe_decode(image_path)).split('.')[0]
        logging.info(f'Reading study {name}')

        image_nib = nib.load(maybe_decode(image_path))
        image_nib = nib.as_closest_canonical(image_nib)

        image = np.asarray(image_nib.dataobj, dtype=np.float32)
        image = image.transpose(2, 1, 0)[..., None]
        shape = np.asarray(image.shape, dtype=np.int64)[:3]

        affine = np.asarray(image_nib.affine, dtype=np.float32)
        affine = affine[np.ix_([2, 1, 0, 3], [2, 1, 0, 3])]
        dim = np.abs(np.diag(affine))[[0, 1, 2]]

        if mask_path:
            mask_nib = nib.load(maybe_decode(mask_path))
            mask_nib = nib.as_closest_canonical(mask_nib)
            mask = np.asarray(mask_nib.dataobj, dtype=np.int64)
            mask = mask.transpose(2, 1, 0)

            mask_l3 = np.isin(mask, [1, 2, 3])
            mask_l3_z = np.sum(mask_l3, axis=(1, 2))
            index_l3 = np.argmax(mask_l3_z)

            if np.sum(mask_l3_z != 0) > 1:
                indices_l3 = np.argwhere(mask_l3_z).flatten()
                logging.warning((
                    f'More than one slice with annotations: {indices_l3}. '
                    f'Using {index_l3}.'))

        else:
            mask = np.zeros(
                (shape[0],) + InputFn.image_shape,
                dtype=np.int64)
            index_l3 = np.asarray(0, dtype=np.int64)

        return (name, image, mask, dim, shape, affine, index_l3)

    def _write_tfrecord(self, study, tfrecord_path):
        logging.info(f'Caching TFRecord into {tfrecord_path}')

        if self.parse_study_as_batch:
            tfrecords = [
                dict(zip(study.keys(), values))
                for values in zip(*study.values())]
        else:
            tfrecords = [study]

        super()._write_tfrecord(
            tfrecords=tfrecords, tfrecord_path=tfrecord_path)

    @with_input('image', dtype=tf.float32)
    @with_input('mask', dtype=np.int64, default=())
    @with_output('image', dtype=tf.float32, shape=(None, None, image_channels))
    @with_output('mask', dtype=tf.int64, shape=(None, None))
    def _augment(self, image, mask, training):
        augmentation = {
            True: self.train_augmentation,
            False: self.test_augmentation}[training]

        images = [image.astype(np.float32)]

        if mask.size == 0:
            masks = None
        else:
            masks = [
                np.expand_dims(mask.astype(np.int32), axis=-1)]

        (images, masks) = augmentation(
            images=images, segmentation_maps=masks)

        image = images[0].astype(np.float32)
        if mask.size == 0:
            mask = np.zeros((0, 0), dtype=np.int64)
        else:
            mask = np.squeeze(
                masks[0].astype(np.int64), axis=-1)

        return (image, mask)

    def _preprocess(self, example, training):
        image = example['image']
        example['raw_image'] = image
        example['image'] = tf.maximum(image / 1024, -1)

        example = as_tensorflow_func(
            self._augment, training=training, include_inputs=True)(example)

        return self._example_to_features_and_labels(
            example, label_keys=self.label_keys)

    def _input_fn_train_or_eval(self, training, batch_size=None):
        study_names = sorted(np.loadtxt(
            self.split_csv_path(training=training), dtype=np.str, ndmin=1))

        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)

        tfrecord_paths = []
        for study_name in tqdm.tqdm(
                study_names, desc='Loading tfrecord files'):
            tfrecord_path = self.tfrecord_path(study_name)
            tfrecord_paths.append(tfrecord_path)

            if os.path.isfile(tfrecord_path):
                logging.info(f'Using cached TFRecord {tfrecord_path}')
                continue

            study = as_numpy_func(self._parse_study, include_inputs=False)({
                'image_path': self.image_path(study_name),
                'mask_path': self.mask_path(study_name)})

            self._write_tfrecord(study, tfrecord_path)

        dataset = tf.data.Dataset.from_tensor_slices(tfrecord_paths)
        if training:
            dataset = dataset.shuffle(len(tfrecord_paths))
            dataset = dataset.repeat()

        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=tf.data.experimental.AUTOTUNE,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(
            self._parse_tfrecord,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(
            functools.partial(self._preprocess, training=training),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self.resample_key and training:
            class_func = lambda features, labels: labels[self.resample_key]
            target_dist = np.full(
                (self.num_classes,), 1 / self.num_classes, dtype=np.float32)
            dataset = dataset.apply(
                tf.data.experimental.rejection_resample(
                    class_func=class_func, target_dist=target_dist))
            dataset = dataset.map(lambda label, example: example)

        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        if batch_size:
            dataset = dataset.batch(batch_size=batch_size)

        return dataset

    def _input_fn_predict(self, training=False, batch_size=None):
        df = pd.DataFrame([
            {'image_path': path}
            for path in sorted(glob.glob(FLAGS.predict_path))
            if path.endswith('nii.gz')])

        dataset = tf.data.Dataset.from_tensor_slices(dict(df.items()))
        dataset = dataset.map(
            as_tensorflow_func(self._parse_study, include_inputs=True),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self.parse_study_as_batch:
            dataset = dataset.unbatch()

        dataset = dataset.map(
            functools.partial(self._preprocess, training=training),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        if batch_size:
            dataset = dataset.batch(batch_size=batch_size)

        return dataset

    def _predict_batch(self, groupby='name'):
        predictions = super().predict()

        current_name = None
        _predictions = []
        for prediction in predictions:
            name = prediction[groupby].decode()

            if (name != current_name) and len(_predictions):
                yield (current_name, pd.DataFrame(_predictions))
                _predictions = []

            current_name = name
            _predictions.append(prediction)

        yield (current_name, pd.DataFrame(_predictions))


@gin.configurable
@attrs
class FirstStage(InputFn, estimator.Estimator):
    split = attrib(default='first')
    num_classes = attrib(default=2)
    model_cls = attrib(default=None)

    z_step = attrib(default=0.25)
    t_dim = attrib(default=0.01)
    t_dim_max = attrib(default=2.0)
    z_dim_interp = attrib(default=0.1)
    z_dim_filter = attrib(default=25.0)
    prob_sample = attrib(default=10000)
    hist_thresh = attrib(default=0.1)
    sigma = attrib(default=1)
    topk = attrib(default=5)

    parse_study_as_batch = True
    label_keys = ['index_l3', 'label']

    @with_input('image_path')
    @with_input('mask_path', dtype=tf.string, default='')
    @with_output('image_path', dtype=tf.string, shape=(None,))
    @with_output('mask_path', dtype=tf.string, shape=(None,))
    @with_output('name', dtype=tf.string, shape=(None,))
    @with_output('image',
                 dtype=tf.float32,
                 shape=np.r_[None, InputFn.image_shape, InputFn.image_channels])
    @with_output('dim', dtype=tf.float32, shape=(None, 3))
    @with_output('shape', dtype=tf.int64, shape=(None, 2))
    @with_output('affine', dtype=tf.float32, shape=(None, 4, 4))
    @with_output('index', dtype=tf.int64, shape=(None,))
    @with_output('index_l3', dtype=tf.int64, shape=(None,))
    def _parse_study(self, image_path, mask_path):
        (name, image, mask, dim, shape, affine, index_l3) = \
            super()._parse_study(image_path, mask_path)
        z_len = shape[0]

        return (
            np.repeat(image_path, z_len),
            np.repeat(mask_path, z_len),
            np.repeat(name, z_len),
            image,
            np.repeat(dim[None, ...], z_len, axis=0),
            np.repeat(shape[None, 1:3], z_len, axis=0),
            np.repeat(affine[None, ...], z_len, axis=0),
            np.arange(z_len, dtype=np.int64),
            np.repeat(index_l3, z_len))

    def _parse_tfrecord(self, tfrecord):
        features = {
            'name': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            'image': tf.io.VarLenFeature(dtype=tf.float32),
            'dim': tf.io.FixedLenFeature(shape=[3], dtype=tf.float32),
            'shape': tf.io.FixedLenFeature(shape=[2], dtype=tf.int64),
            'index': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            'index_l3': tf.io.FixedLenFeature(shape=[], dtype=tf.int64)}

        example = tf.io.parse_single_example(tfrecord, features)
        shape = example['shape']

        image = tf.reshape(
            tf.sparse.to_dense(example['image']),
            tf.concat([shape, [self.image_channels]], axis=0))

        example.update({'image': image})

        return example

    def _preprocess(self, example, training, tau=20):
        (features, labels) = super()._preprocess(example, training=training)

        dim = features['dim']
        index = features['index']

        index_l3 = labels.get('index_l3')
        if index_l3 is not None:
            dist = tf.cast(index - index_l3, tf.float32) * dim[0]

            target = tf.sigmoid(dist / tau)
            label = tf.clip_by_value(
                tf.floor(target * self.num_classes), 0, self.num_classes - 1)
            label = tf.cast(label, tf.int64)

            labels['label'] = label

        return (features, labels)

    def model_fn(self, features, labels, mode):
        image = features['image']
        training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = self.model_cls(num_classes=self.num_classes)
        logit = model(image, training=training)
        prob = tf.nn.softmax(logit, axis=-1)

        predictions = {
            'logit': logit,
            'prob': prob,
            **features}

        if mode == tf.estimator.ModeKeys.PREDICT:
            return self._build_estimator_spec(
                mode=mode, predictions=predictions)

        self._register_model_updates(model, features)

        label = labels['label']
        model_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=label, logits=logit))

        return self._build_estimator_spec(
            mode=mode, model_loss=model_loss, predictions=predictions)

    def predict(self):
        result_dir = FLAGS.result_dir or self.create_dir(self.result_dir_root)
        os.path.isdir(result_dir) or os.makedirs(result_dir)

        df_out = []
        for (name, df) in self._predict_batch(groupby='name'):
            study_name = os.path.basename(name).split('.')[0]

            z_size = len(df)
            z_dim = df.dim.iloc[0][0]
            image = np.stack(df.raw_image, axis=0).squeeze(axis=-1)
            _logit = np.stack(df.logit, axis=0)

            z1 = np.arange(0, z_size, step=1.0)
            _prob1 = sigmoid(_logit[:, 1] - _logit[:, 0])
            f = scipy.interpolate.interp1d(
                z1, _prob1, fill_value='extrapolate')

            # interpolate w.r.t. z
            z2 = np.arange(0, z_size, step=self.z_dim_interp / z_dim)
            _prob2 = f(z2)

            # interpolate w.r.t. prob
            _prob_diff = np.r_[0, np.diff(_prob2)]
            _prob_unrolled = _prob2[0] + np.cumsum(np.abs(_prob_diff))

            g = scipy.interpolate.interp1d(
                _prob_unrolled, z2, fill_value='extrapolate')
            z3 = g(np.linspace(
                _prob_unrolled.min(), _prob_unrolled.max(), self.prob_sample))

            # interpolate w.r.t. z
            _prob3 = f(z3)

            z0 = np.arange(0, z_size, step=self.z_step)
            t = np.arange(0, self.t_dim_max, step=self.t_dim)
            t_edge = np.r_[-self.t_dim, t] + self.t_dim / 2

            _arr = -(z3[None, :] - z0[:, None]) / np.log(1 / _prob3 - 1)
            _hist = np.stack(np.apply_along_axis(
                np.histogram, axis=1, arr=_arr, bins=t_edge)[:, 0])
            _hist = _hist + 1e-5 * np.random.uniform(size=_hist.shape)  # to avoid equal-heighted peaks
            _hist = scipy.ndimage.gaussian_filter(
                _hist, sigma=(self.sigma / z_dim / self.z_step, 5))
            _hist_max = scipy.ndimage.maximum_filter(
                _hist, size=(
                    int(self.z_dim_filter / z_dim / self.z_step), len(t)))

            # find the local maximums
            _index = np.nonzero(_hist == _hist_max)
            # only keep top k non-zero local maximums
            _index_topk = np.argsort(-_hist[_index])[:self.topk]
            _index_topk = _index_topk[
                _hist[_index][_index_topk] > _hist_max.max() * self.hist_thresh]

            (_index_l3, _t_index) = [_i[_index_topk] for _i in _index]
            _z0_topk = z0[_index_l3]
            _z0_topk_int = np.clip(
                _z0_topk.round().astype(np.int), 0, z_size - 1)
            _t_topk = t[_t_index]
            _h_topk = _hist[(_index_l3, _t_index)]

            logging.info(
                (
                    f'Post-processing {study_name}, '
                    f'L3 slice index proposals: ')
                + ', '.join([
                    f'({_z0:.1f}±{_t:.1f})'
                    for (_z0, _t) in zip(_z0_topk, _t_topk)]))

            # Save .nii for each slice candidate
            for (num, _z0_int) in enumerate(_z0_topk_int):
                _image = image[[_z0_int]].transpose(2, 1, 0)
                _affine = df.affine[0][np.ix_([2, 1, 0, 3], [2, 1, 0, 3])]
                _image_nib = nib.Nifti1Image(_image, affine=_affine)
                nib.save(
                    _image_nib,
                    os.path.join(result_dir, f'{study_name}-{_z0_int}.nii.gz'))

            # Plot
            cmap = cm.get_cmap('tab10')

            plt.close()
            (fig, axes) = plt.subplots(
                nrows=1, ncols=3, figsize=(10, 6),
                gridspec_kw={'width_ratios': [2.5, 0.75, 0.75]})
            ax = axes[0]
            ax.imshow(
                image[:, ::-1, 256], cmap='bone', aspect='auto', origin='lower')
            for (num, (_z0, _h)) in enumerate(zip(_z0_topk, _h_topk)):
                ax.axhline(_z0, c=cmap(num))
                ax.text(
                    x=0, y=_z0, s=f'Slice #{_z0:.0f} ({_h:.1f})',
                    bbox=dict(facecolor='w', alpha=0.5))
            ax.set_xticks([])
            ax.set_yticks([])

            ax = axes[1]
            for (num, (_z0, _t)) in enumerate(zip(_z0_topk, _t_topk)):
                ax.plot(
                    sigmoid(z2, z0=_z0, t=_t),
                    z2, c=cmap(num), linestyle=':', linewidth=1.0)
            ax.plot(_prob2, z2, c='b', linewidth=2.0)
            ax.axvline(0, c='k', ls=':')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, z_size)
            ax.grid(True)
            ax.set_yticks([])
            ax.set_xlabel('Probability')
            ax.set_title('Slice Prediction')

            ax = axes[2]
            ax.imshow(
                _hist, cmap='jet', aspect='auto', origin='lower',
                extent=(t_edge.min(), t_edge.max(), -0.5, z_size - 0.5))
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')
            ax.set_xlabel('Width (mm)')
            ax.set_ylabel('Slice #')
            ax.set_title('Hough Transform')

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.suptitle(f'Study {study_name}')

            pdf_path = os.path.join(result_dir, f'{study_name}.pdf')
            fig.savefig(pdf_path)
            fig.savefig(os.path.join(result_dir, f'{study_name}.png'), dpi=320)

            df_out.extend([{
                'study_name': study_name,
                'path': pdf_path,
                'index_l3': _z0_int,
                'index': _z0,
                'index_std': _t,
                'strength': _h}
                for (_z0, _z0_int, _t, _h) in zip(
                    _z0_topk, _z0_topk_int, _t_topk, _h_topk)])

        df_out = pd.DataFrame(df_out)
        paths = df_out.pop('path').drop_duplicates().tolist()
        df_out.to_csv(
            os.path.join(result_dir, 'stat.csv'),
            float_format='%.2f', index=False)

        pdfcombine.combine(paths, os.path.join(result_dir, 'all.pdf'))


@gin.configurable
@attrs
class SecondStage(InputFn, estimator.Estimator):
    split = attrib(default='second')
    num_classes = attrib(default=4)
    model_cls = attrib(default=None)

    parse_study_as_batch = False
    label_keys = ['mask']

    @with_input('image_path')
    @with_input('mask_path', dtype=tf.string, default='')
    @with_output('name', dtype=tf.string, shape=())
    @with_output('image',
                 dtype=tf.float32,
                 shape=InputFn.image_shape + (InputFn.image_channels,))
    @with_output('mask', dtype=tf.int64, shape=InputFn.image_shape)
    @with_output('dim', dtype=tf.float32, shape=(3,))
    @with_output('shape', dtype=tf.int64, shape=(2,))
    @with_output('affine', dtype=tf.float32, shape=(4, 4))
    def _parse_study(self, image_path, mask_path):
        (name, image, mask, dim, shape, affine, index_l3) = \
            super()._parse_study(image_path, mask_path)

        return (
            name,
            image[index_l3],
            mask[index_l3],
            dim,
            shape[1:3],
            affine)

    def _parse_tfrecord(self, tfrecord):
        features = {
            'name': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            'image': tf.io.VarLenFeature(dtype=tf.float32),
            'mask': tf.io.VarLenFeature(dtype=tf.int64),
            'dim': tf.io.FixedLenFeature(shape=[3], dtype=tf.float32),
            'shape': tf.io.FixedLenFeature(shape=[2], dtype=tf.int64)}

        example = tf.io.parse_single_example(tfrecord, features)
        shape = example['shape']

        image = tf.reshape(
            tf.sparse.to_dense(example['image']),
            tf.concat([shape, [self.image_channels]], axis=0))
        mask = tf.reshape(
            tf.sparse.to_dense(example['mask']),
            shape)

        example.update({'image': image, 'mask': mask})

        return example

    def model_fn(self, features, labels, mode):
        image = features['image']
        training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = self.model_cls(num_classes=self.num_classes)
        logit = model(image, training=training)
        prob = tf.nn.softmax(logit, axis=-1)
        mask_pred = tf.argmax(logit, axis=-1)

        predictions = {
            'logit': logit,
            'prob': prob,
            'mask_pred': mask_pred,
            **features}

        if mode == tf.estimator.ModeKeys.PREDICT:
            return self._build_estimator_spec(
                mode=mode, predictions=predictions)

        self._register_model_updates(model, features)

        mask = labels['mask']
        model_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=mask, logits=logit))

        if self.summarize_image:
            tf_v1.summary.image('image', image)
            tf_v1.summary.image('mask', self._class_to_color(mask))
            tf_v1.summary.image('mask_pred', self._class_to_color(mask_pred))

        return self._build_estimator_spec(
            mode=mode, model_loss=model_loss, predictions=predictions)

    def predict(self):
        result_dir = FLAGS.result_dir or self.create_dir(self.result_dir_root)
        os.path.isdir(result_dir) or os.makedirs(result_dir)

        df_out = []
        for item in super().predict():
            item = pd.Series(item)

            study_name = maybe_decode(item['name'])

            image = item.raw_image
            shape = item['shape']
            affine = item.affine
            image = np.expand_dims(item.image, axis=0).squeeze(axis=-1)
            mask = np.expand_dims(item.mask_pred, axis=0)

            logging.info(f'Post-processing {study_name}')

            # Save .nii
            _image = mask.astype(np.uint8).transpose(2, 1, 0)
            _affine = affine[np.ix_([2, 1, 0, 3], [2, 1, 0, 3])]
            _image_nib = nib.Nifti1Image(_image, affine=_affine)
            nib.save(
                _image_nib, os.path.join(result_dir, f'{study_name}.nii.gz'))

            _area = np.prod(_image_nib.header.get_zooms()[:2]) / 100
            area = np.bincount(mask.flatten(), minlength=4) * _area
            s = (
                f'Muscle: {area[1]:.1f} cm$^2$\n'
                f'Subcutaneous Fat: {area[2]:.1f} cm$^2$\n'
                f'Visceral Fat: {area[3]:.1f} cm$^2$')

            # Plot
            plt.close()
            (fig, axes) = plt.subplots(1, 2, figsize=(12, 6))
            ax = axes[0]
            ax.imshow(
                image[0], cmap='bone', aspect='auto', origin='lower',
                extent=np.r_[0, shape[0], 0, shape[1]] - 0.5)
            ax.set_xticks([])
            ax.set_yticks([])

            color = np.r_[[[0, 0, 0, 1]], cm.Set1([0, 1, 2])]

            ax = axes[1]
            ax.imshow(
                color[mask[0]], aspect='equal', origin='lower',
                interpolation='none',
                extent=np.r_[0, shape[0], 0, shape[1]] - 0.5)
            ax.text(
                x=shape[0], y=0, s=s,
                horizontalalignment='right', verticalalignment='bottom',
                bbox=dict(facecolor='w', alpha=0.75))
            ax.set_xticks([])
            ax.set_yticks([])

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.suptitle(f'Study {study_name}')

            pdf_path = os.path.join(result_dir, f'{study_name}.pdf')
            fig.savefig(pdf_path)
            fig.savefig(os.path.join(result_dir, f'{study_name}.png'), dpi=320)

            # Statistics
            df_out.append({
                'study_name': study_name,
                'path': pdf_path,
                'muscle': area[1],
                'subcutaneous_fat': area[2],
                'visceral_fat': area[3]})

        df_out = pd.DataFrame(df_out)
        paths = df_out.pop('path').drop_duplicates().tolist()
        df_out.to_csv(os.path.join(result_dir, 'stat.csv'), index=False)

        pdfcombine.combine(paths, os.path.join(result_dir, 'all.pdf'))


if __name__ == '__main__':
    plt.rc('font', family='Arial', size=12)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    warnings.filterwarnings('ignore')

    for (iaa_object_name, iaa_object) in iaa.__dict__.items():
        if (isinstance(iaa_object, type) and
                issubclass(iaa_object, iaa.Augmenter)):
            gin.external_configurable(iaa_object, module='iaa')

    app.run(main)
