import gin
import math
import tensorflow as tf
import tensorflow_addons as tfa

from thin.utils import namedtuple


class Shape(object):
    def __init__(self, tensor, dtype=tf.int32):
        self._static_shape = tensor.shape
        self._shape = tf.shape(tensor)
        self._dtype = dtype
        self._ndims = tensor.shape.ndims  # 3 for 2D, 4 for 3D

    def _get_dim(self, ndim):
        return self._static_shape[ndim] or tf.cast(self._shape[ndim], self._dtype)

    @property
    def is_3d(self):
        return (self._ndims == 4)

    @property
    def depth(self):
        if self.is_3d:
            return self._get_dim(self._ndims - 4)
        else:
            return None

    @property
    def height(self):
        return self._get_dim(self._ndims - 3)

    @property
    def width(self):
        return self._get_dim(self._ndims - 2)

    @property
    def channels(self):
        return self._get_dim(self._ndims - 1)


@gin.configurable
class MinMaxNormalize(
    namedtuple('MinMaxNormalize', {
        'minval': gin.REQUIRED,
        'maxval': gin.REQUIRED})):

    def __call__(self, tensor):
        return tf.clip_by_value(
            (tensor - self.minval) / (self.maxval - self.minval), 0, 1)


@gin.configurable
class MeanStdNormalize(
    namedtuple('MeanStdNormalize', {
        'mean': 0.0,
        'std': 1.0})):

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


class _Resize(object):
    def __call__(self, tensor, ratio, method):
        if ratio == 1.0:
            return tensor
        else:
            shape = Shape(tensor, dtype=tf.float32)
            size = tf.cast(tf.stack([
                tf.round(ratio * shape.height),
                tf.round(ratio * shape.width)]), tf.int32)

            return tf.image.resize(tensor, size, method)


@gin.configurable
class Resize(
    namedtuple('Resize', {
        'ratio': 1.0,
        'method': tf.image.ResizeMethod.BILINEAR}), _Resize):

    def __call__(self, tensor):
        return super().__call__(
            tensor, ratio=self.ratio, method=self.method)


@gin.configurable
class ResizeTo(
    namedtuple('ResizeTo', {
        'height': None,
        'width': None,
        'method': tf.image.ResizeMethod.BILINEAR}), _Resize):

    def __call__(self, tensor):
        assert (self.height is not None) or (self.width is not None)

        shape = Shape(tensor, dtype=tf.float32)
        if self.height is not None:
            ratio = self.height / shape.height
        elif self.width is not None:
            ratio = self.width / shape.width

        return super().__call__(
            tensor, ratio=ratio, method=self.method)


@gin.configurable
class CropToMultiple(
    namedtuple('CropToMultiple', {
        'size': gin.REQUIRED})):

    def __call__(self, tensor):
        shape = Shape(tensor, dtype=tf.float32)
        assert not shape.is_3d

        target_height = tf.cast(
            tf.math.floor(shape.height / self.size) * self.size, tf.int32)

        target_width = tf.cast(
            tf.math.floor(shape.width / self.size) * self.size, tf.int32)

        tensor = tf.image.resize_with_crop_or_pad(
            tensor, target_height=target_height, target_width=target_width)

        return tensor


@gin.configurable
class PadToSize(
    namedtuple('PadToSize', {
        'size': gin.REQUIRED})):

    def __call__(self, tensor):
        shape = Shape(tensor, dtype=tf.int32)
        assert not shape.is_3d

        pad_height = self.size - shape.height
        pad_width = self.size - shape.width

        offset_height = pad_height // 2
        offset_width = pad_width // 2

        tensor = tf.image.pad_to_bounding_box(
            tensor, offset_height, offset_width, self.size, self.size)

        return tensor


@gin.configurable
class RandomContrast(
    namedtuple('RandomContrast', {
        'delta': 0.0})):

    def __call__(self, tensor):
        if not self.delta:
            return tensor

        return tf.image.random_contrast(
            tensor, 1 - self.delta, 1 + self.delta)


@gin.configurable
class RandomBrightness(
    namedtuple('RandomBrightness', {
        'delta': 0.0})):

    def __call__(self, tensor):
        if not self.delta:
            return tensor

        return tf.image.random_brightness(tensor, self.delta)


@gin.configurable
class RandomRotate(
    namedtuple('RandomRotate', {
        'degrees_delta': 0.0,
        'method': 'NEAREST'})):

    def __call__(self, tensor):
        if not self.degrees_delta:
            return tensor

        shape = Shape(tensor, dtype=tf.float32)

        radian_delta = math.radians(self.degrees_delta)
        return tfa.image.transform(
            tensor,
            transforms=tfa.image.transform_ops.angles_to_projective_transforms(
                tf.random.uniform((), -radian_delta, radian_delta),
                shape.height, shape.width),
            interpolation=self.method)


@gin.configurable
class RandomCrop(
    namedtuple('RandomCrop', {
        'height': None,
        'width': None,
        'size': None})):

    def __call__(self, tensor):
        if self.size is None:
            height = self.height
            width = self.width
        else:
            height = width = self.size

        shape = Shape(tensor)

        new_shape = [height, width, shape.channels]
        if shape.is_3d:
            new_shape.insert(0, shape.depth)

        new_shape = tf.cast(tf.stack(new_shape), tf.int32)
        return tf.image.random_crop(tensor, new_shape)


@gin.configurable
class RandomSelectSlices(
    namedtuple('RandomSelectSlices', {
        'max_slices': gin.REQUIRED})):

    def __call__(self, tensor, *tensors):
        length = tf.shape(tensor)[0]

        num_slices = tf.minimum(length, self.max_slices)
        max_begin = length - num_slices + 1
        begin = tf.random.uniform((), maxval=max_begin, dtype=tf.int32)

        index = tf.range(begin, begin + num_slices, dtype=tf.int32)

        return tuple([
            tf.gather(_tensor, index, axis=0)
            for _tensor in [tensor] + list(tensors)])
