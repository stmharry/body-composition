import functools
import tensorflow as tf

from absl import logging
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Conv3DTranspose
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling3D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import MaxPool3D
from tensorflow.keras.layers import ReLU
from tensorflow_addons.layers import InstanceNormalization


def conv_cls(ndim):
    return functools.partial(
        {2: Conv2D, 3: Conv3D}[ndim],
        padding='SAME', use_bias=False)


def conv_transpose_cls(ndim):
    return functools.partial(
        {2: Conv2DTranspose, 3: Conv3DTranspose}[ndim],
        padding='SAME', use_bias=False)


def max_pool_cls(ndim):
    return {
        2: MaxPool2D, 3: MaxPool3D}[ndim]


def global_pool_cls(ndim):
    return {
        2: GlobalAveragePooling2D,
        3: GlobalAveragePooling3D}[ndim]


def central_crop(x, shape):
    crop_shape = tf.shape(x)[1:-1] - shape[1:-1]
    crop_begin = (crop_shape + 1) // 2
    crop_end = shape[1:-1] + crop_begin
    slices = (
        [slice(None)] +
        [slice(*args) for args in zip(
            tf.unstack(crop_begin), tf.unstack(crop_end))] +
        [slice(None)])

    return x[slices]


def _partial_with_signature(cls, cls_name, block_sizes, block_cls, ndim, name):
    class _new_cls(cls):
        def __init__(self,
                     num_classes=None,
                     norm_cls=InstanceNormalization,
                     **kwargs):

            super().__init__(
                num_classes,
                block_sizes=block_sizes,
                filters=64,
                block_cls=block_cls,
                norm_cls=norm_cls,
                ndim=ndim,
                name=name,
                **kwargs)

    _new_cls.__name__ = cls_name
    return _new_cls


class BasicBlock(tf.keras.Model):
    def __init__(self,
                 filters,
                 strides,
                 skip_conv,
                 conv_cls_a,
                 conv_cls_b,
                 norm_cls,
                 name='basic_block'):

        super().__init__(name=name)

        self.skip_conv = skip_conv

        self.norm1 = norm_cls(name='norm1')
        self.relu1 = ReLU(name='relu1')

        if self.skip_conv:
            self.conv = conv_cls_a(
                filters=filters, kernel_size=1, strides=strides, name='conv')

        self.conv1 = conv_cls_a(
            filters=filters, kernel_size=3, strides=strides, name='conv1')

        self.norm2 = norm_cls(name='norm2')
        self.relu2 = ReLU(name='relu2')
        self.conv2 = conv_cls_b(filters=filters, kernel_size=3, name='conv2')

    def call(self, x, training):
        _x = x
        x = self.norm1(x, training=training)
        x = self.relu1(x)

        if self.skip_conv:
            _x = self.conv(x)

        x = self.conv1(x)

        x = self.norm2(x, training=training)
        x = self.relu2(x)
        x = self.conv2(x)

        logging.debug(f'{self.name}: shape = {x.shape}')
        return x + _x


class BottleneckBlock(tf.keras.Model):
    def __init__(self,
                 filters,
                 strides,
                 skip_conv,
                 conv_cls_a,
                 conv_cls_b,
                 norm_cls,
                 name='bottleneck_block'):

        super().__init__(name=name)

        self.skip_conv = skip_conv

        self.norm1 = norm_cls(name='norm1')
        self.relu1 = ReLU(name='relu1')

        if self.skip_conv:
            self.conv = conv_cls_a(
                filters=4 * filters, kernel_size=1, strides=strides, name='conv')

        self.conv1 = conv_cls_b(
            filters=filters, kernel_size=1, strides=1, name='conv1')

        self.norm2 = norm_cls(name='norm2')
        self.relu2 = ReLU(name='relu2')
        self.conv2 = conv_cls_a(
            filters=filters, kernel_size=3, strides=strides, name='conv2')

        self.norm3 = norm_cls(name='norm3')
        self.relu3 = ReLU(name='relu3')
        self.conv3 = conv_cls_b(
            filters=4 * filters, kernel_size=1, strides=1, name='conv3')

    def call(self, x, training):
        _x = x
        x = self.norm1(x, training=training)
        x = self.relu1(x)

        if self.skip_conv:
            _x = self.conv(x)

        x = self.conv1(x)

        x = self.norm2(x, training=training)
        x = self.relu2(x)
        x = self.conv2(x)

        x = self.norm3(x, training=training)
        x = self.relu3(x)
        x = self.conv3(x)

        logging.debug(f'{self.name}: shape = {x.shape}')
        return x + _x


class DownLayer(tf.keras.Model):
    def __init__(self,
                 num_blocks,
                 filters,
                 strides,
                 block_cls,
                 conv_cls,
                 norm_cls,
                 name='down'):

        super().__init__(name=name)

        self._models = [
            block_cls(
                filters=filters,
                strides=(1 if num_block else strides),
                skip_conv=(num_block == 0),
                conv_cls_a=conv_cls,
                conv_cls_b=conv_cls,
                norm_cls=norm_cls,
                name=f'block{num_block}')
            for num_block in range(num_blocks)]

    def call(self, x, training):
        for model in self._models:
            x = model(x, training=training)

        logging.debug(f'{self.name}: shape = {x.shape}')
        return x


class UpLayer(tf.keras.Model):
    def __init__(self,
                 num_blocks,
                 filters,
                 strides,
                 block_cls,
                 conv_cls,
                 conv_transpose_cls,
                 norm_cls,
                 name='up'):

        super().__init__(name=name)

        self._models = [
            block_cls(
                filters=filters,
                strides=(1 if num_block else strides),
                skip_conv=(num_block == 0),
                conv_cls_a=(conv_cls if num_block else conv_transpose_cls),
                conv_cls_b=conv_cls,
                norm_cls=norm_cls,
                name=f'block{num_block}')
            for num_block in range(num_blocks)]

    def call(self, x, training, _x=None):
        for (num_block, model) in enumerate(self._models):
            x = model(x, training=training)

            if (_x is not None) and (num_block == 0):
                x = central_crop(x, shape=tf.shape(_x))
                x = tf.concat([x, _x], axis=-1)

        logging.debug(f'{self.name}: shape = {x.shape}')
        return x


class _ResNet(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 block_sizes,
                 filters,
                 block_cls,
                 norm_cls,
                 ndim=2,
                 name='_resnet'):

        super().__init__(name=name)

        self.num_classes = num_classes
        self.num_layers = len(block_sizes)

        self.conv0 = conv_cls(ndim=ndim)(
            filters=64, kernel_size=7, strides=2, name='conv0')
        self.max_pool0 = max_pool_cls(ndim=ndim)(
            pool_size=3, strides=2, padding='SAME', name='max_pool0')

        self._down_models = [
            DownLayer(
                num_blocks=block_sizes[num_layer],
                filters=(filters * 2 ** num_layer),
                strides=(2 if num_layer else 1),
                block_cls=block_cls,
                conv_cls=conv_cls(ndim=ndim),
                norm_cls=norm_cls,
                name=f'down{num_layer + 1}')
            for num_layer in range(self.num_layers)]


class ResNet(_ResNet):
    def __init__(self,
                 num_classes=None,
                 block_sizes=[],
                 filters=64,
                 block_cls=BasicBlock,
                 norm_cls=InstanceNormalization,
                 pooling=True,
                 include_top=True,
                 ndim=2,
                 name='resnet'):

        super().__init__(
            num_classes=num_classes,
            block_sizes=block_sizes,
            filters=filters,
            block_cls=block_cls,
            norm_cls=norm_cls,
            ndim=ndim,
            name=name)

        self.pooling = pooling
        self.include_top = include_top

        self.norm1 = norm_cls(name='norm1')
        self.relu1 = ReLU(name='relu1')
        if pooling:
            self.avg_pool1 = global_pool_cls(ndim=ndim)(name='avg_pool1')
        if include_top:
            self.fc1 = Dense(units=num_classes, name='fc1')

    def call(self, x, training):
        x = self.conv0(x)
        x = self.max_pool0(x)

        for model in self._down_models:
            x = model(x, training=training)

        x = self.norm1(x, training=training)
        x = self.relu1(x)

        if self.pooling:
            x = self.avg_pool1(x)
        if self.include_top:
            x = self.fc1(x)

        return x


ResNet18 = _partial_with_signature(
    ResNet, 'ResNet18',
    block_sizes=[2, 2, 2, 2], block_cls=BasicBlock, ndim=2, name='resnet18')

ResNet34 = _partial_with_signature(
    ResNet, 'ResNet34',
    block_sizes=[3, 4, 6, 3], block_cls=BasicBlock, ndim=2, name='resnet34')

ResNet50 = _partial_with_signature(
    ResNet, 'ResNet50',
    block_sizes=[3, 4, 6, 3], block_cls=BottleneckBlock, ndim=2, name='resnet50')

ResNet18_3D = _partial_with_signature(
    ResNet, 'ResNet18_3D',
    block_sizes=[2, 2, 2, 2], block_cls=BasicBlock, ndim=3, name='resnet18')

ResNet34_3D = _partial_with_signature(
    ResNet, 'ResNet34_3D',
    block_sizes=[3, 4, 6, 3], block_cls=BasicBlock, ndim=3, name='resnet34')

ResNet50_3D = _partial_with_signature(
    ResNet, 'ResNet50_3D',
    block_sizes=[3, 4, 6, 3], block_cls=BottleneckBlock, ndim=3, name='resnet50')


class ResUNet(_ResNet):
    def __init__(self,
                 num_classes,
                 block_sizes,
                 filters=64,
                 block_cls=BasicBlock,
                 norm_cls=InstanceNormalization,
                 ndim=2,
                 name='resunet'):

        super().__init__(
            num_classes=num_classes,
            block_sizes=block_sizes,
            filters=filters,
            block_cls=block_cls,
            norm_cls=norm_cls,
            ndim=ndim,
            name=name)

        self._up_models = [
            UpLayer(
                num_blocks=1,
                filters=(filters * 2 ** num_layer),
                strides=2,
                block_cls=block_cls,
                conv_cls=conv_cls(ndim=ndim),
                conv_transpose_cls=conv_transpose_cls(ndim=ndim),
                norm_cls=norm_cls,
                name=f'up{num_layer + 1}')
            for num_layer in reversed(range(self.num_layers - 1))]

        self._up_models.append(
            UpLayer(
                num_blocks=1,
                filters=filters,
                strides=2,
                block_cls=BasicBlock,
                conv_cls=conv_cls(ndim=ndim),
                conv_transpose_cls=conv_transpose_cls(ndim=ndim),
                norm_cls=norm_cls,
                name='up0'))

        self.norm1 = norm_cls(name='norm1')
        self.relu1 = ReLU(name='relu1')
        self.conv1 = conv_transpose_cls(ndim=ndim)(
            filters=num_classes,
            kernel_size=3,
            strides=2,
            use_bias=True,
            name='conv1')

    def call(self, x, training):
        shape = tf.shape(x)

        x2 = self.conv0(x)
        x4 = self.max_pool0(x2)

        xs = [x2]
        x = x4
        for model in self._down_models:
            x = model(x, training=training)
            xs.append(x)

        x = xs.pop(-1)
        for (model, _x) in zip(self._up_models, reversed(xs)):
            x = model(x, _x=_x, training=training)

        x = self.norm1(x, training=training)
        x = self.relu1(x)
        x = self.conv1(x)
        x = central_crop(x, shape=shape)

        return x


ResUNet18 = _partial_with_signature(
    ResUNet, 'ResUNet18',
    block_sizes=[2, 2, 2, 2], block_cls=BasicBlock, ndim=2, name='resunet18')

ResUNet34 = _partial_with_signature(
    ResUNet, 'ResUNet34',
    block_sizes=[3, 4, 6, 3], block_cls=BasicBlock, ndim=2, name='resunet34')

ResUNet50 = _partial_with_signature(
    ResUNet, 'ResUNet50',
    block_sizes=[3, 4, 6, 3], block_cls=BottleneckBlock, ndim=2, name='resunet50')

ResUNet18_3D = _partial_with_signature(
    ResUNet, 'ResUNet18_3D',
    block_sizes=[2, 2, 2, 2], block_cls=BasicBlock, ndim=3, name='resunet18')

ResUNet34_3D = _partial_with_signature(
    ResUNet, 'ResUNet34_3D',
    block_sizes=[3, 4, 6, 3], block_cls=BasicBlock, ndim=3, name='resunet34')

ResUNet50_3D = _partial_with_signature(
    ResUNet, 'ResUNet50_3D',
    block_sizes=[3, 4, 6, 3], block_cls=BottleneckBlock, ndim=3, name='resunet50')
