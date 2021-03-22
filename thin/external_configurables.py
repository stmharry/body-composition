import gin

from tensorflow_addons.layers import GroupNormalization
from tensorflow_addons.layers import InstanceNormalization

gin.config.external_configurable(GroupNormalization, module='tf.keras.layers')
gin.config.external_configurable(InstanceNormalization, module='tf.keras.layers')
