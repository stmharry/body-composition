import gin

from tensorflow.keras import applications
from tensorflow.compat.v1.keras.layers import BatchNormalization

from thin.models import resnet

EfficientNetB0 = gin.configurable(applications.EfficientNetB0, module='tf.keras.applications')
EfficientNetB1 = gin.configurable(applications.EfficientNetB1, module='tf.keras.applications')
EfficientNetB2 = gin.configurable(applications.EfficientNetB2, module='tf.keras.applications')
EfficientNetB3 = gin.configurable(applications.EfficientNetB3, module='tf.keras.applications')
EfficientNetB4 = gin.configurable(applications.EfficientNetB4, module='tf.keras.applications')
EfficientNetB5 = gin.configurable(applications.EfficientNetB5, module='tf.keras.applications')
EfficientNetB6 = gin.configurable(applications.EfficientNetB6, module='tf.keras.applications')
EfficientNetB7 = gin.configurable(applications.EfficientNetB7, module='tf.keras.applications')

ResNet18 = gin.configurable(resnet.ResNet18, module='thin.models')
ResNet34 = gin.configurable(resnet.ResNet34, module='thin.models')
ResNet50 = gin.configurable(resnet.ResNet50, module='thin.models')
ResUNet18 = gin.configurable(resnet.ResUNet18, module='thin.models')
ResUNet34 = gin.configurable(resnet.ResUNet34, module='thin.models')
ResUNet50 = gin.configurable(resnet.ResUNet50, module='thin.models')

ResNet18_3D = gin.configurable(resnet.ResNet18_3D, module='thin.models')
ResNet34_3D = gin.configurable(resnet.ResNet34_3D, module='thin.models')
ResNet50_3D = gin.configurable(resnet.ResNet50_3D, module='thin.models')
ResUNet18_3D = gin.configurable(resnet.ResUNet18_3D, module='thin.models')
ResUNet34_3D = gin.configurable(resnet.ResUNet34_3D, module='thin.models')
ResUNet50_3D = gin.configurable(resnet.ResUNet50_3D, module='thin.models')

gin.external_configurable(BatchNormalization, module='tf.keras.layers')
