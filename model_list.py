import tensorflow as tf
from script_model import VggNet, ResNet, ResNext, DenseNet, EfficientNet, MobileNet


def Modellist(model_name, NUM_CLASSES = 1000, use_se = False, use_cbam = False):
    if model_name == 'vgg13':
        return VggNet.vgg_13(NUM_CLASSES, use_se, use_cbam)
    elif model_name == 'vgg16':
        return VggNet.vgg_16(NUM_CLASSES, use_se, use_cbam)
    elif model_name == 'vgg19':
        return VggNet.vgg_19(NUM_CLASSES, use_se, use_cbam)
    elif model_name == 'resnet18':
        return ResNet.ResNet18(NUM_CLASSES, use_se, use_cbam)
    elif model_name == 'resnet34':
        return ResNet.ResNet34(NUM_CLASSES, use_se, use_cbam)
    elif model_name == 'resnet50':
        return ResNet.ResNet50(NUM_CLASSES, use_se, use_cbam)
    elif model_name == 'resnet101':
        return ResNet.ResNet101(NUM_CLASSES, use_se, use_cbam)
    elif model_name == 'resnet152':
        return ResNet.ResNet152(NUM_CLASSES, use_se, use_cbam)
    elif model_name == 'densenet121':
        return DenseNet.densenet_121(NUM_CLASSES, use_se, use_cbam)
    elif model_name == 'densenet169':
        return DenseNet.densenet_169(NUM_CLASSES, use_se, use_cbam)
    elif model_name == 'densenet201':
        return DenseNet.densenet_201(NUM_CLASSES, use_se, use_cbam)
    elif model_name == 'densenet265':
        return DenseNet.densenet_265(NUM_CLASSES, use_se, use_cbam)
    elif model_name == 'efficientnetb0':
        return EfficientNet.efficient_net_b0(NUM_CLASSES, use_se, use_cbam)
    elif model_name == 'efficientnetb1':
        return EfficientNet.efficient_net_b1(NUM_CLASSES, use_se, use_cbam)
    elif model_name == 'efficientnetb2':
        return EfficientNet.efficient_net_b2(NUM_CLASSES, use_se, use_cbam)
    elif model_name == 'efficientnetb3':
        return EfficientNet.efficient_net_b3(NUM_CLASSES, use_se, use_cbam)
    elif model_name == 'efficientnetb4':
        return EfficientNet.efficient_net_b4(NUM_CLASSES, use_se, use_cbam)
    elif model_name == 'efficientnetb5':
        return EfficientNet.efficient_net_b5(NUM_CLASSES, use_se, use_cbam)
    elif model_name == 'efficientnetb6':
        return EfficientNet.efficient_net_b6(NUM_CLASSES, use_se, use_cbam)
    elif model_name == 'efficientnetb7':
        return EfficientNet.efficient_net_b7(NUM_CLASSES, use_se, use_cbam)
    elif model_name == 'mobilenetv1':
        return MobileNet.MobileNet_V1(NUM_CLASSES, use_se, use_cbam)
    elif model_name == 'resnext50':
        return ResNext.ResNext_50(NUM_CLASSES, use_se, use_cbam)
    elif model_name == 'resnext101':
        return ResNext.ResNext_101(NUM_CLASSES, use_se, use_cbam)
    else:
        raise ValueError("The model_name does not exist.")
        



# model = Modellist('vgg13', 2, True, True)

# # #commit
# # model = model(x = 'resnext50', NUM_CLASSES = 500, use_se = True, use_cbam = False)

# model.build(input_shape=(None, 224, 224, 3))

# print(model.summary())
