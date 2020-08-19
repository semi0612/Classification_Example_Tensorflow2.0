import tensorflow as tf
import script_model.Attention as at



layer_in_block = {'vgg11': [1, 1, 2, 2, 2],
                    'vgg13': [2, 2, 2, 2, 2],
                    'vgg16': [2, 2, 3, 3, 3],
                    'vgg19': [2, 2, 4, 4, 4]}



class VggConv(tf.keras.layers.Layer):
    def __init__(self, filter_num=None, kernel_size=(3, 3), \
                activation='relu', padding='same', kernel_initializer='he_normal'):
        super(VggConv, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=kernel_size,
                                            activation=activation,
                                            padding=padding,
                                            kernel_initializer=kernel_initializer)   
        
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        return x

class VggConvs(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2)):
        super(VggConvs, self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=pool_size)
    
    def call(self, inputs, training=None):
        x = self.bn1(inputs, training=training)
        x = self.pool1(x)
        return x

class VggDense(tf.keras.layers.Layer):
    def __init__(self, filter_num=None, classes=1):
        super(VggDense, self).__init__()
        self.Dense1 = tf.keras.layers.Dense(filter_num, activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        if classes == 1:
            self.Dense2 = tf.keras.layers.Dense(classes, activation=tf.keras.activations.sigmoid)
        else:
            self.Dense2 = tf.keras.layers.Dense(classes, activation=tf.keras.activations.softmax)

    def call(self, inputs):
        x = self.Dense1(inputs)
        x = self.bn1(x)
        x = self.Dense2(x)

        return x


def vgg_convs_layer(filter_num=None, blocks=None,  kernel_size=(3, 3), \
                    activation='relu', padding='same', kernel_initializer='he_normal', \
                    pool_size=(2, 2), use_se = False, use_cbam = False):
    vgg_block = tf.keras.Sequential()
    for i in range(blocks):
        vgg_block.add(VggConv(filter_num=filter_num,  kernel_size=kernel_size, activation=activation, padding=padding, kernel_initializer=kernel_initializer))
    vgg_block.add(VggConvs(pool_size=pool_size))
    if use_se == True:
        vgg_block.add(at.SEBlock(input_channels=filter_num))
    if use_cbam == True:
        vgg_block.add(at.ConvBlockAttentionModule(out_channels = filter_num))
    return vgg_block



class VggNet(tf.keras.Model):
    def __init__(self, layer='vgg16', use_se = False, use_cbam = False, classes=1):
        super(VggNet, self).__init__()
    
        self.conv1 = vgg_convs_layer(filter_num = 64, blocks = layer_in_block[layer][0], use_se = use_se, use_cbam=use_cbam)
        self.conv2 = vgg_convs_layer(filter_num = 128, blocks =  layer_in_block[layer][1], use_se = use_se, use_cbam=use_cbam)
        self.conv3 = vgg_convs_layer(filter_num = 256, blocks = layer_in_block[layer][2], use_se = use_se, use_cbam=use_cbam)
        self.conv4 = vgg_convs_layer(filter_num = 512, blocks = layer_in_block[layer][3], use_se = use_se, use_cbam=use_cbam)
        self.conv5 = vgg_convs_layer(filter_num = 512, blocks = layer_in_block[layer][4], use_se = use_se, use_cbam=use_cbam)
        self.Flatten = tf.keras.layers.Flatten()
        self.dense = VggDense(filter_num = 256, classes = classes)

    def call(self, inputs, cam=None):
        seq_1 = self.conv1(inputs)
        seq_2 = self.conv2(seq_1)
        seq_3 = self.conv3(seq_2)
        seq_4 = self.conv4(seq_3)
        seq_5 = self.conv5(seq_4)
        flatten = self.Flatten(seq_5)
        x = self.dense(flatten)

        if cam == 'grad':
            return [seq_1, seq_2, seq_3, seq_4, seq_5], x
            # return seq_1, x

        if cam == 'flatten':
            return flatten, x
        
        return x



def vgg_11(classes = 1000, use_se = False, use_cbam = False):
    return VggNet(layer='vgg11', use_se = use_se,  use_cbam = use_cbam, classes=classes)

def vgg_13(classes = 1000, use_se = False, use_cbam = False):
    return VggNet(layer='vgg13', use_se = use_se,  use_cbam = use_cbam, classes=classes)

def vgg_16(classes = 1000, use_se = False, use_cbam = False):
    return VggNet(layer='vgg16', use_se = use_se,  use_cbam = use_cbam, classes=classes)

def vgg_19(classes = 1000, use_se = False, use_cbam = False):
    return VggNet(layer='vgg19', use_se = use_se,  use_cbam = use_cbam, classes=classes)
