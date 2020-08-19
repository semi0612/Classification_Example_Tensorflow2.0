import tensorflow as tf
import script_model.Attention as at
import math



def swish(x):
    return x * tf.nn.sigmoid(x)

def round_filters(filters, multiplier):
    depth_divisor = 8
    min_depth = None
    min_depth = min_depth or depth_divisor
    filters = filters * multiplier
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)

def round_repeats(repeats, multiplier):
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))



class MBConv(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate, use_se=True, use_cbam=False):
        super(MBConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.conv1 = tf.keras.layers.Conv2D(filters=in_channels * expansion_factor,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=(k, k),
                                                        strides=stride,
                                                        padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()

        #Convolution Attention Block
        self.use_se = use_se
        self.se = at.SEBlock(input_channels=in_channels * expansion_factor)
        self.use_cbam = use_cbam
        self.cbam = at.ConvBlockAttentionModule(out_channels=in_channels * expansion_factor)

        self.conv2 = tf.keras.layers.Conv2D(filters=out_channels,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(rate=drop_connect_rate)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = swish(x)
        x = self.dwconv(x)
        x = self.bn2(x, training=training)
        if self.use_se == True:
            x = self.se(x)
        if self.use_cbam == True:
            x = self.cbam(x)

        x = swish(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        if self.stride == 1 and self.in_channels == self.out_channels:
            if self.drop_connect_rate:
                x = self.dropout(x, training=training)
            x = tf.keras.layers.add([x, inputs])
        return x


def build_mbconv_block(in_channels, out_channels, layers, stride, expansion_factor, k, drop_connect_rate, use_se=True, use_cbam=False):
    block = tf.keras.Sequential()
    for i in range(layers):
        if i == 0:
            block.add(MBConv(in_channels=in_channels,
                                out_channels=out_channels,
                                expansion_factor=expansion_factor,
                                stride=stride,
                                k=k,
                                drop_connect_rate=drop_connect_rate,
                                use_se=use_se,
                                use_cbam=use_cbam))
        else:
            block.add(MBConv(in_channels=out_channels,
                                out_channels=out_channels,
                                expansion_factor=expansion_factor,
                                stride=1,
                                k=k,
                                drop_connect_rate=drop_connect_rate,
                                use_se=use_se,
                                use_cbam=use_cbam))
    return block


class EfficientNet(tf.keras.Model):
    def __init__(self, width_coefficient, depth_coefficient, dropout_rate, drop_connect_rate=0.2, use_se=True, use_cbam=False, classes=1):
        super(EfficientNet, self).__init__()
        self.classes = classes

        self.conv1 = tf.keras.layers.Conv2D(filters=round_filters(32, width_coefficient),
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.block1 = build_mbconv_block(in_channels=round_filters(32, width_coefficient),
                                            out_channels=round_filters(16, width_coefficient),
                                            layers=round_repeats(1, depth_coefficient),
                                            stride=1,
                                            expansion_factor=1, k=3, drop_connect_rate=drop_connect_rate,
                                            use_se=use_se, use_cbam=use_cbam)
        self.block2 = build_mbconv_block(in_channels=round_filters(16, width_coefficient),
                                            out_channels=round_filters(24, width_coefficient),
                                            layers=round_repeats(2, depth_coefficient),
                                            stride=2,
                                            expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate,
                                            use_se=use_se, use_cbam=use_cbam)
        self.block3 = build_mbconv_block(in_channels=round_filters(24, width_coefficient),
                                            out_channels=round_filters(40, width_coefficient),
                                            layers=round_repeats(2, depth_coefficient),
                                            stride=2,
                                            expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate,
                                            use_se=use_se, use_cbam=use_cbam)
        self.block4 = build_mbconv_block(in_channels=round_filters(40, width_coefficient),
                                            out_channels=round_filters(80, width_coefficient),
                                            layers=round_repeats(3, depth_coefficient),
                                            stride=2,
                                            expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate,
                                            use_se=use_se, use_cbam=use_cbam)
        self.block5 = build_mbconv_block(in_channels=round_filters(80, width_coefficient),
                                            out_channels=round_filters(112, width_coefficient),
                                            layers=round_repeats(3, depth_coefficient),
                                            stride=1,
                                            expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate,
                                            use_se=use_se, use_cbam=use_cbam)
        self.block6 = build_mbconv_block(in_channels=round_filters(112, width_coefficient),
                                            out_channels=round_filters(192, width_coefficient),
                                            layers=round_repeats(4, depth_coefficient),
                                            stride=2,
                                            expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate,
                                            use_se=use_se, use_cbam=use_cbam)
        self.block7 = build_mbconv_block(in_channels=round_filters(192, width_coefficient),
                                            out_channels=round_filters(320, width_coefficient),
                                            layers=round_repeats(1, depth_coefficient),
                                            stride=1,
                                            expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate,
                                            use_se=use_se, use_cbam=use_cbam)

        self.conv2 = tf.keras.layers.Conv2D(filters=round_filters(1280, width_coefficient),
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()



        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        if self.classes > 2:
            self.fc1 = tf.keras.layers.Dense(units=classes,
                                        activation=tf.keras.activations.softmax)
        else:
            self.fc2 = tf.keras.layers.Dense(units=classes,
                                        activation=tf.keras.activations.sigmoid)

    def call(self, inputs, training=None, cam=None):
        conv_1 = self.conv1(inputs)
        bn_1 = self.bn1(conv_1, training=training)
        swish_1 = swish(bn_1)

        block_1 = self.block1(swish_1)
        block_2 = self.block2(block_1)
        block_3 = self.block3(block_2)
        block_4 = self.block4(block_3)
        block_5 = self.block5(block_4)
        block_6 = self.block6(block_5)
        block_7 = self.block7(block_6)

        conv_2 = self.conv2(block_7)

        bn_2 = self.bn2(conv_2, training=training)
        swish_2 = swish(bn_2)
        global_avgpool = self.pool(swish_2)
        dropout = self.dropout(global_avgpool, training=training)

        if self.classes > 2: 
            x = self.fc1(dropout)
        else:
            x = self.fc2(dropout)

        if cam == 'grad':
            return conv_2, x

        if cam == 'flatten':
            Flatten = tf.keras.layers.Flatten()
            flatten = Flatten(swish_2)
            return flatten, x

        return x
        


def get_efficient_net(width_coefficient=1.0, depth_coefficient=1.0, resolution=224, dropout_rate=0.2, use_se=False, use_cbam=False, classes=1000):
    net = EfficientNet(width_coefficient=width_coefficient,
                        depth_coefficient=depth_coefficient,
                        dropout_rate=dropout_rate, 
                        drop_connect_rate=dropout_rate, 
                        use_se=use_se, 
                        use_cbam=use_cbam, 
                        classes=classes)

    return net



def efficient_net_b0(classes = 1000, use_se=False, use_cbam=False):
    return get_efficient_net(1.0, 1.0, 224, 0.2, use_se=use_se, use_cbam=use_cbam, classes=classes)

def efficient_net_b1(classes = 1000, use_se=False, use_cbam=False):
    return get_efficient_net(1.0, 1.1, 240, 0.2, use_se=use_se, use_cbam=use_cbam, classes=classes)

def efficient_net_b2(classes = 1000, use_se=False, use_cbam=False):
    return get_efficient_net(1.1, 1.2, 260, 0.3, use_se=use_se, use_cbam=use_cbam, classes=classes)

def efficient_net_b3(classes = 1000, use_se=False, use_cbam=False):
    return get_efficient_net(1.2, 1.4, 300, 0.3, use_se=use_se, use_cbam=use_cbam, classes=classes)

def efficient_net_b4(classes = 1000, use_se=False, use_cbam=False):
    return get_efficient_net(1.4, 1.8, 380, 0.4, use_se=use_se, use_cbam=use_cbam, classes=classes)

def efficient_net_b5(classes = 1000, use_se=False, use_cbam=False):
    return get_efficient_net(1.6, 2.2, 456, 0.4, use_se=use_se, use_cbam=use_cbam, classes=classes)

def efficient_net_b6(classes = 1000, use_se=False, use_cbam=False):
    return get_efficient_net(1.8, 2.6, 528, 0.5, use_se=use_se, use_cbam=use_cbam, classes=classes)

def efficient_net_b7(classes = 1000, use_se=False, use_cbam=False):
    return get_efficient_net(2.0, 3.1, 600, 0.5, use_se=use_se, use_cbam=use_cbam, classes=classes)
