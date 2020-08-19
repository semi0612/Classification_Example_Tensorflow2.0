import tensorflow as tf
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import activations
import script_model.Attention as at



class GroupConv2D(tf.keras.layers.Layer):
    def __init__(self,
                input_channels,
                output_channels,
                kernel_size,
                strides=(1, 1),
                padding='valid',
                data_format=None,
                dilation_rate=(1, 1),
                activation=None,
                groups=1,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                **kwargs):
        super(GroupConv2D, self).__init__()

        if not input_channels % groups == 0:
            raise ValueError("The value of input_channels must be divisible by the value of groups.")
        if not output_channels % groups == 0:
            raise ValueError("The value of output_channels must be divisible by the value of groups.")

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.groups = groups
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        self.group_in_num = input_channels // groups
        self.group_out_num = output_channels // groups
        self.conv_list = []
        for i in range(self.groups):
            self.conv_list.append(tf.keras.layers.Conv2D(filters=self.group_out_num,
                                                        kernel_size=kernel_size,
                                                        strides=strides,
                                                        padding=padding,
                                                        data_format=data_format,
                                                        dilation_rate=dilation_rate,
                                                        activation=activations.get(activation),
                                                        use_bias=use_bias,
                                                        kernel_initializer=initializers.get(kernel_initializer),
                                                        bias_initializer=initializers.get(bias_initializer),
                                                        kernel_regularizer=regularizers.get(kernel_regularizer),
                                                        bias_regularizer=regularizers.get(bias_regularizer),
                                                        activity_regularizer=regularizers.get(activity_regularizer),
                                                        kernel_constraint=constraints.get(kernel_constraint),
                                                        bias_constraint=constraints.get(bias_constraint),
                                                        **kwargs))

    def call(self, inputs, **kwargs):
        feature_map_list = []
        for i in range(self.groups):
            x_i = self.conv_list[i](inputs[:, :, :, i*self.group_in_num: (i + 1) * self.group_in_num])
            feature_map_list.append(x_i)
        out = tf.concat(feature_map_list, axis=-1)
        return out

    def get_config(self):
        config = {
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "activation": activations.serialize(self.activation),
            "groups": self.groups,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint)
        }
        base_config = super(GroupConv2D, self).get_config()
        return {**base_config, **config}



class ResNext_BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filters, strides, groups, use_se=False, use_cbam=False):
        super(ResNext_BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.group_conv = GroupConv2D(input_channels=filters,
                                        output_channels=filters,
                                        kernel_size=(3, 3),
                                        strides=strides,
                                        padding="same",
                                        groups=groups)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=2 * filters,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()

        #attention module
        self.use_se = use_se
        self.se_block = at.SEBlock(input_channels=2 * filters)
        self.use_cbam = use_cbam
        self.cbam_block = at.ConvBlockAttentionModule(out_channels=2 * filters)

        self.shortcut_conv = tf.keras.layers.Conv2D(filters=2 * filters,
                                                    kernel_size=(1, 1),
                                                    strides=strides,
                                                    padding="same")
        self.shortcut_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.group_conv(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        if self.use_se == True:
            x = self.se_block(x)
        if self.use_cbam == True:
            x = self.cbam_block(x)


        shortcut = self.shortcut_conv(inputs)
        shortcut = self.shortcut_bn(shortcut, training=training)

        output = tf.nn.relu(tf.keras.layers.add([x, shortcut]))
        return output


def resnext_block(filters, strides, groups, repeat_num, use_se=False, use_cbam=False):
    block = tf.keras.Sequential()
    block.add(ResNext_BottleNeck(filters=filters,
                                    strides=strides,
                                    groups=groups,
                                    use_se=use_se,
                                    use_cbam=use_cbam))
    for _ in range(1, repeat_num):
        block.add(ResNext_BottleNeck(filters=filters,
                                        strides=1,
                                        groups=groups,
                                        use_se=use_se,
                                        use_cbam=use_cbam))

    return block

    

class ResNext(tf.keras.Model):
    def __init__(self, repeat_num_list, cardinality, classes=1000, use_se=False, use_cbam=False):
        if len(repeat_num_list) != 4:
            raise ValueError("Thre length of repeat_num_list must be four.")
        super(ResNext, self).__init__()
        self.classes = classes

        self.conv_1 = tf.keras.layers.Conv2D(filters=64, 
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.pool_1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                strides=2,
                                                padding="same")

        self.block_1 = resnext_block(filters=128,
                                    strides=1,
                                    groups=cardinality,
                                    repeat_num=repeat_num_list[0],
                                    use_se=use_se,
                                    use_cbam=use_cbam)
        self.block_2 = resnext_block(filters=256,
                                    strides=2,
                                    groups=cardinality,
                                    repeat_num=repeat_num_list[1],
                                    use_se=use_se,
                                    use_cbam=use_cbam)
        self.block_3 = resnext_block(filters=512,
                                    strides=2,
                                    groups=cardinality,
                                    repeat_num=repeat_num_list[2],
                                    use_se=use_se,
                                    use_cbam=use_cbam)
        self.block_4 = resnext_block(filters=1024,
                                    strides=2,
                                    groups=cardinality,
                                    repeat_num=repeat_num_list[3],
                                    use_se=use_se,
                                    use_cbam=use_cbam)

        self.pool_2 = tf.keras.layers.GlobalAveragePooling2D()

        if self.classes >= 3:
            self.fc_1 = tf.keras.layers.Dense(units=classes,
                                        activation=tf.keras.activations.softmax)
        else:
            self.fc_2 = tf.keras.layers.Dense(units=classes,
                                        activation=tf.keras.activations.sigmoid)

    def call(self, inputs, training=None, cam=None):
        conv_1 = self.conv_1(inputs)
        bn_1 = self.bn_1(conv_1, training=training)
        relu_1 = tf.nn.relu(bn_1)
        pool_1 = self.pool_1(relu_1)

        block_1 = self.block_1(pool_1, training=training)
        block_2 = self.block_2(block_1, training=training)
        block_3 = self.block_3(block_2, training=training)
        block_4 = self.block_4(block_3, training=training)

        pool_2 = self.pool_2(block_4)
        
        if self.classes >= 3: 
            x = self.fc_1(pool_2)
        else:
            x = self.fc_2(pool_2)

        if cam == 'grad':
            return block_4, x

        if cam == 'flatten':
            Flatten = tf.keras.layers.Flatten()
            flatten = Flatten(block_4)
            return flatten, x
    

        return x



def ResNext_50(NUM_CLASSES=1000, use_se=False, use_cbam=False):
    return ResNext(repeat_num_list=[3, 4, 6, 3],
                    cardinality=32,
                    classes=NUM_CLASSES,
                    use_se=use_se,
                    use_cbam=use_cbam)


def ResNext_101(NUM_CLASSES=1000, use_se=False, use_cbam=False):
    return ResNext(repeat_num_list=[3, 4, 23, 3],
                    cardinality=32,
                    classes=NUM_CLASSES,
                    use_se=use_se,
                    use_cbam=use_cbam)
