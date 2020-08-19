import tensorflow as tf
import script_model.Attention as at



class MobileNet_classification(tf.keras.layers.Layer):
    def __init__(self, pooling='avg', classes=1):
        super(MobileNet_classification, self).__init__()
        self.classes = classes
        self.pooling = pooling

        self.avg_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.max_pooling = tf.keras.layers.GlobalMaxPooling2D()

        self.sigmoid_fc = tf.keras.layers.Dense(units=classes, activation=tf.keras.activations.sigmoid)
        self.softmax_fc = tf.keras.layers.Dense(units=classes, activation=tf.keras.activations.softmax)

    def call(self, inputs):
        if self.pooling == 'avg':
            x = self.avg_pooling(inputs)
        elif self.pooling == 'max':
            x = self.max_pooling(inputs)

        if self.classes == 1:
            x = self.sigmoid_fc(x)
        else:
            x = self.softmax_fc(x)
        
        return x

class depthwise_separable_convolution(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride, padd="same", use_bias=False, use_se=False, use_cbam=False):
        super(depthwise_separable_convolution, self).__init__()
        #dw
        self.dw_conv_1 = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3),
                                                    strides=stride,
                                                    padding=padd,
                                                    use_bias=use_bias)
        self.bn1 = tf.keras.layers.BatchNormalization()

        #pw
        self.conv_2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            use_bias=use_bias)
        self.bn2 = tf.keras.layers.BatchNormalization()

        #Convolution Attention Block
        self.use_se = use_se
        self.se_block = at.SEBlock(input_channels=filter_num)
        self.use_cbam = use_cbam
        self.cbam_block = at.ConvBlockAttentionModule(out_channels=filter_num)


    def call(self, inputs):
        x = self.dw_conv_1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)

        x = self.conv_2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)

        if self.use_se == True:
            x = self.se_block(x)
        if self.use_cbam == True:
            x = self.cbam_block(x)
        
        return x
    

class MobileNet(tf.keras.Model):
    def __init__(self, classes=1000, use_se=False, use_cbam=False):
        super(MobileNet, self).__init__()

        self.zero_padd_1 = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))
        self.conv_1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="valid",
                                            use_bias=False)
        self.bn_1 = tf.keras.layers.BatchNormalization()


        self.dw_separable_block_1 = depthwise_separable_convolution(filter_num=64,
                                                                    stride=1,
                                                                    padd="same",
                                                                    use_se=use_se,
                                                                    use_cbam=use_cbam)
        
        self.zero_padd_2 = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))

        self.dw_separable_block_2 = depthwise_separable_convolution(filter_num=128,
                                                                    stride=2,
                                                                    padd="valid",
                                                                    use_se=use_se,
                                                                    use_cbam=use_cbam)
        
        self.dw_separable_block_3 = depthwise_separable_convolution(filter_num=128,
                                                                    stride=1,
                                                                    padd="same",
                                                                    use_se=use_se,
                                                                    use_cbam=use_cbam)
        
        self.zero_padd_3 = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))
        
        self.dw_separable_block_4 = depthwise_separable_convolution(filter_num=256,
                                                                    stride=2,
                                                                    padd="valid",
                                                                    use_se=use_se,
                                                                    use_cbam=use_cbam)
        
        self.dw_separable_block_5 = depthwise_separable_convolution(filter_num=256,
                                                                    stride=1,
                                                                    padd="same",
                                                                    use_se=use_se,
                                                                    use_cbam=use_cbam)
        
        self.zero_padd_4 = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))
        
        self.dw_separable_block_6 = depthwise_separable_convolution(filter_num=512,
                                                                    stride=2,
                                                                    padd="valid",
                                                                    use_se=use_se,
                                                                    use_cbam=use_cbam)
        
        self.dw_separable_block_7 = depthwise_separable_convolution(filter_num=512,
                                                                    stride=1,
                                                                    padd="same",
                                                                    use_se=use_se,
                                                                    use_cbam=use_cbam)
        
        self.zero_padd_5 = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))
        
        self.dw_separable_block_8 = depthwise_separable_convolution(filter_num=1204,
                                                                    stride=2,
                                                                    padd="valid",
                                                                    use_se=use_se,
                                                                    use_cbam=use_cbam)
        
        self.dw_separable_block_9 = depthwise_separable_convolution(filter_num=1024,
                                                                    stride=2,
                                                                    padd="same",
                                                                    use_se=use_se,
                                                                    use_cbam=use_cbam)
        self.fc_1 = MobileNet_classification(pooling='avg', 
                                            classes=classes)
    def call(self, inputs, cam = None):
        x_1 = self.zero_padd_1(inputs)
        x_2 = self.conv_1(x_1)
        x_3 = self.bn_1(x_2)
        x_4 = tf.nn.relu(x_3)
        x_5 = self.dw_separable_block_1(x_4)
        x_6 = self.zero_padd_2(x_5)
        x_7 = self.dw_separable_block_2(x_6)
        x_8 = self.dw_separable_block_3(x_7)
        x_9 = self.zero_padd_3(x_8)
        x_10 = self.dw_separable_block_4(x_9)
        x_11 = self.dw_separable_block_5(x_10)
        x_12 = self.zero_padd_4(x_11)
        x_13 = self.dw_separable_block_6(x_12)

        x_14 = self.dw_separable_block_7(x_13)
        x_15 = self.dw_separable_block_7(x_14)
        x_16 = self.dw_separable_block_7(x_15)
        x_17 = self.dw_separable_block_7(x_16)
        x_18 = self.dw_separable_block_7(x_17)

        x_19 = self.zero_padd_5(x_18)
        x_20 = self.dw_separable_block_8(x_19)
        x_21 = self.dw_separable_block_9(x_20)

        x = self.fc_1(x_21)

        if cam == 'grad':
            return x_21, x
        if cam == 'flatten':
            Flatten = tf.keras.layers.Flatten()
            flatten = Flatten(x_21)
            return flatten, x

        return x



def MobileNet_V1(classes = 1000, use_se = False, use_cbam = False):
    return MobileNet(classes=classes, use_se = use_se, use_cbam = use_cbam)
