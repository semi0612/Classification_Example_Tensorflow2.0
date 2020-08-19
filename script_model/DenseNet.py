import tensorflow as tf
import script_model.Attention as at



layers_in_block = {'DenseNet-121': [6, 12, 24, 16],
                    'DenseNet-169': [6, 12, 32, 32],
                    'DenseNet-201': [6, 12, 48, 32],
                    'DenseNet-265': [6, 12, 64, 48]}



class DenseNet_classification(tf.keras.layers.Layer):
    def __init__(self, pooling='avg', classes=1):
        super(DenseNet_classification, self).__init__()
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

def make_dense_layer(pooling='avg', classes=1):
    res_dense = tf.keras.Sequential()
    res_dense.add(DenseNet_classification(pooling=pooling, classes=classes))
    
    return res_dense

class bottle_building_block(tf.keras.layers.Layer):
    def __init__(self, growth_rate=32, use_se=False, use_cbam=False, dropout_rate=0.0):
        super(bottle_building_block, self).__init__()
        self.bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
        self.use_dropout = dropout_rate
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        self.conv1 = tf.keras.layers.Conv2D(filters=growth_rate*4,
                                            kernel_size=(1, 1),
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=self.bn_axis, epsilon=1.001e-5)
        
        self.conv2 = tf.keras.layers.Conv2D(filters=growth_rate,
                                            kernel_size=(3, 3),
                                            padding='same',
                                            use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization(axis=self.bn_axis, epsilon=1.001e-5)

        self.use_se = use_se
        self.se_block = at.SEBlock(input_channels=growth_rate)

        self.use_cbam = use_cbam
        self.cbam_block = at.ConvBlockAttentionModule(out_channels=growth_rate)

    def call(self, inputs, training=None):
        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        if self.use_dropout > 0:
            x = self.dropout(x)

        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        if self.use_dropout > 0:
            x = self.dropout(x)

        if self.use_se:
            x = self.se_block(x)
        if self.use_cbam:
            x = self.cbam_block(x)    

        x = tf.concat([inputs, x], axis=self.bn_axis)
            
        return x


class transition_block(tf.keras.layers.Layer):
    def __init__(self, output_channels, dropout_rate=0.0):
        super(transition_block, self).__init__()
        self.bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
        self.output_channels = output_channels
        self.use_dropout = dropout_rate
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        self.conv1 = tf.keras.layers.Conv2D(filters=output_channels,
                                            kernel_size=(1, 1),
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=self.bn_axis, epsilon=1.001e-5)
        self.avg_pooling = tf.keras.layers.AveragePooling2D(pool_size=(2,2),
                                                            strides=2,
                                                            padding="same")
        
    def call(self, inputs, training=None):
        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.avg_pooling(x)
        if self.use_dropout > 0:
            x = self.dropout(x)

        return x


def dense_block(growth_rate=32, layer_block=None, use_se=False, use_cbam=False, dropout_rate=0.0):
    for i in range(layer_block):
        x = bottle_building_block(growth_rate=growth_rate, use_se=use_se, use_cbam=use_cbam, dropout_rate=dropout_rate)
    return x 

class DenseNet_model(tf.keras.Model):
    def __init__(self, num_init_features=64, layer_block='DenseNet-121', 
                growth_rate=32, reduction=0.5, classes=1, pooling='avg',  
                use_se=False, use_cbam = False, dropout_rate=0.0):
        super(DenseNet_model, self).__init__()
        
        self.bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
        
        self.zero_padd_1 = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)))
        self.conv_1 = tf.keras.layers.Conv2D(filters=num_init_features, 
                            kernel_size=(7, 7), 
                            strides=2, 
                            use_bias=False)
        self.bn_1 = tf.keras.layers.BatchNormalization(axis=self.bn_axis, epsilon=1.001e-5)
        self.zero_padd_2 = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))
        self.max_pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), 
                                                    strides=2)
        
        self.num_channels = num_init_features
        self.dense_block_1 = dense_block(growth_rate=growth_rate, layer_block=layers_in_block[layer_block][0], use_se=use_se, use_cbam=use_cbam, dropout_rate=dropout_rate)
        self.num_channels += growth_rate * layers_in_block[layer_block][0]
        self.num_channels = reduction * self.num_channels
        self.transition_block_1 = transition_block(output_channels=int(self.num_channels), dropout_rate=dropout_rate)
        self.dense_block_2 = dense_block(growth_rate=growth_rate, layer_block=layers_in_block[layer_block][1], use_se=use_se, use_cbam=use_cbam, dropout_rate=dropout_rate)
        self.num_channels += growth_rate * layers_in_block[layer_block][1]
        self.num_channels = reduction * self.num_channels
        self.transition_block_2 = transition_block(output_channels=int(self.num_channels), dropout_rate=dropout_rate)
        self.dense_block_3 = dense_block(growth_rate=growth_rate, layer_block=layers_in_block[layer_block][2],use_se=use_se, use_cbam=use_cbam, dropout_rate=dropout_rate)
        self.num_channels += growth_rate * layers_in_block[layer_block][2]
        self.num_channels = reduction * self.num_channels
        self.transition_block_3 = transition_block(output_channels=int(self.num_channels), dropout_rate=dropout_rate)
        self.dense_block_4 = dense_block(growth_rate=growth_rate, layer_block=layers_in_block[layer_block][3], use_se=use_se, use_cbam=use_cbam, dropout_rate=dropout_rate)

        self.bn_2 = tf.keras.layers.BatchNormalization(axis=self.bn_axis, epsilon=1.001e-5)
        
        self.fc = make_dense_layer(pooling, classes)

    def call(self, inputs, training=None, cam=None):
        x_1 = self.zero_padd_1(inputs)
        x_2 = self.conv_1(x_1)
        x_3 = self.bn_1(x_2, training=training)
        x_4 = tf.nn.relu(x_3)
        x_5 = self.zero_padd_2(x_4)
        x_6 = self.max_pool_1(x_5)

        x_7 = self.dense_block_1(x_6)
        x_8 = self.transition_block_1(x_7)
        x_9 = self.dense_block_2(x_8)
        x_10 = self.transition_block_2(x_9)
        x_11 = self.dense_block_3(x_10)
        x_12 = self.transition_block_3(x_11)
        x_13 = self.dense_block_4(x_12)

        x_14 = self.bn_2(x_13)
        x_15 = tf.nn.relu(x_14)

        x = self.fc(x_15)

        if cam == 'grad':
            return x_13, x
        
        if cam == 'flatten':
            Flatten = tf.keras.layers.Flatten()
            flatten = Flatten(x_15)
            return flatten, x

        return x



def densenet_121(NUM_CLASSES = 1000, use_se = False, use_cbam = False):
    return DenseNet_model(num_init_features=64, layer_block='DenseNet-121', growth_rate=32, reduction=0.5, classes=NUM_CLASSES, pooling='avg', dropout_rate=0.5, use_se = use_se, use_cbam = use_cbam)

def densenet_169(NUM_CLASSES = 1000, use_se = False, use_cbam = False):
    return DenseNet_model(num_init_features=64, layer_block='DenseNet-169', growth_rate=32, reduction=0.5, classes=NUM_CLASSES, pooling='avg', dropout_rate=0.5, use_se = use_se, use_cbam = use_cbam)

def densenet_201(NUM_CLASSES = 1000, use_se = False, use_cbam = False):
    return DenseNet_model(num_init_features=64, layer_block='DenseNet-201', growth_rate=32, reduction=0.5,classes=NUM_CLASSES, pooling='avg', dropout_rate=0.5, use_se = use_se, use_cbam = use_cbam)

def densenet_265(NUM_CLASSES = 1000, use_se = False, use_cbam = False):
    return DenseNet_model(num_init_features=64, layer_block='DenseNet-265', growth_rate=32, reduction=0.5,classes=NUM_CLASSES, pooling='avg', dropout_rate=0.5, use_se = use_se, use_cbam = use_cbam)
