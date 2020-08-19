import tensorflow as tf

class SEBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, r=16):
        super(SEBlock, self).__init__()
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(units=input_channels // r)
        self.fc2 = tf.keras.layers.Dense(units=input_channels)

    def call(self, inputs, **kwargs):
        branch = self.pool(inputs)
        branch = self.fc1(branch)
        branch = tf.nn.relu(branch)
        branch = self.fc2(branch)
        branch = tf.nn.sigmoid(branch)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = tf.expand_dims(input=branch, axis=1)
        output = tf.keras.layers.multiply(inputs=[inputs, branch])

        return output


class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg= tf.keras.layers.GlobalAveragePooling2D()
        self.max= tf.keras.layers.GlobalMaxPooling2D()
        self.conv1 = tf.keras.layers.Conv2D(filters = in_planes//ratio, 
                                    kernel_size=1,
                                    kernel_initializer='he_normal', 
                                    strides=1,
                                    padding='same')
        
        self.conv2 = tf.keras.layers.Conv2D(filters = in_planes, 
                                            kernel_size=1,
                                            kernel_initializer='he_normal', 
                                            strides=1, 
                                            padding='same')
                                   
    def call(self, inputs):
        avg = self.avg(inputs)
        max = self.max(inputs)
        avg = tf.keras.layers.Reshape((1, 1, avg.shape[1]))(avg)   # shape (None, 1, 1 feature)
        max = tf.keras.layers.Reshape((1, 1, max.shape[1]))(max)   # shape (None, 1, 1 feature)
        avg_out = self.conv2(self.conv1(avg))
        avg_out = tf.nn.relu(avg_out)
        max_out = self.conv2(self.conv1(max))
        max_out = tf.nn.relu(max_out)
        out = avg_out + max_out
        out = tf.nn.sigmoid(out)

        return out

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters = 1,
                                            kernel_size = kernel_size,
                                            kernel_initializer='he_normal',
                                            strides = 1,
                                            padding='same')
    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=3)
        max_out = tf.reduce_max(inputs, axis=3)
        out = tf.stack([avg_out, max_out], axis=3)
        out = self.conv1(out)
        out = tf.nn.relu(out)

        return out

class ConvBlockAttentionModule(tf.keras.layers.Layer):
    def __init__(self, out_channels, ratio = 16, kernel_size = 7):
        super(ConvBlockAttentionModule, self).__init__()
        self.ca = ChannelAttention(in_planes = out_channels,
                                    ratio = ratio)
        self.sa = SpatialAttention(kernel_size= kernel_size)

    def call(self, inputs, **kwargs):
        out = self.ca(inputs) * inputs
        out = self.sa(out) * out

        return out        
