import tensorflow as tf
import script_model.Attention as at

#class BasicBlock(tf.keras.Model):
class BasicBlock(tf.keras.layers.Layer):
    expansion = 1

    def __init__(self, in_channels, out_channels, strides=1, use_se=False, use_cbam = False):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=strides,
                                            padding="same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=1,
                                            padding="same", use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

        # Add SEblock and CBAM
        self.use_se = use_se
        if self.use_se == True:
            self.se_block = at.SEBlock(input_channels=out_channels * 4)

        self.use_cbam = use_cbam
        if self.use_cbam == True:
            self.cbam_block = at.ConvBlockAttentionModule(
                out_channels=out_channels * 4)

        """
        Adds a shortcut between input and residual block and merges them with "sum"
        """
        if strides != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = tf.keras.Sequential([
                tf.keras.layers.Conv2D(self.expansion*out_channels, kernel_size=1,
                                       strides=strides, use_bias=False),
                tf.keras.layers.BatchNormalization()]
            )
        else:
            self.shortcut = lambda x, _: x

    def call(self, x, training=False):
        # if training: print("=> training network ... ")
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        if self.use_se == True:
            out = self.se_block(out)
        if self.use_cbam == True:
            out = self.cbam_block(out)
        out = tf.keras.layers.add([self.shortcut(x, training), out])
        return tf.nn.relu(out)


#class Bottleneck(tf.keras.Model):
class Bottleneck(tf.keras.layers.Layer):
    expansion = 4

    def __init__(self, in_channels, out_channels, strides=1, use_se=False, use_cbam=False):
        super(Bottleneck, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(out_channels, 1, 1, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            out_channels, 3, strides, padding="same", use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(
            out_channels*self.expansion, 1, 1, use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()

        # Add SEblock and CBAM
        self.use_se = use_se
        if self.use_se == True:
            self.se_block = at.SEBlock(input_channels=out_channels*self.expansion)

        self.use_cbam = use_cbam
        if self.use_cbam == True:
            self.cbam_block = at.ConvBlockAttentionModule(
                out_channels=out_channels*self.expansion)

        if strides != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = tf.keras.Sequential([
                tf.keras.layers.Conv2D(self.expansion*out_channels, kernel_size=1,
                                       strides=strides, use_bias=False),
                tf.keras.layers.BatchNormalization()]
            )
        else:
            self.shortcut = lambda x, _: x

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training))
        out = tf.nn.relu(self.bn2(self.conv2(out), training))
        out = self.bn3(self.conv3(out), training)
        if self.use_se == True:
            out = self.se_block(out)
        if self.use_cbam == True:
            out = self.cbam_block(out)
        out = tf.keras.layers.add([self.shortcut(x, training), out])
        return tf.nn.relu(out)


class ResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes, use_se=False, use_cbam=False):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = tf.keras.layers.Conv2D(
            64, 3, 1, padding="same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.layer1 = self._make_layer(
            block,  64, num_blocks[0], stride=1, use_se=use_se, use_cbam=use_cbam)
        self.layer2 = self._make_layer(
            block, 128, num_blocks[1], stride=2, use_se=use_se, use_cbam=use_cbam)
        self.layer3 = self._make_layer(
            block, 256, num_blocks[2], stride=2, use_se=use_se, use_cbam=use_cbam)
        self.layer4 = self._make_layer(
            block, 512, num_blocks[3], stride=2, use_se=use_se, use_cbam=use_cbam)

        # for pca and t-sne
        self.flatten = tf.keras.layers.Flatten()

        self.avg_pool2d = tf.keras.layers.AveragePooling2D(4)
        self.linear = tf.keras.layers.Dense(
            units=num_classes, activation="softmax")

    def _make_layer(self, block, out_channels, num_blocks, stride, use_se=False, use_cbam=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels,
                                stride, use_se=use_se, use_cbam=use_cbam))
            self.in_channels = out_channels * block.expansion
        return tf.keras.Sequential(layers)

    def call(self, x, training=False, mode = None):
        seq_1 = tf.nn.relu(self.bn1(self.conv1(x), training))
        seq_2 = self.layer1(seq_1, training=training)
        seq_3 = self.layer2(seq_2, training=training)
        seq_4 = self.layer3(seq_3, training=training)
        seq_5 = self.layer4(seq_4, training=training)

        # For classification
        out = self.avg_pool2d(seq_5)
        flat_out = self.flatten(out)
        #out = tf.reshape(out, (out.shape[0], -1))
        out = self.linear(flat_out)
        if mode == 'grad':
            return [seq_1, seq_2, seq_3, seq_4, seq_5], out
        elif mode == 'flatten':
            return flat_out
        return out


def ResNet18(classes=10, use_se=False, use_cbam=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes = classes, use_se=use_se, use_cbam=use_cbam)


def ResNet34(classes=10, use_se=False, use_cbam=False):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes = classes, use_se=use_se, use_cbam=use_cbam)


def ResNet50(classes=10, use_se=False, use_cbam=False):
    return ResNet(Bottleneck, [3, 4, 14, 3], num_classes = classes, use_se=use_se, use_cbam=use_cbam)


def ResNet101(classes=10, use_se=False, use_cbam=False):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes = classes, use_se=use_se, use_cbam=use_cbam)


def ResNet152(classes=10, use_se=False, use_cbam=False):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes = classes, use_se=use_se, use_cbam=use_cbam)
