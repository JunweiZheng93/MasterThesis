import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class BinaryShapeEncoder(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(BinaryShapeEncoder, self).__init__(**kwargs)

        self.conv1 = layers.Conv3D(16, 5, (1, 1, 1), padding='same')
        self.conv2 = layers.Conv3D(32, 5, (2, 2, 2), padding='same')
        self.conv3 = layers.Conv3D(64, 5, (2, 2, 2), padding='same')
        self.conv4 = layers.Conv3D(128, 3, (2, 2, 2), padding='same')
        self.conv5 = layers.Conv3D(256, 3, (2, 2, 2), padding='same')

        self.act1 = layers.ReLU()
        self.act2 = layers.ReLU()
        self.act3 = layers.ReLU()
        self.act4 = layers.ReLU()
        self.act5 = layers.ReLU()

        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.bn4 = layers.BatchNormalization()
        self.bn5 = layers.BatchNormalization()

        self.flat = layers.Flatten()
        self.fc = layers.Dense(100, use_bias=False)

    def call(self, inputs, training=False):

        # inputs should be in the shape of (B, H, W, D, C)
        x = self.conv1(inputs)
        x = self.act1(x)
        x = self.bn1(x, training=training)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x, training=training)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.bn3(x, training=training)

        x = self.conv4(x)
        x = self.act4(x)
        x = self.bn4(x, training=training)

        x = self.conv5(x)
        x = self.act5(x)
        x = self.bn5(x, training=training)

        x = self.flat(x)
        outputs = self.fc(x)

        return outputs


class Projection(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Projection, self).__init__(**kwargs)
        self.fc = layers.Dense(100, use_bias=False)
        self.act = layers.ReLU()

    def call(self, inputs):
        x = self.fc(inputs)
        return x


class Decomposer(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Decomposer, self).__init__(**kwargs)
        pass

    def build(self, input_shape):
        pass

    def call(self, inputs, training=False):
        if training:
            pass
        else:
            pass


class PartDecoder(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(PartDecoder, self).__init__(**kwargs)
        pass

    def build(self, input_shape):
        pass

    def call(self, inputs, training=False):
        if training:
            pass
        else:
            pass


class LocalizationNet(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(LocalizationNet, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, training=False):
        if training:
            pass
        else:
            pass


class STN(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(STN, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, training=False):
        if training:
            pass
        else:
            pass


class Model(keras.Model):

    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.loss_tracker = tf.keras.metrics.Mean()  # 实例化一个metric来记录average training loss

    def customized_loss(self):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, inputs, training=False):  # 一定要call来定义你的计算图
        self.x = self.block1(inputs)  # 扩大中间变量x的scope，有利于将来自定义loss的传参
        y_pred = self.block2(x)  # 如果block2在训练和测试的时候有不同表现，而block1表现相同，只需要在block2传入training=True即可
        return y_pred

    def train_step(self, data):
        x, y_true = data  # 如果fit(x, y, ...)，那么data参数将会是一个元组(x, y)。类推tf.data.Dataset数据类型
        with tf.GradientTape() as tape:
            # 因为重写了train_step，所以调用fit时，不会自动调用call，导致不会自动传入training=True。需要手动调用call以及传入training=True
            y_pred = self.call(x, training=True)
            bce_loss = self.customized_loss()(y_true, y_pred) + tf.math.reduce_sum(self.losses)  # 记得加上之前layer的额外自定义loss
        block1_grads, block2_grads = tape.gradient(bce_loss, [self.block1.trainable_weights, self.block2.trainable_weights])
        self.optimizer.apply_gradients(zip(block1_grads, self.block1.trainable_weights))
        self.optimizer.apply_gradients(zip(block2_grads, self.block2.trainable_weights))
        self.loss_tracker.update_state(bce_loss)  # 计算average training loss，并保存在之前的实例化的loss tracker里面
        return {'BCE_Loss': self.loss_tracker.result()}  # 这个返回值很重要，它为fit的progress bar以及其他的callback提供了数据

    @property
    def metrics(self):
        return [self.loss_tracker]  # 把需要自动reset的tracker放到list里面即可

