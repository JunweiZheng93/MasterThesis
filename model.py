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

    def call(self, inputs):
        outputs = self.fc(inputs)
        return outputs


class Decomposer(keras.layers.Layer):

    def __init__(self, num_partrs, **kwargs):
        super(Decomposer, self).__init__(**kwargs)
        self.projection_layer_list = list()

        self.binary_shape_encoder = BinaryShapeEncoder()
        for i in range(num_partrs):
            self.projection_layer_list.append(Projection())

    def call(self, inputs, training=False):
        projection_layer_outputs = list()
        x = self.binary_shape_encoder(inputs, training=training)
        for each_layer in self.projection_layer_list:
            projection_layer_outputs.append(each_layer(x))
        # outputs should be a tensor in the shape of (num_parts, batch_size, encoding_dimensions)
        outputs = tf.constant(projection_layer_outputs)
        return outputs


class SharedPartDecoder(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(SharedPartDecoder, self).__init__(**kwargs)

        # inputs should be in the shape of (batch_size, num_dimension)
        self.fc = layers.Dense(2*2*2*256)
        self.reshape = layers.Reshape((2, 2, 2, 256))

        # inputs should be in the shape of (B, H, W, D, C)
        self.deconv1 = layers.Conv3DTranspose(128, 3, (2, 2, 2), padding='same', output_padding=(1, 1, 1))
        self.deconv2 = layers.Conv3DTranspose(64, 3, (2, 2, 2), padding='same', output_padding=(1, 1, 1))
        self.deconv3 = layers.Conv3DTranspose(32, 5, (2, 2, 2), padding='same', output_padding=(1, 1, 1))
        self.deconv4 = layers.Conv3DTranspose(16, 5, (1, 1, 1), padding='same', output_padding=(0, 0, 0))
        self.deconv5 = layers.Conv3DTranspose(1, 5, (2, 2, 2), padding='same', output_padding=(1, 1, 1))

        self.act = layers.ReLU()
        self.act1 = layers.ReLU()
        self.act2 = layers.ReLU()
        self.act3 = layers.ReLU()
        self.act4 = layers.ReLU()
        self.act5 = layers.ReLU()

        self.bn = layers.BatchNormalization()
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.bn4 = layers.BatchNormalization()
        self.bn5 = layers.BatchNormalization()

        self.dropout = layers.Dropout(0.2)
        self.dropout1 = layers.Dropout(0.2)
        self.dropout2 = layers.Dropout(0.2)
        self.dropout3 = layers.Dropout(0.2)
        self.dropout4 = layers.Dropout(0.2)
        self.dropout5 = layers.Dropout(0.2)

    def call(self, inputs, training=False):

        # inputs should be in the shape of (batch_size, num_dimension)
        x = self.fc(inputs)
        x = self.act(x)
        x = self.bn(x, training=training)
        x = self.dropout(x, training=training)

        x = self.reshape(x)

        # inputs should be in the shape of (B, H, W, D, C)
        x = self.deconv1(x)
        x = self.act1(x)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)

        x = self.deconv2(x)
        x = self.act2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)

        x = self.deconv3(x)
        x = self.act3(x)
        x = self.bn3(x, training=training)
        x = self.dropout3(x, training=training)

        x = self.deconv4(x)
        x = self.act4(x)
        x = self.bn4(x, training=training)
        x = self.dropout4(x, training=training)

        x = self.deconv5(x)
        x = self.act5(x)
        x = self.bn5(x, training=training)
        outputs = self.dropout5(x, training=training)

        return outputs


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

    def customized_loss(self):
        pass

    def call(self, inputs, training=False):
        pass

    def train_step(self, data):
        pass

    @property
    def metrics(self):
        pass
