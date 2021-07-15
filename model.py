import tensorflow as tf
import numpy as np
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

    def __init__(self, num_parts, **kwargs):
        super(Decomposer, self).__init__(**kwargs)
        self.projection_layer_list = list()
        self.binary_shape_encoder = BinaryShapeEncoder()
        for i in range(num_parts):
            self.projection_layer_list.append(Projection())

    def call(self, inputs, training=False):
        projection_layer_outputs = list()
        x = self.binary_shape_encoder(inputs, training=training)
        for each_layer in self.projection_layer_list:
            projection_layer_outputs.append(each_layer(x))
        # outputs should be in the shape of (num_parts, B, encoding_dimensions)
        outputs = tf.convert_to_tensor(projection_layer_outputs)
        return outputs


class SharedPartDecoder(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(SharedPartDecoder, self).__init__(**kwargs)

        # inputs should be in the shape of (batch_size, num_dimension)
        self.fc = layers.Dense(2*2*2*256)
        self.reshape = layers.Reshape((2, 2, 2, 256))

        # inputs should be in the shape of (B, D, H, W, C)
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

        # inputs should be in the shape of (B, D, H, W, C)
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
        x = self.dropout5(x, training=training)
        outputs = tf.transpose(x, [0, 2, 3, 1, 4])

        return outputs


class LocalizationNet(keras.layers.Layer):

    def __init__(self, num_parts, **kwargs):
        super(LocalizationNet, self).__init__(**kwargs)

        # the shape of the stacked input should be (B, num_parts, H, W, D, C)
        # the shape of the summed input should be (B, encoding_dimensions)
        self.stacked_flat = layers.Flatten()

        self.stacked_fc1 = layers.Dense(256)
        self.stacked_fc2 = layers.Dense(256)
        self.summed_fc1 = layers.Dense(128)
        self.final_fc1 = layers.Dense(128)
        self.final_fc2 = layers.Dense(num_parts*12)

        self.stacked_act1 = layers.ReLU()
        self.stacked_act2 = layers.ReLU()
        self.summed_act1 = layers.ReLU()
        self.final_act1 = layers.ReLU()

        self.stacked_dropout1 = layers.Dropout(0.3)
        self.stacked_dropout2 = layers.Dropout(0.3)
        self.summed_dropout1 = layers.Dropout(0.3)
        self.final_dropout1 = layers.Dropout(0.3)

    def call(self, inputs, training=False):

        stacked_inputs = inputs[0]
        summed_inputs = inputs[1]

        # processing stacked inputs
        stacked_x = self.stacked_flat(stacked_inputs)
        stacked_x = self.stacked_fc1(stacked_x)
        stacked_x = self.stacked_act1(stacked_x)
        stacked_x = self.stacked_dropout1(stacked_x, training=training)
        stacked_x = self.stacked_fc2(stacked_x)
        stacked_x = self.stacked_act2(stacked_x)
        stacked_x = self.stacked_dropout2(stacked_x, training=training)

        # processing summed inputs
        summed_x = self.summed_fc1(summed_inputs)
        summed_x = self.summed_act1(summed_x)
        summed_x = self.summed_dropout1(summed_x, training=training)

        # concatenate stacked inputs and summed inputs into final inputs
        final_x = tf.concat([stacked_x, summed_x], axis=1)
        final_x = self.final_fc1(final_x)
        final_x = self.final_act1(final_x)
        final_x = self.final_dropout1(final_x, training=training)
        final_x = self.final_fc2(final_x)

        # the shape of final_x should be (B, num_parts*12)
        return final_x


class Resampling(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Resampling, self).__init__(**kwargs)

    def call(self, inputs):

        # input_fmap has shape (B, num_parts, H, W, D, C)
        # theta has shape (B, num_parts*12)
        input_fmap = inputs[0]
        theta = inputs[1]

        # batch_grids has shape (B, num_parts, 3, H, W, D)
        batch_grids = self._affine_grid_generator(input_fmap, theta)

        # x_s, y_s and z_s have shape (B, num_parts, H, W, D)
        x_s = batch_grids[:, :, 0, :, :, :]
        y_s = batch_grids[:, :, 1, :, :, :]
        z_s = batch_grids[:, :, 2, :, :, :]

        # output_fmap has shape (B, num_parts, H, W, D, C)
        output_fmap = self._trilinear_sampler(input_fmap, x_s, y_s, z_s)

        return output_fmap

    @staticmethod
    def _affine_grid_generator(input_fmap, theta):
        """
        :param input_fmap: the stacked decoded parts in the shape of (B, num_parts, H, W, D, C)
        :param theta: the output of LocalizationNet. has shape (B, num_parts*12)
        :return: affine grid for the input feature map. affine grid has shape (B, num_parts, 3, H, W, D)
        """

        # get B, num_parts, H, W, D
        B = tf.shape(input_fmap)[0]
        num_parts = tf.shape(input_fmap)[1]
        H = tf.shape(input_fmap)[2]
        W = tf.shape(input_fmap)[3]
        D = tf.shape(input_fmap)[4]

        # reshape theta to (B, num_parts, 3, 4)
        theta = tf.reshape(theta, [B, num_parts, 3, 4])

        # create regular 3D grid, which are the x, y, z coordinates of the output feature map
        x = tf.linspace(-1.0, 1.0, W)
        y = tf.linspace(-1.0, 1.0, H)
        z = tf.linspace(-1.0, 1.0, D)
        x_t, y_t, z_t = tf.meshgrid(x, y, z)

        # flatten every x, y, z coordinates
        x_t_flat = tf.reshape(x_t, [-1])
        y_t_flat = tf.reshape(y_t, [-1])
        z_t_flat = tf.reshape(z_t, [-1])
        # x_t_flat has shape (H*W*D,)

        # reshape to (x_t, y_t, z_t, 1), which is homogeneous form
        ones = tf.ones_like(x_t_flat)
        sampling_grid = tf.stack([x_t_flat, y_t_flat, z_t_flat, ones])
        # sampling_grid now has shape (4, H*W*D)

        # repeat the grid num_batch times along axis=0 and num_parts times along axis=1
        sampling_grid = tf.expand_dims(sampling_grid, axis=0)
        sampling_grid = tf.expand_dims(sampling_grid, axis=1)
        sampling_grid = tf.tile(sampling_grid, [B, num_parts, 1, 1])
        # sampling grid now has shape (B, num_parts, 4, H*W*D)

        # cast to float32 (required for matmul)
        theta = tf.cast(theta, 'float32')
        sampling_grid = tf.cast(sampling_grid, 'float32')

        # transform the sampling grid using batch multiply
        batch_grids = theta @ sampling_grid
        # batch grid now has shape (B, num_parts, 3, H*W*D)

        # reshape to (B, num_parts, 3, H, W, D)
        batch_grids = tf.reshape(batch_grids, [B, num_parts, 3, H, W, D])

        return batch_grids

    def _trilinear_sampler(self, input_fmap, x, y, z):

        """
        :param input_fmap: the stacked decoded parts in the shape of (B, num_parts, H, W, D, C)
        :param x: x coordinate of input_fmap in the shape of (B, num_parts, H, W, D)
        :param y: y coordinate of input_fmap in the shape of (B, num_parts, H, W, D)
        :param z: z coordinate of input_fmap in the shape of (B, num_parts, H, W, D)
        :return: interpolated volume in the shape of (B, num_parts, H, W, D, C)
        """

        H = tf.shape(input_fmap)[2]
        W = tf.shape(input_fmap)[3]
        D = tf.shape(input_fmap)[4]
        C = tf.shape(input_fmap)[5]
        max_x = tf.cast(W - 1, 'int32')
        max_y = tf.cast(H - 1, 'int32')
        max_z = tf.cast(D - 1, 'int32')
        zero = tf.zeros([], dtype='int32')

        # rescale x/y/z to [0, W-1/H-1/D-1]
        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        z = tf.cast(z, 'float32')
        x = 0.5 * ((x + 1.0) * tf.cast(max_x - 1, 'float32'))
        y = 0.5 * ((y + 1.0) * tf.cast(max_y - 1, 'float32'))
        z = 0.5 * ((z + 1.0) * tf.cast(max_z - 1, 'float32'))

        # grab 8 nearest corner points for each (x_i, y_i, z_i) in input_fmap
        # 2*2*2 combination, so that there are 8 corner points in total
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1
        z0 = tf.cast(tf.floor(z), 'int32')
        z1 = z0 + 1

        # clip to range [0, H-1/W-1/D-1] to not violate boundaries
        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        z0 = tf.clip_by_value(z0, zero, max_z)
        z1 = tf.clip_by_value(z1, zero, max_z)

        # get voxel value at corner coords
        # pay attention to the difference of the ordering between real world coordinates and voxel coordinates
        c000 = self._get_voxel_value(input_fmap, x0, y1, z0)
        c001 = self._get_voxel_value(input_fmap, x0, y1, z1)
        c010 = self._get_voxel_value(input_fmap, x0, y0, z0)
        c011 = self._get_voxel_value(input_fmap, x0, y0, z1)
        c100 = self._get_voxel_value(input_fmap, x1, y1, z0)
        c101 = self._get_voxel_value(input_fmap, x1, y1, z1)
        c110 = self._get_voxel_value(input_fmap, x1, y0, z0)
        c111 = self._get_voxel_value(input_fmap, x1, y0, z1)
        # cxxx has shape (B, num_parts, H, W, D, C)

        # recast as float for delta calculation
        x0 = tf.cast(x0, 'float32')
        y1 = tf.cast(y1, 'float32')
        z0 = tf.cast(z0, 'float32')

        # calculate deltas
        xd = x - x0
        yd = y1 - y
        zd = z - z0

        # compute output (this is the  trilinear interpolation formula in real world coordinate system)
        output_fmap = list()
        for channel in range(C):
            output_fmap_channel = c000[:, :, :, :, :, channel] * (1 - xd) * (1 - yd) * (1 - zd) + c100[:, :, :, :, :, channel] * xd * (1 - yd) * (1 - zd) + \
                                  c010[:, :, :, :, :, channel] * (1 - xd) * yd * (1 - zd) + c001[:, :, :, :, :, channel] * (1 - xd) * (1 - yd) * zd + \
                                  c101[:, :, :, :, :, channel] * xd * (1 - yd) * zd + c011[:, :, :, :, :, channel] * (1 - xd) * yd * zd + \
                                  c110[:, :, :, :, :, channel] * xd * yd * (1 - zd) + c111[:, :, :, :, :, channel] * xd * yd * zd
            output_fmap.append(output_fmap_channel)
        return tf.stack(output_fmap, axis=5)

    @staticmethod
    def _get_voxel_value(input_fmap, x, y, z):

        """
        :param input_fmap: input feature map in the shape of (B, num_parts, H, W, D, C)
        :param x: x coordinates in the shape of (B, num_parts, H, W, D)
        :param y: y coordinates in the shape of (B, num_parts, H, W, D)
        :param z: z coordinates in the shape of (B, num_parts, H, W, D)
        :return: voxel value in the shape of (B, num_parts, H, W, D, C)
        """

        # pay attention to the difference of ordering between tensor indexing and voxel coordinates
        indices = tf.stack([y, x, z], axis=5)
        return tf.gather_nd(input_fmap, indices, batch_dims=2)


class STN(keras.layers.Layer):

    def __init__(self, num_parts, **kwargs):
        super(STN, self).__init__(**kwargs)
        self.localizationnet = LocalizationNet(num_parts)
        self.resampling = Resampling()

    def call(self, inputs, training=False):
        input_fmap = inputs[0]
        self.theta = self.localizationnet(inputs, training=training)
        resampling_inputs = (input_fmap, self.theta)
        output_fmap = self.resampling(resampling_inputs)
        return output_fmap


class Composer(keras.layers.Layer):

    def __init__(self, num_parts, **kwargs):
        super(Composer, self).__init__(**kwargs)
        self.part_decoder = SharedPartDecoder()
        self.stn = STN(num_parts)

    def call(self, inputs, training=False):
        decoder_outputs = list()
        for each in inputs:
            decoder_outputs.append(self.part_decoder(each, training=training))

        # stacked_decoded_part should be in the shape of (B, num_parts, H, W, D, C)
        self.stacked_decoded_parts = tf.stack(decoder_outputs, axis=1)
        # summed_inputs should be in the shape of (B, encoding_dims)
        summed_inputs = tf.reduce_sum(inputs, axis=0)
        localization_inputs = (self.stacked_decoded_parts, summed_inputs)
        # output_fmap has shape (B, num_parts, H, W, D, C)
        output_fmap = self.stn(localization_inputs, training=training)
        output_fmap = tf.where(output_fmap >= 0.5, 1., 0.)
        # reconstructed model shape has the shape of (B, H, W, D, C)
        reconstructed_shape = tf.reduce_sum(output_fmap, axis=1)
        reconstructed_shape = tf.where(reconstructed_shape >= 1., 1., 0.)

        resolution = tf.shape(output_fmap)[2]
        C = tf.shape(output_fmap)[5]
        vote_dict = dict()
        reconstructed_shape_label_list = list()
        for each_shape in output_fmap:
            for part_label, each_part in enumerate(each_shape):
                part_label_coord = tf.where(each_part >= 1.)
                for coord in part_label_coord:
                    code = int(coord[0] + coord[1] * resolution + coord[2] * resolution ** 2, coord[3] * C * resolution ** 2)
                    if code not in list(vote_dict.keys()):
                        vote_dict[code] = [part_label]
                    else:
                        vote_dict[code].append(part_label)
            reconstructed_part_label = tf.zeros((resolution, resolution, resolution, C),  dtype=tf.int32)
            for code in list(vote_dict.keys()):
                count_list = vote_dict[code]
                voxel_label = max(set(count_list), key=count_list.count)
                idx = [code % resolution, code // resolution % resolution, code // resolution // resolution % resolution,
                       code // resolution // resolution // resolution % C]
                reconstructed_part_label[idx[0], idx[1], idx[2], idx[3]] = voxel_label
            reconstructed_shape_label_list.append(reconstructed_part_label)
        reconstructed_shape_label = tf.convert_to_tensor(reconstructed_shape_label_list)
        return reconstructed_shape, reconstructed_shape_label


class Model(keras.Model):

    def __init__(self, num_parts, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.num_parts = num_parts
        self.training_process = None

        self.decomposer = Decomposer(num_parts)
        self.composer = Composer(num_parts)

        # create some loss tracker
        self.pi_loss_tracker = tf.keras.metrics.Mean()
        self.part_reconstruction_loss_tracker = tf.keras.metrics.Mean()
        self.transformation_loss_tracker = tf.keras.metrics.Mean()
        self.cycle_loss_tracker = tf.keras.metrics.Mean()
        self.total_loss_tracker = tf.keras.metrics.Mean()

    def choose_training_process(self, training_process):
        self.training_process = training_process

    def _cal_pi_loss(self):
        params = list()
        for each_layer in self.decomposer.projection_layer_list:
            params.append(each_layer.trainable_weights[0])
        # params should be list of tensor, whose elements are the trainable weights of Projection layer
        params_tensor = tf.convert_to_tensor(params)
        # params_tensor has shape (num_parts, encoding_dims, encoding_dims)
        pi_loss1 = tf.reduce_sum(tf.norm(params_tensor ** 2 - params_tensor, axis=[-2, -1]) ** 2)
        pi_loss2 = list()
        for idx, each_param in enumerate(params_tensor):
            unstack_params = tf.unstack(params_tensor, axis=0)
            del unstack_params[idx]
            new_params = tf.convert_to_tensor(unstack_params)
            pi_loss2.append(tf.norm(each_param * new_params, axis=[-2, -1]) ** 2)
        pi_loss2 = tf.reduce_sum(pi_loss2)
        pi_loss3 = tf.norm(tf.reduce_sum(params_tensor, axis=0) - tf.eye(tf.shape(params_tensor)[1])) ** 2
        return pi_loss1 + pi_loss2 + pi_loss3

    @staticmethod
    def _cal_part_reconstruction_loss(gt, pred):
        return tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(gt, pred), axis=(1, 2, 3, 4)))

    @staticmethod
    def _cal_transformation_loss(gt, pred):
        return tf.nn.l2_loss(gt-pred) / tf.cast(tf.shape(gt)[0], dtype=tf.float32)

    @staticmethod
    def _cal_cycle_loss(gt, pred):
        return tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(gt, pred), axis=(1, 2, 3)))

    @staticmethod
    def _generate_mixed_order(num_parts, batch_size):
        mixed_order_list = list()
        for i in range(num_parts):
            mixed_order_list.append(tf.convert_to_tensor(np.random.choice(batch_size, batch_size, False)))
        return tf.convert_to_tensor(mixed_order_list)

    @staticmethod
    def _generate_de_mixed_order(mixed_order):
        return tf.argsort(mixed_order, axis=1)

    @staticmethod
    def _mix(inputs, mixed_order, mode):
        mixed_list = list()
        if mode == 'label':
            inputs = tf.transpose(inputs, [1, 0, 2, 3, 4, 5])
            for each_input, each_order in zip(inputs, mixed_order):
                mixed_list.append(tf.gather(each_input, each_order, axis=0))
            mixed_output = tf.stack(mixed_list, axis=1)
        elif mode == 'encoding':
            for each_input, each_order in zip(inputs, mixed_order):
                mixed_list.append(tf.gather(each_input, each_order, axis=0))
            mixed_output = tf.convert_to_tensor(mixed_list)
        elif mode == 'transformation':
            inputs = tf.transpose(inputs, [1, 0, 2, 3])
            for each_input, each_order in zip(inputs, mixed_order):
                mixed_list.append(tf.gather(each_input, each_order, axis=0))
            mixed_output = tf.stack(mixed_list, axis=1)
        else:
            raise ValueError('\'mode\' should be \'label\', \'encoding\' or \'transformation\'.')
        return mixed_output

    def call(self, inputs, training=False, mode='mix', mixed_order=None):

        if self.training_process == 1:
            # decomposer output has shape (num_parts, B, encoding_dims)
            decomposer_output = self.decomposer(inputs, training=training)
            # composer output has shape (B, H, W, D, C)
            self.composer(decomposer_output, training=training)

        elif self.training_process == 2:
            # decomposer output has shape (num_parts, B, encoding_dims)
            decomposer_output = self.decomposer(inputs, training=training)
            # composer output has shape (B, H, W, D, C)
            self.composer(decomposer_output, training=training)
            return decomposer_output

        elif self.training_process == 3:
            # decomposer output has shape (num_parts, B, encoding_dims)
            decomposer_output = self.decomposer(inputs, training=training)
            if mode == 'mix':
                num_parts, B, _ = decomposer_output.shape
                # mixed order has shape (num_parts, B)
                order = self._generate_mixed_order(num_parts, B)
            elif mode == 'de_mix':
                order = self._generate_de_mixed_order(mixed_order)
            else:
                raise ValueError('mode should be one of mix and de_mix')
            # mixed decomposer output has shape (num_parts, B, encoding_dims)
            decomposer_output = self._mix(decomposer_output, order, mode='encoding')
            # composer output has shape (B, H, W, D, C)
            composer_output = self.composer(decomposer_output, training=training)
            return decomposer_output, composer_output, order

    def train_step(self, data):

        # x has shape (B, H, W, D, C), label has shape (B, num_parts, H, W, D, C), trans has shape (B, num_parts, 3, 4)
        x, label, trans = data

        if self.training_process == 1:  # training process for pretraining BinaryShapeEncoder, Projection, PartDecoder
            with tf.GradientTape() as tape:
                self.call(x, training=True)
                pi_loss = self._cal_pi_loss()
                part_recon_loss = self._cal_part_reconstruction_loss(label, self.composer.stacked_decoded_parts)
                total_loss = pi_loss + part_recon_loss

            decomposer_grads, decoder_grads = tape.gradient(total_loss, [self.decomposer.trainable_weights,
                                                                         self.composer.part_decoder.trainable_weights])
            self.optimizer.apply_gradients(zip(decomposer_grads, self.decomposer.trainable_weights))
            self.optimizer.apply_gradients(zip(decoder_grads, self.composer.part_decoder.trainable_weights))

            self.pi_loss_tracker.update_state(pi_loss)
            self.part_reconstruction_loss_tracker.update_state(part_recon_loss)
            self.total_loss_tracker.update_state(total_loss)

            return {'PI_Loss': self.pi_loss_tracker.result(),
                    'Part_Recon_Loss': self.part_reconstruction_loss_tracker.result(),
                    'Total_Loss': self.total_loss_tracker.result()}

        elif self.training_process == 2:  # training process for pretraining STN
            with tf.GradientTape() as tape:
                decomposer_output = self.call(x, training=True)
                num_parts, B, _ = decomposer_output.shape
                trans_loss = self._cal_transformation_loss(trans,
                                                           tf.reshape(self.composer.stn.theta, (B, num_parts, 3, 4)))

            grads = tape.gradient(trans_loss, self.composer.stn.localizationnet.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.composer.stn.localizationnet.trainable_weights))

            self.transformation_loss_tracker.update_state(trans_loss)

            return {'Transformation_Loss': self.transformation_loss_tracker.result()}

        elif self.training_process == 3:  # training process for fine tune all parameters using cycle loss
            with tf.GradientTape() as tape:
                # below is the first application:
                decomposer_output, composer_output, mixed_order = self.call(x, training=True, mode='mix')
                # mixed label has shape (B, num_parts, H, W, D, C)
                mixed_label = self._mix(label, mixed_order, mode='label')
                # mixed trans has shape (B, num_parts, 3, 4)
                mixed_trans = self._mix(trans, mixed_order, mode='transformation')
                # calculate PI loss
                pi_loss1 = self._cal_pi_loss()
                # calculate part reconstruction loss
                part_recon_loss1 = self._cal_part_reconstruction_loss(mixed_label, self.composer.stacked_decoded_parts)
                # calculate transformation loss
                num_parts, B, _ = decomposer_output.shape
                trans_loss1 = self._cal_transformation_loss(mixed_trans,
                                                            tf.reshape(self.composer.stn.theta, (B, num_parts, 3, 4)))

                # below is the second application:
                _, composer_output, _ = self.call(composer_output[0], training=True, mode='de_mix', mixed_order=mixed_order)
                de_mixed_label = label
                de_mixed_trans = trans
                pi_loss2 = self._cal_pi_loss()
                part_recon_loss2 = self._cal_part_reconstruction_loss(de_mixed_label,
                                                                      self.composer.stacked_decoded_parts)
                trans_loss2 = self._cal_transformation_loss(de_mixed_trans,
                                                            tf.reshape(self.composer.stn.theta, (B, num_parts, 3, 4)))

                pi_loss = pi_loss1 + pi_loss2
                part_recon_loss = part_recon_loss1 + part_recon_loss2
                trans_loss = trans_loss1 + trans_loss2
                cycle_loss = self._cal_cycle_loss(x, composer_output[0])
                total_loss = 0.1 * pi_loss + 100 * part_recon_loss + 0.1 * trans_loss + 0.1 * cycle_loss

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            self.pi_loss_tracker.update_state(pi_loss)
            self.part_reconstruction_loss_tracker.update_state(part_recon_loss)
            self.transformation_loss_tracker.update_state(trans_loss)
            self.cycle_loss_tracker.update_state(cycle_loss)
            self.total_loss_tracker.update_state(total_loss)

            return {'PI_Loss': self.pi_loss_tracker.result(),
                    'Part_Recon_Loss': self.part_reconstruction_loss_tracker.result(),
                    'Transformation_Loss': self.transformation_loss_tracker.result(),
                    'Cycle_Loss': self.cycle_loss_tracker.result(),
                    'Total_Loss': self.total_loss_tracker.result()}

        else:
            raise ValueError('Please choose training process. Valid value of training_process should be 1, 2 or 3.')

    @property
    def metrics(self):
        return [self.pi_loss_tracker, self.part_reconstruction_loss_tracker, self.transformation_loss_tracker,
                self.cycle_loss_tracker, self.total_loss_tracker]
