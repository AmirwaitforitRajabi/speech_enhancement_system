from enum import Enum
import shelve
import time
from enum import Enum
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.python.keras.callbacks as cbs
from tensorflow.python.keras import backend

from PJ00_Tools.InpuDataRead import DataAcquisition, DataScale, InputNumberOutputNumber
from PJ00_Tools.plot_learning import plot_learning_curves


class NeuronalNetworkType(Enum):
    original_cruse = 1,
    microsoft_cruse = 2,
    adjusted_cruse_1x2 = 3,
    crd320 = 4,
    crd512 = 5,





class ModelGenerator:
    def __init__(self, fft: int = 512, n1: int = 16, n2: int = 32, n3: int = 64, n4: int = 128, n5: int = 256, w1 = 1,w2=0.8
                 , frames: int = 15, n_neuron=1024, nn_type: NeuronalNetworkType = NeuronalNetworkType.original_cruse):
        self.validation_steps = None
        self.train_steps = None
        self.x_data_valid = None
        self.x_data_train = None
        self.data_loader = None
        self.FFT = fft
        self.frames = frames
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n4 = n4
        self.n5 = n5
        self.n_neuron = n_neuron
        self.w1 = w1
        self.w2 = w2
        self.nn_type = nn_type
        self.lr = None
        self.kernel_size = None
        self.strides = None
        self.model = None

    def create_model(self, learning_rate, kernel_size, strides, parallel=False, parallel_factor=2):
        self.kernel_size = kernel_size
        self.strides = strides
        adam = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        INPUT_Noisy_SHAPE = [self.frames, int(self.FFT / 2) + 1, 1]
        input_noisy = tf.keras.layers.Input(shape=INPUT_Noisy_SHAPE)
        input_noise_component = tf.keras.layers.Input(shape=(self.frames, int(self.FFT / 2) + 1), dtype=tf.complex64)
        if self.nn_type == NeuronalNetworkType.original_cruse:
            # layer 1
            c1 = tf.keras.layers.Conv2D(self.n1, kernel_size=self.kernel_size, strides=self.strides)(input_noisy)
            c1_b = tf.keras.layers.BatchNormalization()(c1)
            c1_activ = tf.keras.layers.ELU()(c1_b)

            c1_b_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c1_activ)
            #
            # skip 1
            c1_b_skip = c1_b_zero_padded
            #
            # layer 2
            c2 = tf.keras.layers.Conv2D(self.n2, kernel_size=kernel_size, strides=strides)(c1_b_zero_padded)
            c2_b = tf.keras.layers.BatchNormalization()(c2)
            c2_activ = tf.keras.layers.ELU()(c2_b)

            c2_b_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c2_activ)

            # skip 2
            c2_b_skip = c2_b_zero_padded

            # layer 3
            c3 = tf.keras.layers.Conv2D(self.n3, kernel_size=kernel_size, strides=strides)(c2_b_zero_padded)
            c3_b = tf.keras.layers.BatchNormalization()(c3)
            c3_activ = tf.keras.layers.ELU()(c3_b)

            c3_b_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c3_activ)

            # skip 3
            c3_b_skip = c3_b_zero_padded

            # layer 4
            c4 = tf.keras.layers.Conv2D(self.n4, kernel_size=kernel_size, strides=strides)(c3_b_zero_padded)
            c4_b = tf.keras.layers.BatchNormalization()(c4)
            c4_activ = tf.keras.layers.ELU()(c4_b)

            c4_b_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c4_activ)

            # skip 4
            c4_b_skip = c4_b_zero_padded

            # layer 5 ,kernel_regularizer=tf.keras.regularizers.l1_l1(l1=0.01, l1=0.01)
            c5 = tf.keras.layers.Conv2D(self.n5, kernel_size=kernel_size, strides=strides)(c4_b_zero_padded)
            c5_b = tf.keras.layers.BatchNormalization()(c5)
            c5_activ = tf.keras.layers.ELU()(c5_b)
            c5_b_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c5_activ)

            # skip 5
            c5_b_skip = c5_b_zero_padded

            # LSTM-Teil
            Re_shape = tf.keras.layers.Reshape((self.frames, self.n_neuron))(c5_b_zero_padded)
            if parallel and self.nn_type == NeuronalNetworkType.adjusted_cruse_1x2:
                R0 = tf.compat.v1.keras.layers.CuDNNLSTM(int(self.n_neuron / parallel_factor), return_sequences=True)(
                    Re_shape[:, :, :int(self.n_neuron / parallel_factor)])
                R1 = tf.compat.v1.keras.layers.CuDNNLSTM(int(self.n_neuron / parallel_factor), return_sequences=True)(
                    Re_shape[:, :, int(self.n_neuron / parallel_factor):])
                R_output = tf.concat([R0, R1], 2)

            else:
                R0 = tf.compat.v1.keras.layers.CuDNNLSTM(self.n_neuron, return_sequences=True)(Re_shape)
                R_output = tf.compat.v1.keras.layers.CuDNNLSTM(self.n_neuron, return_sequences=True)(R0)

            Re_Reshape = tf.keras.layers.Reshape((self.frames, 4, self.n5))(R_output)

            Skip1 = tf.concat([c5_b_skip, Re_Reshape], 3)

            CT0 = tf.keras.layers.Conv2DTranspose(self.n4, kernel_size=kernel_size, strides=strides, padding='valid')(Skip1)
            CT0_b = tf.keras.layers.BatchNormalization()(CT0)
            CT0_activ = tf.keras.layers.ELU()(CT0_b)
            CT0_re = tf.keras.layers.Lambda(lambda x: x[:, :-1, :])(CT0_activ)

            Skip2 = tf.concat([c4_b_skip, CT0_re], 3)

            CT1 = tf.keras.layers.Conv2DTranspose(self.n3, kernel_size=kernel_size, strides=strides, padding='valid')(Skip2)
            CT1_b = tf.keras.layers.BatchNormalization()(CT1)
            CT1_activ = tf.keras.layers.ELU()(CT1_b)
            CT1_re = tf.keras.layers.Lambda(lambda x: x[:, :-1, :])(CT1_activ)

            Skip3 = tf.concat([c3_b_skip, CT1_re], 3)

            CT2 = tf.keras.layers.Conv2DTranspose(self.n2, kernel_size=kernel_size, strides=strides, padding='valid')(Skip3)
            CT2_b = tf.keras.layers.BatchNormalization()(CT2)
            CT2_activ = tf.keras.layers.ELU()(CT2_b)
            CT2_re = tf.keras.layers.Lambda(lambda x: x[:, :-1, :])(CT2_activ)

            Skip4 = tf.concat([c2_b_skip, CT2_re], 3)

            CT3 = tf.keras.layers.Conv2DTranspose(self.n1, kernel_size=kernel_size, strides=strides, padding='valid')(Skip4)
            CT3_b = tf.keras.layers.BatchNormalization()(CT3)
            CT3_activ = tf.keras.layers.ELU()(CT3_b)

            Zero_Padding = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (0, 1)))(CT3_activ)
            CT3_re = tf.keras.layers.Lambda(lambda x: x[:, :-1, :])(Zero_Padding)

            Skip5 = tf.concat([c1_b_skip, CT3_re], 3)

            CT4 = tf.keras.layers.Conv2DTranspose(1, kernel_size=kernel_size, strides=strides, padding='valid')(Skip5)
            CT4_b = tf.keras.layers.BatchNormalization()(CT4)
            CT4_activ = tf.keras.activations.softplus(CT4_b)
            CT4_re = tf.keras.layers.Reshape([CT4_activ.shape[1], CT4_activ.shape[2] * CT4_activ.shape[3]])(CT4_activ)
            Output = tf.keras.layers.Lambda(lambda x: x[:, :-1, :])(CT4_re)

            cruse_model = tf.keras.Model(inputs=input_noisy, outputs=Output)
            cruse_model.summary()

            cruse_model.compile(optimizer=adam, loss=['mse'], metrics=['msle'])
            return cruse_model
        elif self.nn_type == NeuronalNetworkType.microsoft_cruse:

            # layer 1
            c1 = tf.keras.layers.Conv2D(self.n1, kernel_size=self.kernel_size, strides=self.strides)(input_noisy)
            c1_activ = tf.keras.layers.LeakyReLU()(c1)

            c1_b_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c1_activ)
            #
            # skip 1
            c1_b_skip = tf.keras.layers.Conv2D(self.n1, kernel_size=(1, 1), padding='same')(c1_b_zero_padded)
            #
            # layer 2
            c2 = tf.keras.layers.Conv2D(self.n2, kernel_size=self.kernel_size, strides=self.strides)(c1_b_zero_padded)
            c2_activ = tf.keras.layers.LeakyReLU()(c2)

            c2_b_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c2_activ)

            # skip 2
            c2_b_skip = tf.keras.layers.Conv2D(self.n2, kernel_size=(1, 1), padding='same')(c2_b_zero_padded)

            # layer 3
            c3 = tf.keras.layers.Conv2D(self.n3, kernel_size=self.kernel_size, strides=self.strides)(c2_b_zero_padded)
            c3_activ = tf.keras.layers.LeakyReLU()(c3)

            c3_b_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c3_activ)

            # skip 3
            c3_b_skip = tf.keras.layers.Conv2D(self.n3, kernel_size=(1, 1), padding='same')(c3_b_zero_padded)

            # layer 4
            c4 = tf.keras.layers.Conv2D(self.n4, kernel_size=self.kernel_size, strides=self.strides)(c3_b_zero_padded)
            c4_activ = tf.keras.layers.LeakyReLU()(c4)

            c4_b_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c4_activ)

            # skip 4
            c4_b_skip = tf.keras.layers.Conv2D(self.n4, kernel_size=(1, 1), padding='same')(c4_b_zero_padded)

            # LSTM-Teil
            Re_shape = tf.keras.layers.Reshape((self.frames, 9 * 128))(c4_b_zero_padded)

            R0 = tf.compat.v1.keras.layers.CuDNNGRU(288, return_sequences=True)(Re_shape[:, :, :288])
            R1 = tf.compat.v1.keras.layers.CuDNNGRU(288, return_sequences=True)(Re_shape[:, :, 288:576])
            R2 = tf.compat.v1.keras.layers.CuDNNGRU(288, return_sequences=True)(Re_shape[:, :, 576:864])
            R3 = tf.compat.v1.keras.layers.CuDNNGRU(288, return_sequences=True)(Re_shape[:, :, 864:1152])

            R_output = tf.concat([R0, R1, R2, R3], 2)

            Re_Reshape = tf.keras.layers.Reshape((self.frames, 9, 128))(R_output)

            Skip2 = tf.keras.layers.Add()([c4_b_skip, Re_Reshape])

            CT1 = tf.keras.layers.Conv2DTranspose(self.n3, kernel_size=self.kernel_size, strides=self.strides,
                                                  padding='valid')(Skip2)
            CT1_activ = tf.keras.layers.LeakyReLU()(CT1)
            CT1_re = tf.keras.layers.Lambda(lambda x: x[:, :-1, :])(CT1_activ)

            Skip3 = tf.keras.layers.Add()([c3_b_skip, CT1_re])

            CT2 = tf.keras.layers.Conv2DTranspose(self.n2, kernel_size=self.kernel_size, strides=self.strides,
                                                  padding='valid')(Skip3)
            CT2_activ = tf.keras.layers.LeakyReLU()(CT2)
            CT2_re = tf.keras.layers.Lambda(lambda x: x[:, :-1, :])(CT2_activ)

            Skip4 = tf.keras.layers.Add()([c2_b_skip, CT2_re])

            CT3 = tf.keras.layers.Conv2DTranspose(self.n1, kernel_size=self.kernel_size, strides=self.strides,
                                                  padding='valid')(Skip4)
            CT3_activ = tf.keras.layers.LeakyReLU()(CT3)
            CT3_re = tf.keras.layers.Lambda(lambda x: x[:, :-1, :])(CT3_activ)
            Zero_Padding = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (0, 1)))(CT3_re)

            Skip5 = tf.keras.layers.Add()([c1_b_skip, Zero_Padding])

            CT4 = tf.keras.layers.Conv2DTranspose(1, kernel_size=self.kernel_size, strides=self.strides, padding='valid')(
                Skip5)
            CT4_activ = tf.keras.activations.sigmoid(CT4)
            CT4_re = tf.keras.layers.Reshape([CT4_activ.shape[1], CT4_activ.shape[2] * CT4_activ.shape[3]])(CT4_activ)
            Output = tf.keras.layers.Lambda(lambda x: x[:, :-1, :])(CT4_re)
            Mask = tf.cast(Output, tf.complex64)
            Output_magnitude = tf.keras.layers.Multiply()([Mask, input_noise_component])
            microsoft_model = tf.keras.Model(inputs=[input_noisy, input_noise_component], outputs=[Output_magnitude])


            microsoft_model.compile(optimizer=adam, loss=self.custom_loss_phase_aware, metrics=['msle'], run_eagerly=True)

            microsoft_model.summary()
            return microsoft_model
        elif self.nn_type == NeuronalNetworkType.crd320:

            c0 = tf.keras.layers.Conv2D(1, kernel_size=(1, 1), padding='same')(input_noisy)
            c0_activ = tf.keras.layers.ELU()(c0)
            c0_activ_b = tf.keras.layers.BatchNormalization()(c0_activ)
            c1 = tf.keras.layers.Conv2D(self.n2, kernel_size=self.kernel_size, strides=self.strides)(c0_activ_b)
            c1_activ = tf.keras.layers.ELU()(c1)
            c1_activ_b = tf.keras.layers.BatchNormalization()(c1_activ)
            c1_activ_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c1_activ_b)
            c2 = tf.keras.layers.Conv2D(self.n2, kernel_size=self.kernel_size, strides=self.strides)(c1_activ_zero_padded)
            c2_activ = tf.keras.layers.ELU()(c2)
            c2_activ_b = tf.keras.layers.BatchNormalization()(c2_activ)
            c2_activ_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c2_activ_b)
            c3 = tf.keras.layers.Conv2D(self.n2, kernel_size=self.kernel_size, strides=self.strides)(c2_activ_zero_padded)
            c3_activ = tf.keras.layers.ELU()(c3)
            c3_activ_b = tf.keras.layers.BatchNormalization()(c3_activ)
            c3_activ_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c3_activ_b)
            c4 = tf.keras.layers.Conv2D(self.n2, kernel_size=self.kernel_size, strides=self.strides)(c3_activ_zero_padded)
            c4_activ = tf.keras.layers.ELU()(c4)
            c4_activ_b = tf.keras.layers.BatchNormalization()(c4_activ)
            c4_activ_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c4_activ_b)
            c1_reshape = tf.keras.layers.Reshape(
                (self.frames, c4_activ_zero_padded.shape[2] * c4_activ_zero_padded.shape[3]))(
                c4_activ_zero_padded)

            Dense1 = tf.keras.layers.Dense(c4_activ_zero_padded.shape[2] * c4_activ_zero_padded.shape[3],
                                           activation=tf.keras.layers.ELU())(c1_reshape)
            Dense1_dr = tf.keras.layers.Dropout(0.1)(Dense1)
            #first half
            R0 = tf.compat.v1.keras.layers.CuDNNGRU(int(Dense1.shape[2]/2), return_sequences=True)(Dense1_dr[:, :, :144])
            R0_act = tf.keras.layers.ELU()(R0)
            R0_1 = tf.compat.v1.keras.layers.CuDNNGRU(144, return_sequences=True)(R0_act)
            R0_1_act = tf.keras.layers.ELU()(R0_1)
            #second half
            R1 = tf.compat.v1.keras.layers.CuDNNGRU(144, return_sequences=True)(Dense1_dr[:, :, 144:])
            R1_act = tf.keras.layers.ELU()(R1)
            R1_1 = tf.compat.v1.keras.layers.CuDNNGRU(144, return_sequences=True)(R1_act)
            R1_1_act = tf.keras.layers.ELU()(R1_1)
            R_output = tf.concat([R0_1_act, R1_1_act], 2)

            Dense2 = tf.keras.layers.Dense(288, activation=tf.keras.layers.ELU())(R_output)
            Dense2_dr = tf.keras.layers.Dropout(0.1)(Dense2)
            last_dense = tf.keras.layers.Dense(int(self.FFT / 2 + 1), activation=tf.keras.layers.ELU())(Dense2_dr)
            #last_dense = tf.keras.activations.sigmoid(last_dense)
            # calculating the magnitude
            Mask = tf.cast(last_dense, tf.complex64)
            Output_magnitude = tf.keras.layers.Multiply()([Mask, input_noise_component])
            crd_model = tf.keras.Model(inputs=[input_noisy, input_noise_component], outputs=Output_magnitude)
            crd_model.compile(optimizer=adam, loss=self.custom_loss_phase_aware, metrics='mae', run_eagerly=True)

            crd_model.summary()
            return  crd_model
        elif self.nn_type == NeuronalNetworkType.crd512:
            #muss noch gemacht werden.
            c0 = tf.keras.layers.Conv2D(1, kernel_size=(1, 1), padding='same')(input_noisy)
            c0_activ = tf.keras.layers.ELU()(c0)
            # c0_activ_b = tf.keras.layers.BatchNormalization()(c0_activ)
            c1 = tf.keras.layers.Conv2D(64, kernel_size=(2, 3), strides=(1, 2))(c0_activ)
            c1_activ = tf.keras.layers.ELU()(c1)
            # c1_activ_b = tf.keras.layers.BatchNormalization()(c1_activ)
            c1_activ_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c1_activ)
            c2 = tf.keras.layers.Conv2D(64, kernel_size=(2, 3), strides=(1, 2))(c1_activ_zero_padded)
            c2_activ = tf.keras.layers.ELU()(c2)
            # c2_activ_b = tf.keras.layers.BatchNormalization()(c2_activ)
            c2_activ_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c2_activ)
            c3 = tf.keras.layers.Conv2D(32, kernel_size=(2, 3), strides=(1, 2))(c2_activ_zero_padded)
            c3_activ = tf.keras.layers.ELU()(c3)
            # c3_activ_b = tf.keras.layers.BatchNormalization()(c3_activ)
            c3_activ_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c3_activ)
            c4 = tf.keras.layers.Conv2D(32, kernel_size=(2, 3), strides=(1, 2))(c3_activ_zero_padded)
            c4_activ = tf.keras.layers.ELU()(c4)
            # c4_activ_b = tf.keras.layers.BatchNormalization()(c4_activ)
            c4_activ_zero_padded = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(c4_activ)
            c1_reshape = tf.keras.layers.Reshape((200, c4_activ_zero_padded.shape[2] * c4_activ_zero_padded.shape[3]))(
                c4_activ_zero_padded)

            Dense1 = tf.keras.layers.Dense(c4_activ_zero_padded.shape[2] * c4_activ_zero_padded.shape[3],
                                           activation=tf.keras.layers.ELU())(c1_reshape)
            Dense1_dr = tf.keras.layers.Dropout(0.1)(Dense1)

            RNN1 = tf.compat.v1.keras.layers.CuDNNGRU(500, return_sequences=True)(Dense1_dr)
            RNN1_act = tf.keras.layers.ELU()(RNN1)
            RNN1_dr = tf.keras.layers.Dropout(0.1)(RNN1_act)
            RNN2 = tf.compat.v1.keras.layers.CuDNNGRU(400, return_sequences=True)(RNN1_dr)
            RNN2_act = tf.keras.layers.ELU()(RNN2)
            RNN2_dr = tf.keras.layers.Dropout(0.1)(RNN2_act)
            Dense2 = tf.keras.layers.Dense(300, activation=tf.keras.layers.ELU())(RNN2_dr)
            Dense2_dr = tf.keras.layers.Dropout(0.1)(Dense2)
            last_dense = tf.keras.layers.Dense(257, activation=tf.keras.layers.ELU())(Dense2_dr)
            Mask = tf.cast(last_dense, tf.complex64)
            Output_magnitude = tf.keras.layers.Multiply()([Mask, input_noise_component])

            model = tf.keras.Model(inputs=[input_noisy, input_noise_component], outputs=[Output_magnitude])
            adam = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

            model.compile(optimizer=adam, loss=self.custom_loss_phase_aware, metrics=['msle'], run_eagerly=True)
            model.summary()

    def read_data(self, input_data_path, mini_batch, repeat_factor,data_scale,siso = True):
        if not siso:
            in_out = InputNumberOutputNumber.miso
            self.data_loader = DataAcquisition(minibatch_size=mini_batch, repeat_faktor=repeat_factor, n_frames=self.frames,
                                           n_fft=self.FFT,
                                           input_type=data_scale,in_out = in_out)
        else:
            self.data_loader = DataAcquisition(minibatch_size=mini_batch, repeat_faktor=repeat_factor, n_frames=self.frames,
                                           n_fft=self.FFT,
                                           input_type=data_scale)

        # add the training und validation data to the generator
        self.x_data_train = self.data_loader.read_data(data_path=input_data_path, mode='train')
        self.x_data_valid = self.data_loader.read_data(data_path=input_data_path, mode='valid')

        # calculate steps_per_epoch

        num_train_examples = len(self.data_loader.train_audio_paths)
        if mini_batch > num_train_examples:
            print('set a smaller number for the mini_batch...')
        self.train_steps = num_train_examples // mini_batch

        num_valid_samples = len(self.data_loader.valid_audio_paths)
        self.validation_steps = num_valid_samples // mini_batch

    def model_train(self, model, result_path, nb_epochs, batch_size):
        startAll = time.time()
        backend.clear_session()
        # Reduce learning rate when stop improving lr = lr*factor
        reduce_LR = cbs.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=8, verbose=1, mode='auto')

        # Stop training after 1000 epoches if the vali_loss not decreasing
        stop_str = cbs.EarlyStopping(monitor='val_loss', patience=24, verbose=1, mode='auto')

        checkpoints_path = result_path.joinpath('Checkpoints')
        if not checkpoints_path.exists():
            checkpoints_path.mkdir(parents=True, exist_ok=True)
        weights_path = checkpoints_path.joinpath('resetmodel{epoch:04d}.h5')
        checkpoint = cbs.ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True)

        callbacks_list = [checkpoint]

        history = model.fit(self.data_loader.batch_train(self.train_steps), batch_size=batch_size,
                            steps_per_epoch=self.train_steps,
                            epochs=nb_epochs,
                            validation_data=self.data_loader.batch_valid(self.validation_steps),
                            validation_steps=self.validation_steps,
                            callbacks=[callbacks_list, reduce_LR, stop_str], verbose=1, shuffle=True)
        save_model_path = checkpoints_path.joinpath('resetmodel0000.h5')
        model.save(save_model_path)

        # tf.keras.utils.plot_model(model, to_file=result_path.joinpath('Flussdiagramm.png'), dpi=100)

        # string:DDDD(.name)
        ff = shelve.open(str(result_path.joinpath('history.slv')))
        ff['train_loss'] = history.history['loss']
        ff['val_loss'] = history.history['val_loss']
        ff.close()
        plot_learning_curves(history.history['loss'], history.history['val_loss'], result_path)
        plt.clf()
        print("> All Trainings Completed, Duration : ", time.time() - startAll)

    def _inverse_fft_func(self,fft):
        magnitude = tf.reshape(fft, [fft.shape[0] * fft.shape[1], fft.shape[2]])
        rec = tf.signal.inverse_stft(magnitude, frame_length=self.FFT, frame_step=int(self.FFT/ 2),
                                     window_fn=tf.signal.hann_window)
        magnitude_enh = tf.signal.stft(rec, frame_length=self.FFT, frame_step=int(self.FFT / 2), window_fn=tf.signal.hann_window)
        return tf.cast(rec, dtype=tf.float64), tf.cast(magnitude_enh, dtype=tf.complex64)

    def custom_loss_phase_aware(self, y_true, y_predict):
        s, S = self._inverse_fft_func(y_true)

        x_tilde, X_tilde = self._inverse_fft_func(y_predict)

        return self.w1 * K.mean(K.square(tf.abs(tf.abs(S) - tf.abs(X_tilde))), axis=-1) + self.w2 * K.mean(
            K.square(tf.abs(S - X_tilde)), axis=-1)