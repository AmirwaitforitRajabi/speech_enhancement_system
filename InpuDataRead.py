from enum import Enum
import numpy as np
import random
import shelve
from itertools import repeat
from sklearn import preprocessing
import tensorflow as tf
from keras import backend as K
np.random.seed(1337)


class ShapeFrom(Enum):
    casual = 0,
    look_backward = 1,

class DataScale(Enum):
    logarithmic = 0,
    normalized_logarithmic = 1,
    not_logarithmic = 2
class InputNumberOutputNumber(Enum):
    siso = 1,
    miso = 2,

class DataAcquisition:

    def __init__(self, minibatch_size: int = 16, n_frames: int = 250, n_fft: int = 320,
                 repeat_faktor: int = 1,
                 input_type: DataScale = DataScale.logarithmic,
                 shape_form: ShapeFrom = ShapeFrom.casual,
                 in_out: InputNumberOutputNumber = InputNumberOutputNumber.siso,
                 look_forward: int = 0,
                 look_backward: int = 6):
        self.valid_noisy = None
        self.valid_audio_paths = None
        self.train_noisy = None
        self.train_audio_paths = None
        self.noisy_path = None
        self.clean_path = None
        self.audio_noisy_paths = None
        self.audio_clean_paths = None
        self.nn_type = in_out
        self.input_type = input_type
        self.shape_from = shape_form
        self.look_forward = look_forward
        self.look_backward = look_backward
        self.train_index = 0
        self.valid_index = 0
        self.test_index = 0
        self.repeat_faktor = repeat_faktor
        self.minibatch_size = minibatch_size
        self.n_frames = n_frames
        self.n_fft = n_fft
        self.overlap = int(n_fft / 2) + 1


    def read_data(self, data_path, mode):

        self.clean_path = data_path.joinpath(mode, 'slv', 'clean')
        self.noisy_path = data_path.joinpath(mode, 'slv', 'noisy')
        clean_path_wav = list(self.clean_path.glob("**/*.slv.dat"))
        clean_path_wav = [p for p in clean_path_wav for p in repeat(p, self.repeat_faktor)]
        noisy_path_wav = list(self.noisy_path.glob("**/*.slv.dat"))

        map_index = list(zip(clean_path_wav, noisy_path_wav))
        random.shuffle(map_index)
        audio_clean_paths, audio_noisy_paths = zip(*map_index)


        if mode == 'train':
            self.train_audio_paths = audio_clean_paths
            self.train_noisy = audio_noisy_paths

        elif mode == 'valid':
            self.valid_audio_paths = audio_clean_paths
            self.valid_noisy = audio_noisy_paths

    def _get_batch(self,mode):

        if mode == 'train':
            cur_index = self.train_index
            audio_clean_paths = self.train_audio_paths
            audio_noisy_paths = self.train_noisy
        elif mode == 'valid':
            cur_index = self.valid_index
            audio_clean_paths = self.valid_audio_paths
            audio_noisy_paths = self.valid_noisy
        else:
            return
        features_noisy = [a for a in audio_noisy_paths[cur_index:cur_index + self.minibatch_size]]
        features_clean = [b for b in audio_clean_paths[cur_index:cur_index + self.minibatch_size]]

        features_index = list(zip(features_clean, features_noisy))
        random.shuffle(features_index)
        features_clean, features_noisy = zip(*features_index)

        out_put_matrix_3d_complex = self._import_data(features_clean, state='clean')
        input_matrix_4d, input_matrix_3d = self._import_data(features_noisy, state='noisy')

        if self.nn_type == InputNumberOutputNumber.siso:
            return [input_matrix_4d], [np.abs(out_put_matrix_3d_complex)]
        elif self.nn_type == InputNumberOutputNumber.miso:
            return [input_matrix_4d,input_matrix_3d], [out_put_matrix_3d_complex]
        else:
            print('no data is available!!!!!')

    def batch_train(self, range_audio):
        while True:
            out = self._get_batch('train')
            self.train_index += self.minibatch_size
            if self.train_index >= range_audio - self.minibatch_size:
                self.train_index = 0
            yield out

    def batch_valid(self, range_audio):
        while True:
            out = self._get_batch('valid')
            self.valid_index += self.minibatch_size
            if self.valid_index >= range_audio - self.minibatch_size:
                self.valid_index = 0
            yield out

    def _import_data(self, path_list,state):
        # _________________________________________________________________________________________________________
        # read data and get data from every shelve
        magnitude_list = []
        amplitude_list = []
        if state == 'clean':
            for path in path_list:
                # loose .dat from path string and open shelve
                data_input = shelve.open(str(path)[:-4])
                #print(list(data_input.keys()))
                magnitude = data_input['complex_spectrogram_' + state]
                data_input.close()
                magnitude_list.append(magnitude)

            if self.shape_from == ShapeFrom.casual:
                return self._reshape(np.concatenate(magnitude_list, axis=0), _3d=True)
            elif self.shape_from == ShapeFrom.look_backward:
                return np.reshape(np.concatenate(magnitude_list, axis=0),(-1,self.overlap,1))

        if state == 'noisy':
            for path in path_list:
                # loose .dat from path string and open shelve
                data_input = shelve.open(str(path)[:-4])

                #print(list(data_input.keys()))
                magnitude = data_input['complex_spectrogram_' + state]
                amplitude = data_input['abs_spectrogram_' + state]
                data_input.close()
                magnitude_list.append(magnitude)
                amplitude_list.append(amplitude)
            amplitude_list = np.concatenate(amplitude_list, axis=0)
            magnitude_list = np.concatenate(magnitude_list, axis=0)

            if self.input_type == DataScale.not_logarithmic:
                pass
            # elif self.input_type == DataScale.logarithmic:
            #     amplitude_list = np.log10(amplitude_list ** 2 + np.finfo(np.float32).eps)
            # elif self.input_type == DataScale.normalized_logarithmic:
            #     amplitude_list = self._normalization(np.concatenate(amplitude_list, axis=0))
            # else:
            #     return print('DataScale mode is not valid')

            if self.shape_from == ShapeFrom.casual:
                amplitude_list = self._reshape(amplitude_list, _3d=False)
                magnitude_list = self._reshape(magnitude_list, _3d=True)
                return amplitude_list, magnitude_list
            elif self.shape_from == ShapeFrom.look_backward:
                amplitude_list = self._reshape_backward(amplitude_list, look_backward=self.look_backward,
                                                        look_forward=self.look_forward, _4D=True)
                magnitude_list = np.reshape(magnitude_list, (-1, self.overlap, 1))

                return amplitude_list, magnitude_list



    def _normalization(self, data):
        data_abs_log_power_2 = np.log10(data ** 2)
        scaler = preprocessing.StandardScaler()
        data_log_power_2_scaled = scaler.fit_transform(data_abs_log_power_2)
        return data_log_power_2_scaled

    def _reshape(self, data, _3d=True):
        wert = data.shape[0] % self.n_frames
        if wert > 0:
            data = np.append(data, np.zeros([self.n_frames - wert, self.overlap]), axis=0)

        if _3d:
            return data.reshape((int((data.shape[0] / self.n_frames)), self.n_frames, self.overlap))
        else:
            return data.reshape((int((data.shape[0] / self.n_frames)), self.n_frames, self.overlap, 1))

    def _reshape_backward(self, data, look_backward, look_forward, complex_value=False, _4D=True):

        new_dim_len = look_forward + look_backward + 1
        if complex_value:
            data_matrix_out = np.zeros((data.shape[0], new_dim_len, self.overlap), dtype=np.complex)
        else:
            data_matrix_out = np.zeros((data.shape[0], new_dim_len, self.overlap))
        for i in range(data.shape[0]):
            for j in range(-look_backward, look_forward + 1):
                if i < look_backward:
                    idx = max(i + j, 0)
                elif i >= data.shape[0] - look_forward:
                    idx = min(i + j, data.shape[0] - 1)
                else:
                    idx = i + j
                data_matrix_out[i, j + look_backward] = data[idx, :]
        if _4D:
            return data_matrix_out.reshape([-1, data_matrix_out.shape[1], data_matrix_out.shape[2], 1])
        else:
            return data_matrix_out



    def _inverse_fft_func(self,fft):
        mag = tf.reshape(fft, [fft.shape[0]*fft.shape[1],fft.shape[2]])
        rec = tf.signal.inverse_stft(mag, frame_length=self.n_fft, frame_step=int(self.n_fft / 2), window_fn=tf.signal.hann_window)
        mag_enh = tf.signal.stft(rec, frame_length=self.n_fft, frame_step=int(self.n_fft / 2), window_fn=tf.signal.hann_window)
        return rec, mag_enh

    def custom_loss(self,y_true, y_predict):

        rec_true, y_true_d_type = self._inverse_fft_func(y_true)
        rec_predict, y_predict_d_type = self._inverse_fft_func(y_predict)
        y_true_d_type = tf.cast(y_true_d_type, dtype=tf.complex64)
        y_predict_d_type = tf.cast(y_predict_d_type, dtype=tf.complex64)
        return 0.7 * K.square(tf.abs(tf.abs(y_true_d_type) - tf.abs(y_predict_d_type)), axis=-1) + 0.3 * K.square(
            tf.abs(y_true_d_type - y_predict_d_type), axis=-1)
