import shelve
import enum

import soundfile as sf
import librosa
import numpy as np
from sklearn import preprocessing
import tensorflow.python.keras.backend as backend
from keras.models import load_model



class TimeStorage:
    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time


class TimeController:
    def __init__(self, read, predict):
        self.read: TimeStorage = read
        self.predict: TimeStorage = predict

    def __str__(self):
        output = "Read"
        formatted_duration = '{:5.3f}s'.format(self.read.duration)
        output = f"{output}\n{formatted_duration}\nPredict"
        formatted_duration = '{:5.3f}s'.format(self.predict.duration)
        output = f"{output}\n{formatted_duration}"
        return output


class DataScale(enum.Enum):
    logarithmic = 0,
    normalized_logarithmic = 1,
    not_logarithmic = 2

class ShapeFrom(enum.Enum):
    causal = 0,
    look_backward = 1

class ModelType(enum.Enum):
    end_to_end = 0,
    mask_based = 1


class ShelveFile:
    __slots__ = ["file_lengths", "complex_spectrogram_test_noisy", "abs_spectrogram_noisy", "file_names", "phase_noisy", "length", "shapes_of_heart",]
    def __init__(self, file_lengths, complex_spectrogram_test_noisy, abs_spectrogram_noisy, file_names, phase_noisy, shapes):
        self.file_lengths = file_lengths
        self.complex_spectrogram_test_noisy = complex_spectrogram_test_noisy
        self.abs_spectrogram_noisy = abs_spectrogram_noisy
        self.file_names = file_names
        self.phase_noisy = phase_noisy
        self.shapes_of_heart = shapes
        self.length = np.sum(self.file_lengths)


class PredictReconstruct:
    def __init__(self, FFT: int = 512, frames: int = 20, minibatch: int = 15, look_backward: int = 2,batch_size = 16,
                 input_type: DataScale = DataScale.logarithmic,
                 nn_type: ModelType = ModelType.mask_based,
                 shape_from: ShapeFrom = ShapeFrom.causal):
        self.noisy_path = None
        self.FFT = FFT
        self.batch_size = batch_size
        self.test_index = 0
        self.look_backward = look_backward
        self.shape_from = shape_from
        self.block_length = int(self.FFT / 2) + 1
        self.frames = frames
        self.minibatch = minibatch
        self.nn_type = nn_type
        self.input_type = input_type
        self.current_shelve_file: ShelveFile = None

    def __del__(self):
        print('class is deleted!!!')

    def _read_data_indicis(self, test_path):
        abs_spectrogram_noisy, complex_spectrogram_noisy, file_lengths, file_names, phase_noisy, shapes = [], [], [], [], [], []

        for path in test_path:
            # loose .dat from path string and open shelve
            data_input = shelve.open(str(path)[:-4])
            #print(list(data_input.keys()))
            abs = data_input['abs_spectrogram_noisy']
            mag = data_input['complex_spectrogram_noisy']
            file_length = data_input['file_length_noisy']
            file_name = data_input['file_name_noisy']
            phases = data_input['phase_noisy']
            shape = data_input['frame_length_noisy']
            data_input.close()

            file_lengths.append(file_length)
            phase_noisy.append(phases)
            shapes.append(shape)
            abs_spectrogram_noisy.append(abs)
            complex_spectrogram_noisy.append(mag)
            file_names.append(file_name)
        abs_spectrogram_noisy = np.concatenate(abs_spectrogram_noisy, axis=0)
        complex_spectrogram_test_noisy = np.concatenate(complex_spectrogram_noisy, axis=0)
        phase_noisy = np.concatenate(phase_noisy, axis=0)
        self.lengths = np.sum(file_lengths)

        self.current_shelve_file = ShelveFile(file_lengths, complex_spectrogram_test_noisy, abs_spectrogram_noisy,
                                              file_names,
                                              phase_noisy,shapes)

    def test_prediction(self, model_path,test_data_path):
        print('> Loading Models... ')
        model_paths = list(model_path.joinpath('checkpoints').glob( "**/*.h5"))
        # set the path of .slv of your test dataset for prediction phase
        test_path = test_data_path.joinpath('test','slv','noisy',str(self.FFT))
        noisy_path = list(test_path.glob("**/*.slv.dat"))
        for i in range(len(model_paths)):
            model = load_model(model_paths[i], compile=False)
            if i == 0:
                model.summary()
            print('     The predicited Model: %s' % str(model_paths[i].name))

            for k in range(int(len(noisy_path)/self.minibatch)+1):
                if self.test_index + self.minibatch < len(noisy_path):
                    path_small = [a for a in noisy_path[self.test_index:self.test_index + self.minibatch]]
                else:
                    path_small = [a for a in noisy_path[self.test_index - self.minibatch:]]
                self._read_data_indicis(test_path=path_small)

                if self.input_type == DataScale.logarithmic:
                    test_input_noisy_abs = np.log10(self.current_shelve_file.abs_spectrogram_noisy ** 2)

                elif self.input_type == DataScale.normalized_logarithmic:
                    scaler = preprocessing.StandardScaler()
                    test_input_noisy_abs = scaler.fit_transform(
                        np.log10(self.current_shelve_file.abs_spectrogram_noisy ** 2))

                elif self.input_type == DataScale.not_logarithmic:
                    test_input_noisy_abs = self.current_shelve_file.abs_spectrogram_noisy
                else:
                    return print('test_input_noisy_abs is not exist!!!!')
                if self.shape_from == ShapeFrom.causal:
                    test_input_noisy_abs_reshaped_4D, test_input_noisy_abs_3D = self._input_reshape(test_input_noisy_abs)
                    _, input_test_noisy_complex_reshaped_3D = self._input_reshape(
                        self.current_shelve_file.complex_spectrogram_test_noisy)
                elif self.shape_from == ShapeFrom.look_backward:
                    test_input_noisy_abs_reshaped_4D = self._reshape_backward(test_input_noisy_abs,
                                                                           look_backward=self.look_backward,
                                                                           look_forward=0,
                                                                           complex_value=False)
                    input_test_noisy_complex_reshaped_3D = np.reshape(
                        self.current_shelve_file.complex_spectrogram_test_noisy, (-1, self.block_length, 1))
                else:
                    #set all the input data equal zero and exit
                    test_input_noisy_abs_reshaped_4D = np.zeros_like(test_input_noisy_abs_3D)
                    input_test_noisy_complex_reshaped_3D = np.zeros_like(test_input_noisy_abs_3D)
                    print('test_input_noisy_abs is not exist!!!!')

                if self.nn_type == ModelType.end_to_end:
                    results_on_input_noisy = model.predict([test_input_noisy_abs_reshaped_4D], batch_size=self.batch_size )

                elif self.nn_type == ModelType.mask_based:
                    results_on_input_noisy = model.predict([test_input_noisy_abs_reshaped_4D,input_test_noisy_complex_reshaped_3D], batch_size=self.batch_size)
                else:
                    return print('Results are not exist!!!!')


                reconstructed_enhanced_audio = self._calc_enhanced_spec(nn_output=results_on_input_noisy)

            #change the index of the test data for the next minibatch
                self._create_wav_files(reconstructed_enhanced_audio, model_paths, i)
                self.test_index += self.minibatch
            if self.test_index >= len(noisy_path) - self.minibatch:
                self.test_index = 0

    def _create_wav_files(self, reconstructed_enhanced_audio, model_paths, i):

        current_file_position = 0
        intern = model_paths[0].parents[1]
        for j in range(len(self.current_shelve_file.file_lengths)):

            temp_data = reconstructed_enhanced_audio[
                        current_file_position:(current_file_position + self.current_shelve_file.shapes_of_heart[j])]
            current_file_position = current_file_position + self.current_shelve_file.shapes_of_heart[j]

            rec = librosa.istft(temp_data.T, window='hann', hop_length=int(self.FFT / 2), win_length=self.FFT, center=True,
                                length=self.current_shelve_file.file_lengths[j])
            destination_2 = intern.joinpath('enhanced_audio','final_enhanced_audio'+'_'+model_paths[i].name[10:14])

            if not destination_2.exists():
                destination_2.mkdir(parents=True, exist_ok=True)

            name = f"{self.current_shelve_file.file_names[j][:-4]}.{'wav'}"
            sf.write(destination_2.joinpath(name), rec, 16000, 'PCM_24')
            backend.clear_session()

    def _input_reshape(self, data):
        wert = data.shape[0] % self.frames
        if wert > 0:
            data = np.append(data, np.zeros([self.frames - wert, self.block_length]), axis=0)
        data_3d = data.reshape((int(data.shape[0] / self.frames), self.frames, self.block_length))
        data_4d = data_3d.reshape([-1, data_3d.shape[1], data_3d.shape[2], 1])
        return data_4d, data_3d

    def _reshape_backward(self, data, look_backward, look_forward, complex_value=False, _4D=True):

        new_dim_len = look_forward + look_backward + 1
        if complex_value:
            data_matrix_out = np.zeros((data.shape[0], new_dim_len, self.block_length), dtype=np.complex)
        else:
            data_matrix_out = np.zeros((data.shape[0], new_dim_len, self.block_length))
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

    def _calc_enhanced_spec(self, nn_output):
        nn_output = nn_output.reshape(
            nn_output.shape[0] * nn_output.shape[1], nn_output.shape[2])

        wert = self.current_shelve_file.abs_spectrogram_noisy.shape[0] % self.frames
        if wert > 0:
            diff = self.frames - wert
            nn_output = nn_output[0:-diff]
        else:
            pass

        if self.nn_type == ModelType.end_to_end:
            return np.abs(nn_output) * self.current_shelve_file.phase_noisy
        else:
            return nn_output

