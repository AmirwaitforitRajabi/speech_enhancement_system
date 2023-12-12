import random
import scipy.signal
import soundfile
import pickle
import shelve
import librosa
from enum import Enum
import concurrent.futures
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut

class MixMethode(Enum):
    my_mix_methode = 1
    gent_mix_methode = 2
    just_clean_data = 3
    th_mix_methode = 4


class DataAugmentation(Enum):
    filter_independently = 1
    filter_dependently = 2
    no_filter = 3


class Mode(Enum):
    train_und_valid = 1
    test_noisy = 2


class SignalPower:
    def __init__(self):
        pass

    def calculate_rms_level(self, x):
        if isinstance(x[0], np.int16):
            x = x / 2 ** 15
        if max(abs(x)) > 1:
            raise ValueError('Normalize signal to (-1, 1) first.')

        EPS = np.finfo("float").eps
        return 10 * np.log10(np.mean(x ** 2) + EPS)

    def calculate_active_speech_level(self, x, fs):
        if isinstance(x[0], np.int16):
            x = x / 2 ** 15

        if max(abs(x)) > 1:
            raise ValueError('Normalize signal to (-1, 1) first.')

        # parameters
        T = 0.03  # time constant of smoothing in seconds
        g = np.exp(-1 / (T * fs))  # coefficient of smoothing
        H = 0.20  # Hangover time in seconds
        I = int(np.ceil(H * fs))
        M = 15.9  # Margin between c_dB and a_dB
        nbits = 16

        a = np.zeros(nbits - 1)  # activity count
        c = 0.5 ** np.arange(nbits - 1, 0, step=-1)  # threshold level

        # initialize
        h = np.ones(nbits) * I  # Hangover count
        sq = 0
        p = 0
        q = 0
        asl = -100

        sq = sum(x ** 2)
        c_dB = 20 * np.log10(c)

        for xi in x:
            p = g * p + (1 - g) * abs(xi)
            q = g * q + (1 - g) * p

            for j in range(nbits - 1):
                if q >= c[j]:
                    a[j] += 1
                    h[j] = 0
                elif h[j] < I:
                    a[j] += 1
                    h[j] += 1

        a = np.divide(sq, a, out=np.ones_like(a) * (10 ** (-10)), where=a != 0)
        a_dB = 10 * np.log10(a)

        delta = a_dB - c_dB - M
        idx = np.where(delta <= 0)[0]
        if len(idx) > 0:
            idx = idx[0]
            if idx > 0:
                asl, c_thr, j0 = self._interp(a_dB[idx - 1], a_dB[idx], c_dB[idx - 1], c_dB[idx], idx, M)
            else:
                asl = a_dB[idx]

        return asl

    def _interp(self, a0, a1, c0, c1, idx, M):
        ka = a1 - a0
        kc = c1 - c0
        ba = a1 - ka * idx
        bc = c1 - kc * idx

        j0 = (M + bc - ba) / (ka - kc)

        asl = ka * j0 + ba
        c_thr = kc * j0 + bc

        return asl, c_thr, j0


class SignalMixedProcessor:
    def __init__(self, clean_amp, picked_noise_amp, soll_snr):
        self.clean_amp = clean_amp
        self.picked_noise_amp = picked_noise_amp
        self.soll_snr = soll_snr
        self.signal_power = SignalPower()


    def process_snr_th_ab(self):
        asl_speech = self.signal_power.calculate_rms_level(self.clean_amp)
        asl_picked_noise_amp = self.signal_power.calculate_rms_level(self.picked_noise_amp)
        # current snr
        curr_snr = asl_speech - asl_picked_noise_amp
        # snr difference to the soll snr
        alpha_dB = curr_snr - self.soll_snr
        alpha = 10 ** (alpha_dB / 20)
        return self.clean_amp + alpha * self.picked_noise_amp

    def process_snr_gent(self):
        asl_speech = self.signal_power.calculate_active_speech_level(self.clean_amp, fs=16000)
        asl_picked_noise_amp = self.signal_power.calculate_rms_level(self.picked_noise_amp)
        # current snr
        curr_snr = asl_speech - asl_picked_noise_amp
        # snr difference to the soll snr
        alpha_dB = curr_snr - self.soll_snr

        if alpha_dB < 0:
            alpha = 10 ** (alpha_dB / 20)
            return self.clean_amp + alpha * self.picked_noise_amp

        else:
            alpha = 10 ** (-alpha_dB / 20)
            return alpha * self.clean_amp + self.picked_noise_amp

    def process_snr_ar_methode(self):
        # get the initial energy for reference
        signal_energy = np.mean(self.clean_amp ** 2)
        noise_energy = np.mean(self.picked_noise_amp ** 2)
        # calculates the gain to be applied to the noise to achieve the given SNR
        g = np.sqrt((signal_energy / noise_energy) * 10.0 ** (-self.soll_snr / 10))
        return self.clean_amp + g * self.picked_noise_amp


class DataAcquisition:
    def __init__(self, clean_speech_folder, noise_folder, SNrs, destination, fft,NoiseInit=True, Slv=False,
                 start=0,
                 order: int = 2,
                 rms: int = -36, sample_rate= 16000,
                 gen_mode: MixMethode = MixMethode.gent_mix_methode,
                 aug_mode: DataAugmentation = DataAugmentation.no_filter,
                 output_type: Mode = Mode.train_und_valid):
        self.signal_power = SignalPower()
        self.start = start
        self.noisy_des_slv = None
        self.noisy_des_wav = None
        self.clean_des_wav = None
        self.file_length = None
        self.clean_des_slv = None
        self.sample_rate = sample_rate
        self.fft = fft
        self.rms = rms
        self.Slv = Slv
        self.NoiseInit = NoiseInit
        self.output_type = output_type
        self.order = order
        self.clean_speech_folder = clean_speech_folder
        self.noise_folder = noise_folder
        self.gen_mode = gen_mode
        self.aug_mode = aug_mode
        self._create_destination(destination)
        self.SNrs = SNrs

    def acquire_data(self):
        print('> Loading wav data... ')
        clean_path_wav = list(self.clean_speech_folder.glob('**/*.wav'))
        noise_path_wav = list(self.noise_folder.glob('**/*.wav'))
        with concurrent.futures.ThreadPoolExecutor(1) as executor:
            for clean in clean_path_wav:
                future = executor.submit(func_timeout, 600, self._pick_noise_snr, (clean, noise_path_wav))
                future.add_done_callback(self.create_done_callback("_"))

    def create_done_callback(self, _):
        def handle_future_done(future: concurrent.futures.Future):
            try:
                future.result()
            except (FunctionTimedOut, Exception) as e:
                print(e)

        return handle_future_done

    def _pick_noise_snr(self, clean, noise_path_wav):
        if self.gen_mode == MixMethode.just_clean_data:
            in_data_amp, _ = librosa.load(clean, sr=self.sample_rate)
            self.file_length = len(in_data_amp)
            if self.Slv:
                for fft in self.fft:
                    destination_final = self.clean_des_slv.joinpath(str(fft))
                    destination_final.mkdir(parents=True, exist_ok=True)
                    name = destination_final.joinpath(clean.name[:-4] + ".slv")
                    complex_spec, abs_spec, phase = self._calculate_fft(in_data_amp,fft)
                    self.save_slv(complex_spec, abs_spec, phase, name, self.start, 'clean')


        elif self.gen_mode == MixMethode.gent_mix_methode or self.gen_mode == MixMethode.my_mix_methode or \
                self.gen_mode == MixMethode.th_mix_methode:
            for noise in noise_path_wav:
                clean_amp, _ = librosa.load(clean, sr=self.sample_rate)
                noise_amp, _ = librosa.load(noise, sr=self.sample_rate)
                self.file_length = len(clean_amp)


                for soll_snr in self.SNrs:
                    if self.NoiseInit:
                        self.start = random.randint(0, len(noise_amp) - len(clean_amp) - 1)
                    picked_noise_amp = noise_amp[self.start: self.start + len(clean_amp)]
                    mixed_amp_gained = self._get_mixed_amplitude(clean_amp, picked_noise_amp, soll_snr)

                    name = f"{clean.name[:-4]}_{noise.name[:-4]}_SNR_{soll_snr}"
                    destination_wav = self.noisy_des_wav.joinpath(name + '.wav')
                    self._save_enhanced_signal(destination_wav, mixed_amp_gained)
                    if self.Slv:
                        for fft in self.fft:
                            destination_final = self.noisy_des_slv.joinpath(str(fft))
                            destination_final.mkdir(parents=True, exist_ok=True)
                            complex_spec, abs_spec, phase = self._calculate_fft(mixed_amp_gained,fft)
                            destination_slv = destination_final.joinpath(name + '.slv')
                            self.save_slv(complex_spec, abs_spec, phase, destination_slv, self.start, 'noisy')

        else:
            return print('no valid mode is selected!!!')

    def _create_destination(self, destination):
        self.clean_des_slv = destination.joinpath('slv', 'clean')
        self.noisy_des_wav = destination.joinpath('wav', 'noisy')
        self.noisy_des_slv = destination.joinpath('slv', 'noisy')
        if not self.clean_des_slv.exists() \
                or not self.noisy_des_wav.exists() or not self.noisy_des_slv.exists():
            self.clean_des_slv.mkdir(parents=True, exist_ok=True)
            self.noisy_des_wav.mkdir(parents=True, exist_ok=True)
            self.noisy_des_slv.mkdir(parents=True, exist_ok=True)


    def _gain_data(self, in_data):
        rms = np.mean(in_data ** 2)
        alpha = np.sqrt(10 ** (self.rms / 10) / rms)
        return alpha * in_data
    def _iir_filter(self,fft):
        c = scipy.signal.butter(N=1, Wn=0.1)

        return scipy.signal.filtfilt(c[0], c[1], fft, axis=0, method='gust')

    def _enhance_asl(self, in_data,asl_soll):
        asl_ist = self.signal_power.calculate_active_speech_level(in_data,fs=self.sample_rate)
        beta = asl_ist - asl_soll
        if beta > 0:
            gamma = 10**(-beta/20)
        else:
            gamma = 10**(beta/20)
        return gamma * in_data

    def _get_filtered_data(self, clean_amp, noise_amp):
        if self.aug_mode == DataAugmentation.no_filter:
            return clean_amp, noise_amp
        elif self.aug_mode == DataAugmentation.filter_independently:
            c, d = self._random_coff(self.order)
            clean_amp = scipy.signal.filtfilt(c, d, clean_amp)
            c, d = self._random_coff(self.order)
            noise_amp = scipy.signal.filtfilt(c, d, noise_amp)
            return clean_amp, noise_amp
        elif self.aug_mode == DataAugmentation.filter_dependently:
            c, d = self._random_coff(self.order)
            clean_amp = scipy.signal.filtfilt(c, d, clean_amp)
            noise_amp = scipy.signal.filtfilt(c, d, noise_amp)
            return clean_amp, noise_amp

    def _get_mixed_amplitude(self, clean_amp, picked_noise_amp, soll_snr):


        signal_mixer = SignalMixedProcessor(clean_amp, picked_noise_amp, soll_snr)

        if self.gen_mode == self.gen_mode.gent_mix_methode:
            return signal_mixer.process_snr_gent()

        elif self.gen_mode == self.gen_mode.my_mix_methode:
            return signal_mixer.process_snr_ar_methode()

        elif self.gen_mode == self.gen_mode.th_mix_methode:
            return signal_mixer.process_snr_th_ab()


    def _avoid_clipping(self, mixed_amp):
        # Avoid clipping noise
        max_int16 = np.iinfo(np.int16).max
        min_int16 = np.iinfo(np.int16).min
        if mixed_amp.max(axis=0) > max_int16 or mixed_amp.min(axis=0) < min_int16:
            if mixed_amp.max(axis=0) >= abs(mixed_amp.min(axis=0)):
                reduction_rate = max_int16 / mixed_amp.max(axis=0)
            else:
                reduction_rate = min_int16 / mixed_amp.min(axis=0)
            mixed_amp = mixed_amp * reduction_rate
        return mixed_amp


    def _save_enhanced_signal(self, name, amp):
        soundfile.write(name, amp, self.sample_rate, 'PCM_16')

    def _random_coff(self, order):
        coefficients = []
        for i in range(self.order * 2):
            if i == 0 or i == order:
                coefficients.append(1)
            cof = random.uniform(-0.375, 0.375)
            coefficients.append(cof)
        return coefficients[:order], coefficients[order + 1:]

    def _calculate_fft(self, data,fft):
        data = librosa.util.fix_length(data, size=len(data) + fft // 2, mode='constant')
        complex_spec = librosa.stft(data, window='hann', n_fft=fft,
                                    hop_length=int(fft / 2),
                                    win_length=fft, center=True)
        mag, phase = librosa.magphase(
            librosa.stft(data, window='hann', n_fft=fft, hop_length=int(fft / 2),
                         win_length=fft, center=True))
        return complex_spec.T, mag.T, phase.T

    def save_slv(self, complex_spec, abs_spec, phase, name,start, x):
        g = shelve.open(str(name), protocol=pickle.HIGHEST_PROTOCOL)
        g['abs_spectrogram_' + x] = abs_spec
        g['complex_spectrogram_' + x] = complex_spec
        g['noise_index'] = start

        if self.output_type == Mode.test_noisy:
            g['phase_' + x] = phase
            g["file_length_" + x] = self.file_length
            g["frame_length_" + x] = abs_spec.shape[0]
            g["file_name_" + x] = name.name[:-4]+'.wav'
        g.close()



class BigData(Enum):
    big_dataset = 0,
    small_dataset = 1


class DataPrepration:
    def __init__(self, fft, data_set: BigData = BigData.big_dataset,
                 input_type: Mode = Mode.train_und_valid):

        self.FFT = fft

        self.input_type = input_type
        self.data_set = data_set
        self.file_names = []
        self.file_lengths = []

    def data_acquisition(self, data_path, destination):
        print('> Loading wav data... ')

        if self.input_type == Mode.test_noisy:
            noisy_test_path_wav = list(data_path.glob('**/*.wav'))
            self._save_the_input(input_path=noisy_test_path_wav, des_path=destination, x='noisy')
            return

        elif self.input_type == Mode.train_und_valid:

            clean_path = data_path.joinpath('clean')
            noisy_path = data_path.joinpath('noisy')
            clean_path_wav = list(clean_path.glob('**/*.wav'))
            noisy_path_wav = list(noisy_path.glob('**/*.wav'))
            if self.data_set == BigData.small_dataset:
                self._save_the_input(input_path=clean_path_wav, des_path=destination, x='clean')
                self._save_the_input(input_path=noisy_path_wav, des_path=destination, x='noisy')
            elif self.data_set == BigData.big_dataset:
                self._save_the_input(input_path=clean_path_wav, des_path=destination, x='clean')
                self._save_the_input(input_path=noisy_path_wav, des_path=destination, x='noisy')

        else:
            return print('no dataset is available!!!!!')

    def _read_data(self, pathlist):
        if self.data_set == BigData.small_dataset:
            self.file_names.clear()
            sum_of_input_signal = []
            for count, line in enumerate(pathlist):
                # read dada
                sig, _ = librosa.load(line, sr=16000)
                if self.input_type == Mode.test_noisy:
                    self.file_lengths.append(len(sig))
                    self.file_names.append(line.name)
                sum_of_input_signal.append(sig)
            sum_of_input_signal_concat = np.concatenate(sum_of_input_signal, axis=0)
            self.sig_length_fix = librosa.util.fix_length(sum_of_input_signal_concat,
                                                          size=len(
                                                              sum_of_input_signal_concat) + int(self.FFT/2))

        elif self.data_set == BigData.big_dataset:
            sig, _ = librosa.load(str(pathlist), sr=16000)

            self.sig_length_fix = sig
            self.file_lengths = len(sig)
            self.file_names = pathlist.name

    def _calculate_fft(self, path,fft):
        self._read_data(path)
        self.sig_length_fix = librosa.util.fix_length(self.sig_length_fix, size=len(self.sig_length_fix) + int(fft / 2))

        self.complex_spec = librosa.stft(self.sig_length_fix, window='hann', n_fft=fft,
                                         hop_length=int(fft/2),
                                         win_length=fft, center=True)
        self.mag, self.phase = librosa.magphase(
            librosa.stft(self.sig_length_fix, window='hann', n_fft=fft, hop_length=int(fft/2),
                         win_length=fft, center=True))

    def _save_the_input(self, input_path, des_path, x):
        if not des_path.exists():
            des_path.mkdir(parents=True, exist_ok=True)

        if self.data_set == BigData.small_dataset:

            for fft in self.FFT:
                final_path = des_path.joinpath('slv', x, str(fft))
                if not final_path.exists():
                    final_path.mkdir(parents=True, exist_ok=True)
                    self._calculate_fft(input_path,fft)
                g = shelve.open(str(final_path.joinpath("input_" + x + ".slv")),
                                protocol=pickle.HIGHEST_PROTOCOL)

                g['abs_spectrogram_' + x] = self.mag.T
                g['complex_spectrogram_' + x] = self.complex_spec.T

                if self.input_type == Mode.test_noisy:
                    g['phase_' + x] = self.phase.T
                    g["file_length_" + x] = self.file_lengths
                    g["file_name_" + x] = self.file_names
                g.close()

        if self.data_set == BigData.big_dataset:
            for i in range(len(input_path)):
                for fft in self.FFT:
                    self._calculate_fft(input_path[i],fft)
                    final_path = des_path.joinpath('slv',x,str(fft))
                    if not final_path.exists():
                        final_path.mkdir(parents=True, exist_ok=True)
                    g = shelve.open(str(final_path.joinpath(input_path[i].name[:-4] + ".slv")),
                                    protocol=pickle.HIGHEST_PROTOCOL)

                    g['abs_spectrogram_' + x] = self.mag.T
                    g['complex_spectrogram_' + x] = self.complex_spec.T

                    if self.input_type == Mode.test_noisy:
                        g['phase_' + x] = self.phase.T
                        g["file_length_" + x] = self.file_lengths
                        g["frame_length_" + x] = self.mag.shape[1]
                        g["file_name_" + x] = self.file_names
                    g.close()
