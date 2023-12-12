import time
import pathlib
from PJ01_Data_Generator.generate_data_set import DataAcquisition, MixMethode, DataAugmentation, Mode, DataPrepration, BigData


class DataGeneration:
    your_own_data_set = 1
    ready_dataset = 2


reference_path = pathlib.Path('E:/Data/raw_data/Edinburgh Dataset/28_speakers/03_test')
destination = pathlib.Path('F:/Feature/Edinburgh_320/test')
# change the pad of noise if u want to generate the unique test data set
noise = pathlib.Path('C:/Users/audiolab/Desktop/small dataset/feat/selected-noise/test')

# for the test data-set set it to test_noisy
output_type_dataset = Mode.test_noisy
######################################
n_fft = 320
snrs = [-5, 0, 5, 10, 15, 20]
######################################
# select you mode of dataa generation
launch_mode = DataGeneration.ready_dataset
######################################
# use just_clean_data for generating the clean .slv data
mix_methode = MixMethode.my_mix_methode
######################################

if launch_mode == DataGeneration.your_own_data_set:
    start = time.time()
    X = DataAcquisition(fft=[n_fft], clean_speech_folder=reference_path, noise_folder=noise, SNrs=snrs,
                        destination=destination, gen_mode=mix_methode,
                        NoiseInit=True, output_type=output_type_dataset, Slv=True)
    X.acquire_data()
    ende = time.time()
    del X

    print("> Saving Completed, Time : ", ende - start)
    print('> +++++++++++++++++++++++++++++++++++++++++++++++++++++ ')

if launch_mode == DataGeneration.ready_dataset:
    x = DataPrepration(fft=[n_fft], data_set=BigData.big_dataset, input_type=output_type_dataset)
    y = x.data_acquisition(data_path=reference_path, destination=destination)
