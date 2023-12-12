import time
import warnings
warnings.filterwarnings('ignore')
import pathlib
from enum import Enum
import tensorflow as tf
import numpy as np

from PJ02_Training_Evaluations.ModelGenerator import ModelGenerator, NeuronalNetworkType
from PJ02_Training_Evaluations.PredictReconstruct import PredictReconstruct,ModelType, DataScale
from PJ02_Training_Evaluations.score_masked_base_all import score

np.random.seed(1337)  # for reproducibility
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_physical_devices('GPU')
        print(len(gpus), 'Physical GPUs', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        print(e)

class Mode(Enum):
    All = 1,
    train_valid = 2,
    predict = 3,
    score = 4,
startAll = time.time()
##################################################################################################################
# start with evaluating your test dataset with setting the mode on score and
# score_mode == 1 (mode 1 if u want to compare your data set with your predicted data)
# after evaluating the data
score_mode = 1
mode = Mode.All

##################################################################################################################
# select elect the model you want to train.
model_type = NeuronalNetworkType.original_cruse

##################################################################################################################
# now you can set up some parameters.
# mini_batch represents the number of WAV data to be read per batch in the generator.
batch_size = 8
look_backward = 6
mini_batch = 40
# jf using the Edinburgh set, set repeat_factor to 1.
# Otherwise, calculate repeat_factor as the product of selected_noise and snr_range.
repeat_factor = 1
n_frames = 200
n_fft = 320
nb_epochs = 1000
learning_rate = 5e-4

# now, you need to specify the path of the trained model.
result_path = pathlib.Path(f'F:/battle/cruse/{batch_size}_mse_{n_frames}_number_of_wav_data_{mini_batch}_lr{learning_rate}')

# specify the main path of your generated features, which are saved as .slv data
train_valid_test_path = pathlib.Path(f'F:/Feature/Edinburgh_{n_fft}')

# specify the path of the noisy audio dataset for evaluation.
ref_dataset = pathlib.Path(f"E:/Raw Data/Edinburgh/28_speakers/03_test")




if mode == Mode.train_valid or mode == Mode.All:
    #init

    kernel_size = (2, 3)
    strides = (1, 2)
    run_model = ModelGenerator(fft=n_fft, frames=n_frames, nn_type=model_type)
    # check if we have single input single output model or multiple input single output model
    if model_type == NeuronalNetworkType.original_cruse or model_type == NeuronalNetworkType.adjusted_cruse_1x2:
        XOXO = True
    else:
        XOXO = False
    model = run_model.create_model(learning_rate=learning_rate, kernel_size=kernel_size, strides=strides)
    run_model.read_data(input_data_path=train_valid_test_path, data_scale=DataScale.not_logarithmic, mini_batch=mini_batch,
                        repeat_factor=repeat_factor, siso=XOXO)
    run_model.model_train(model, result_path=result_path, nb_epochs=nb_epochs, batch_size=batch_size)
if mode == Mode.predict or mode == Mode.All:
    if model_type == NeuronalNetworkType.original_cruse or model_type == NeuronalNetworkType.adjusted_cruse_1x2:
        predict_reconstruct = PredictReconstruct(FFT=n_fft, frames=n_frames, minibatch=mini_batch,
                                                 batch_size=batch_size,
                                                 input_type=DataScale.not_logarithmic, nn_type=ModelType.end_to_end)
    else:
        predict_reconstruct = PredictReconstruct(FFT=n_fft, frames=n_frames, minibatch=mini_batch,
                                                 batch_size=batch_size,
                                                 input_type=DataScale.not_logarithmic, nn_type=ModelType.mask_based)
    predict_reconstruct.test_prediction(model_path=result_path, test_data_path=train_valid_test_path)

if mode == Mode.score or mode == Mode.All:
    if score_mode == 1:
        result_path = ref_dataset
    else:
        result_path = result_path
    score(result_path=result_path, ref_path=ref_dataset,
          repeat_faktor=repeat_factor, mode=score_mode, dns_mos=True)
