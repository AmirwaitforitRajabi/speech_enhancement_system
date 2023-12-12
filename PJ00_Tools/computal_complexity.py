from enum import Enum
from cffi.backend_ctypes import xrange


class Mode(Enum):
    cruse = 1
    my_crd_opt = 5
    my_crd = 2
    my_crd_width = 3
    my_crd_width_2 = 6
    microsoft = 4
    my_crd_width_3 = 7


def cnn_complexity(input_cl, output_cl, k_size, output_dim):
    sum_encoder = 0
    for i in xrange(len(input_cl)):
        x1 = (k_size[0] * k_size[1]) * input_cl[i] * 1 * output_dim[
            i] * output_cl[i]
        if mode == Mode.my_crd and i == 0 or mode == Mode.my_crd_width and i == 0:
            # first layer hat kernel size(1,1)
            x1 = x1 / (k_size[0] * k_size[1])
        # print(x1)
        sum_encoder += x1
    return sum_encoder


def rnn_complexity(input_dim, neuron_num):
    sum_rnn = 0
    for i in xrange(len(input_dim)):
        x = 4 * (neuron_num[i] ** 2 + neuron_num[i] * input_dim[i])
        sum_rnn += x
    return sum_rnn


def dense_complexitiy(input_dime, neuron_num):
    sum_dense = 0
    for i in range(len(input_dime)):
        x = input_dime[i] * neuron_num[i]
        sum_dense += x
    return sum_dense


if __name__ == '__main__':

    mode = Mode.my_crd_opt

    kernel_size = [2, 3]
    stride = [1, 2]
    skip_kernel = [1, 1]

    if mode == Mode.cruse:
        input_channel = [1, 16, 32, 64, 128]
        output_channel = [16, 32, 64, 128, 256]
        output_dimension = [80, 39, 19, 9, 4]
        input_channel_t = [256, 128, 64, 32, 1]
        output_channel_t = [128, 64, 32, 16, 1]
        output_dimension_t = [9, 19, 39, 80, 161]
        n = [512, 512]
        m = [512, 512]
    if mode == Mode.my_crd:
        input_channel = [1, 1, 64, 64, 32]
        output_channel = [1, 64, 64, 32, 32]
        output_dimension = [161, 80, 39, 19, 9]
        n = [144, 144, 288, 288]
        m = [144, 144, 288, 288]
        f = [288, 288, 288]
        c = [288, 288, 161]
    if mode == Mode.my_crd_opt:
        input_channel = [1, 1, 32, 32, 32]
        output_channel = [1, 32, 32, 32, 32]
        output_dimension = [161, 80, 39, 19, 9]
        n = [144, 144, 144, 144]
        m = [144, 144, 144, 144]
        f = [288, 288, 288]
        c = [288, 288, 161]

    if mode == Mode.my_crd_width:
        input_channel = [1, 64, 64, 32, 32]
        output_channel = [1, 64, 64, 32, 32]
        output_dimension = [257, 128, 64, 31, 15]
        n = [480, 480]
        m = [480, 300]
        f = [480, 300, 300]
        c = [480, 300, 257]
    if mode == Mode.my_crd_width_2:
        input_channel = [1, 64, 32, 32, 32]
        output_channel = [1, 64, 32, 32, 32]
        output_dimension = [257, 128, 64, 31, 15]
        n = [120, 120, 120, 120]
        m = [120, 120, 120, 120]
        f = [480, 260, 480]
        c = [480, 480, 257]
    if mode == Mode.my_crd_width_3:
        input_channel = [1, 32, 32, 32, 32]
        output_channel = [1, 32, 32, 32, 32]
        output_dimension = [257, 128, 64, 31, 15]
        n = [240, 240, 240, 240]
        m = [240, 240, 240, 240]
        f = [480, 480, 300]
        c = [480, 480, 257]
    if mode == Mode.microsoft:
        input_channel = [1, 16, 32, 64]
        output_channel = [16, 32, 64, 128]
        output_dimension = [80, 39, 19, 9]
        input_channel_skip = [16, 32, 64, 128]
        output_channel_skip = [16, 32, 64, 128]
        output_dimension_skip = [80, 39, 19, 9]
        input_channel_t = [128, 64, 32, 16]
        output_channel_t = [64, 32, 16, 1]
        output_dimension_t = [19, 39, 79, 161]
        n = [288, 288, 288, 288]
        m = [288, 288, 288, 288]
    if mode == Mode.cruse or mode == Mode.microsoft:
        enc_mos = cnn_complexity(input_channel, output_channel, kernel_size, output_dimension, stride)
        dec_mos = cnn_complexity(input_channel_t, output_channel_t, kernel_size, output_dimension_t, stride)
        if mode == Mode.microsoft:
            rnn_mos = 0.75 * rnn_complexity(n, m)
            skip_connection = cnn_complexity(input_channel_skip,output_channel_skip,skip_kernel, output_dimension_skip)
            # print(skip_connection)
            print(enc_mos + dec_mos + skip_connection)
            print(rnn_mos)
            print('all:', mode, enc_mos + dec_mos + rnn_mos + skip_connection)
        else:
            rnn_mos = rnn_complexity(n, m)
            print(enc_mos + dec_mos)
            # print(dec_mos)
            print(rnn_mos)
            print('all:', mode, enc_mos + dec_mos + rnn_mos)

    else:
        dense = dense_complexitiy(f, c)
        enc_mos = cnn_complexity(input_channel, output_channel, kernel_size, output_dimension)
        rnn_mos = 0.75 * rnn_complexity(n, m)
        print('mode:', mode)
        print('enc:', enc_mos)
        print('dense:', dense)
        print('rnn:', rnn_mos)
        print('mode:', enc_mos + dense + rnn_mos)

