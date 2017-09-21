import numpy as np
import tensorflow as tf
import random

import midi_io


def introspection():
    midi_list = midi_io.get_pieces()
    for i in range(0, 20):
        print('-----------------NEW----------------')
        print(os.listdir('./data')[i])
        midi_io.print_statematrix(midi_list[i][-20:])
        print('------------------------------------')


def get_random_sample(pieces, sample_length):
    rand_piece = pieces[random.randint(0, len(pieces) - 1)]

    while len(rand_piece) == 0:
        rand_piece = pieces[random.randint(0, len(pieces) - 1)]

    # print("RANDOM SAMPLE LENGTH")
    # print(len(rand_piece))

    if len(rand_piece) < sample_length:
        return rand_piece

    init_index = random.randint(0, len(rand_piece) - sample_length)
    init_index -= init_index % 4
    return rand_piece[init_index:init_index+sample_length]


def get_sample_output(sample):
    # input shifted 1, plus a 0 state
    sample_output = sample[1:]
    input_size = len(sample[0])
    return np.vstack((sample_output, np.array([[[0.0, 0.0] for i in range(input_size)]])));
    # return sample_output;
