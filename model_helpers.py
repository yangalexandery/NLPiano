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
    # init_index -= init_index % 4
    return rand_piece[init_index:init_index+sample_length]


def get_sample_output(sample):
    # input shifted 1, plus a 0 state
    sample_output = sample[1:]
    input_size = len(sample[0])
    return np.vstack((sample_output, np.array([[[0.0, 0.0] for i in range(input_size)]])));
    # return sample_output;

def transform_statematrix(statematrix):
    # given statematrix, return feature matrix
    new_feature_matrix = []
    feature_vector_length = 79

    for i in range(len(statematrix)):
        state = statematrix[i]
        new_time_slice = []
        for j in range(len(state)):
            new_feature_state = [0]*feature_vector_length
            # position (1)
            new_feature_state[0] = j #/ feature_vector_length

            # pitch class (12)
            new_feature_state[j % 12 + 1] += 1

            # previous vicinity (50)
            # this occupies the space [13, 63)
            for k in range(-12, 13):
                if -1 < j + k < len(state):
                    prev_note = state[j + k]
                    context_index = (k + 12) * 2 + 13
                    new_feature_state[context_index] += prev_note[0]
                    new_feature_state[context_index + 1] += prev_note[1]

            # previous context (12)
            tally = [0] * 12
            for k in range(len(state)):
                tally[k % 12] += state[k][0]
            for k in range(12):
                new_feature_state[k + 63] += tally[k] # maybe normalize this?

            # beat
            new_feature_state[75 + i % 4] ++ 1
            new_time_slice.append(new_feature_state)
        new_feature_matrix.append(new_time_slice)
    return new_feature_matrix # convert to numpy array?
    # shape = [# timeslices, # notes, # attributes]
    # this method is probably bugged. TODO: fix.

def get_transformed_output(sample):
    output = get_sample_output(sample)
    # given output, transform into usable 3-output matrix
    new_output = []
    for output_slice in output:
        new_output_slice = []
        for note in output_slice:
            val = 0
            if note[0] == 1 and note[1] == 0:
                val = 1
            if note[0] == 1 and note[1] == 1:
                val = 2
            new_output_slice.append(val)
        new_output.append(new_output_slice)
    return new_output