import os
import numpy as np
import tensorflow as tf
import random
# import matplotlib.pyplot as plt

import midi_io

num_epochs = 1
truncated_backprop_length = 15
state_size = 44
batch_size = 1
# num_batches = total_series_length // batch_size // truncated_backprop_length

input_size = 78
state_size = 100

sample_length = 128

def create_model():
    batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
    batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

    init_state = tf.placeholder(tf.float32, [batch_size, state_size])
    W = tf.Variable(np.random.rand(state_size + input_size, state_size))
    b = tf.Variable(np.zeros((1, state_size)), dtype=tf.float32)

    W2 = tf.Variable(np.random.rand(state_size, input_size), dtype=tf.float32)
    b2 = tf.Variable(np.zeros(1, input_size), dtype=tf.float32)


def learn_things():
    midi_list = midi_io.get_pieces()
    for i in range(0, 20):
        print('-----------------NEW----------------')
        print(os.listdir('./data')[i])
        midi_io.print_statematrix(midi_list[i][-20:])
        print('------------------------------------')

def get_random_sample(pieces):
    rand_piece = pieces[random.randint(0, len(pieces))]

    while len(rand_piece) == 0:
        rand_piece = pieces[random.randint(0, len(pieces))]

    print("RANDOM SAMPLE LENGTH")
    print(len(rand_piece))

    if len(rand_piece) < sample_length:
        return rand_piece

    init_index = random.randint(0, len(rand_piece) - sample_length + 1)
    init_index -= init_index % 4
    return rand_piece[init_index:init_index+sample_length]

if __name__ == '__main__':
    # learn_things()
    pieces = midi_io.get_pieces()
    print("TEST " + str(len(pieces)))
    midi_io.print_statematrix(get_random_sample(pieces))
    print("DONE")