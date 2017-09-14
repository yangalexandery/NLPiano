import os
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt

import midi_io

num_epochs = 1
truncated_backprop_length = 15
state_size = 44
batch_size = 1
# num_batches = total_series_length // batch_size // truncated_backprop_length

input_size = 78
state_size = 100

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
        midi_io.print_statematrix(midi_list[i][:20])
        print('------------------------------------')

if __name__ == '__main__':
    learn_things()