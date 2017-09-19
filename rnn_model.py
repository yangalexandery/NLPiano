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
num_substates = 2

def create_model():
    batchX_placeholder = tf.placeholder(tf.float32, [batch_size, sample_size, input_size, num_substates])
    batchY_placeholder = tf.placeholder(tf.int32, [batch_size, sample_size, input_size, num_substates])

    init_state = tf.placeholder(tf.float32, [batch_size, state_size])
    U = tf.Variable(np.random.rand(num_substates*input_size, state_size)) # weights for input -> state t
    W = tf.Variable(np.random.rand(state_size, state_size)) # weights for state t-1 -> state t
    b = tf.Variable(np.zeros((1, state_size)), dtype=tf.float32)


    V = tf.Variable(np.random.rand(state_size, input_size), dtype=tf.float32)
    b_V = tf.Variable(np.zeros(1, input_size), dtype=tf.float32)

    inputs_series = tf.unpack(batchX_placeholder, axis=1)
    outputs_series = tf.unpack(batchY_placeholder, axis=1)

    # forward pass
    current_state = init_state
    states_series = []
    for current_input in inputs_series: # there should be sample_size number of inputs, which is correct.
        # current_input = tf.reshape(current_input, [batch_size, 1]) # old code -- commented out for now because idk if this is correct
        current_input = tf.reshape(current_input, [batch_size, num_substates*input_size]) # let's just use this for now to see if it works

        next_state = tf.tanh(tf.matmul(current_input, U) + tf.matmul(current_state, W)+ b) # s_t = f(U x_t + W s_t-1)
        # we want each state to be a vector of shape [state_size, 1]
        # in this case [batch_size, state_size, 1]
        states_series.append(next_state)
        current_state = next_state


    # TODO: figure out if this stuff actually works
    logits_series = [tf.matmul(state, V) + b_V for state in states_series] #Broadcasted addition
    predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

    losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) for logits, labels in zip(logits_series,labels_series)]
    total_loss = tf.reduce_mean(losses)
    # TODO: see if matrix multiplications can be optimized by using tensorflow's a_is_sparse or b_is_sparse flag.
    train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

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

def get_sample_output(sample):
    # input shifted 1, plus a 0 state
    sample_output = sample[1:]
    sample_output.append([[0, 0] for i in range(input_size)]);
    return sample_output;

if __name__ == '__main__':
    # learn_things()
    pieces = midi_io.get_pieces()
    print("TEST " + str(len(pieces)))
    rand_sample = get_random_sample(pieces)
    midi_io.print_statematrix(rand_sample)
    print("DONE")
    midi_io.print_statematrix(get_sample_output(rand_sample))