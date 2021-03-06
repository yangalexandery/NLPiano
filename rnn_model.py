import os
import numpy as np
import tensorflow as tf
import random
# import matplotlib.pyplot as plt

import midi_io
import model_helpers

num_epochs = 100
truncated_backprop_length = 15
state_size = 44
batch_size = 32
# num_batches = total_series_length // batch_size // truncated_backprop_length

input_size = 78
state_size = 384

sample_length = 128
num_substates = 2


batchX_placeholder = tf.placeholder(tf.float32, [batch_size, sample_length, input_size, num_substates])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, sample_length, input_size, num_substates])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])
U = tf.Variable(np.random.rand(num_substates*input_size, state_size), dtype=tf.float32) # weights for input -> state t
W = tf.Variable(np.random.rand(state_size, state_size), dtype=tf.float32) # weights for state t-1 -> state t
b = tf.Variable(np.zeros((1, state_size)), dtype=tf.float32)


V = tf.Variable(np.random.rand(state_size, num_substates*input_size*2), dtype=tf.float32)
b_V = tf.Variable(np.zeros((1, num_substates*input_size*2)), dtype=tf.float32)

inputs_series = tf.unstack(batchX_placeholder, axis=1)
outputs_series = tf.unstack(batchY_placeholder, axis=1)

# forward pass
current_state = init_state
states_series = []
for current_input in inputs_series: # there should be sample_length number of inputs, which is correct.
    # current_input = tf.reshape(current_input, [batch_size, 1]) # old code -- commented out for now because idk if this is correct
    current_input = tf.reshape(current_input, [batch_size, num_substates*input_size]) # let's just use this for now to see if it works

    next_state = tf.tanh(tf.matmul(current_input, U) + tf.matmul(current_state, W)+ b) # s_t = f(U x_t + W s_t-1)
    # we want each state to be a vector of shape [state_size, 1]
    # in this case [batch_size, state_size, 1]
    states_series.append(next_state)
    current_state = next_state


# TODO: figure out if this stuff actually works
logits_series = [tf.matmul(state, V) + b_V for state in states_series] #Broadcasted addition
# for logit in logits_series:
#     print(logit.dtype)
predictions_series = [tf.reshape(tf.nn.softmax(logits), [batch_size, num_substates*input_size, 2]) for logits in logits_series]

losses = []
for logits, labels in zip(logits_series, outputs_series):
    # print(logits_seriess.dtype, " ", labels.dtype)
    losses.append(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(logits, [batch_size, num_substates*input_size, 2]),
        labels=tf.reshape(labels, [batch_size, num_substates*input_size])
    ))
# losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(
#     logits=tf.reshape(logits, [batch_size, num_substates*input_size]), 
#     labels=tf.reshape(tf.nn.softmax(labels), [batch_size, num_substates*input_size])) for logits, labels in zip(logits_series,outputs_series)]
total_loss = tf.reduce_mean(losses)
# TODO: see if matrix multiplications can be optimized by using tensorflow's a_is_sparse or b_is_sparse flag.
train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

def train_model():
    pieces = midi_io.get_pieces()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_list = []

        for epoch_idx in range(num_epochs):
            _current_state = np.zeros((batch_size, state_size))

            x = []
            y = []
            for i in range(batch_size):
                x.append(model_helpers.get_random_sample(pieces, sample_length))
                y.append(model_helpers.get_sample_output(x[-1]))
            x = np.array(x)
            y = np.array(y)
            # x = np.array(get_random_sample(pieces))
            # y = np.array(get_sample_output(x))
            # x = np.expand_dims(x, axis=0)
            # y = np.expand_dims(y, axis=0)
            # x = np.expand_dims(np.array(get_random_sample(pieces)), axis=0)
            # y = np.expand_dims(np.array(get_sample_output(x)), axis=0)
            # np.expand_dims(y, axis=0)
            # print(x.shape)

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series], 
                feed_dict={
                    batchX_placeholder: x,
                    batchY_placeholder: y,
                    init_state: _current_state
                })

            loss_list.append(_total_loss)
            print("LOSS FOR EPOCH #%d (mean cross entropy): %f" % (epoch_idx, _total_loss))
            if epoch_idx % 10 == 9:
                midi_io.print_predictions(_predictions_series)

    print("DONE")


if __name__ == '__main__':
    print("TRAINING MODEL")
    train_model()
    # learn_things()
    # pieces = midi_io.get_pieces()
    # print("TEST " + str(len(pieces)))
    # rand_sample = get_random_sample(pieces)
    # midi_io.print_statematrix(rand_sample)
    # print("DONE")
    # midi_io.print_statematrix(get_sample_output(rand_sample))