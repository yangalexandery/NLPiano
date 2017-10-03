import os
import numpy as np
import tensorflow as tf
import random
# import matplotlib.pyplot as plt

import midi_io
import model_helpers

num_epochs = 500
batch_size = 3
sample_length = 128

note_size = 78
attr_size = 79 

time_lstm_state_size = 384
note_lstm_state_size = 384

num_output_categories = 3


batchX_placeholder = tf.placeholder(tf.float32, [batch_size, sample_length, note_size, attr_size])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, sample_length, note_size])

time_lstm = tf.contrib.rnn.BasicLSTMCell(time_lstm_state_size)
time_state = tf.zeros([batch_size * note_size, time_lstm_state_size])

note_lstm = tf.contrib.rnn.BasicLSTMCell(note_lstm_state_size)
note_state = tf.zeros([batch_size * sample_length, note_lstm_state_size])

# regular fully-connected layer
W = tf.Variable(np.random.rand(note_lstm_state_size, num_output_categories), dtype=tf.float32) # weights for state t-1 -> state t
b = tf.Variable(np.zeros((1, num_output_categories)), dtype=tf.float32)

# time_layer_input = tf.unstack(batchX_placeholder, axis=1)
time_layer_input = tf.transpose(batchX_placeholder, [0, 2, 1, 3])
time_layer_input = tf.reshape(time_layer_input, [batch_size * note_size, sample_length, attr_size])
# print(time_layer_input.shape)
# time_layer_input = tf.transpose(batchX_placeholder, [0, 1, 2, 3])

time_layer_output, time_state = tf.nn.dynamic_rnn(cell=time_lstm, inputs=time_layer_input, dtype=tf.float32)

# time_layer_output shape is [batch_size * note_size, sample_length, time_lstm_state_size]
time_layer_output = tf.reshape(time_layer_output, [batch_size, note_size, sample_length, time_lstm_state_size])



# transpose layer: switch axes
# new shape should be [note_size, batch_size * sample_length, time_lstm_state_size]
transpose_layer_output = tf.transpose(time_layer_output, [0, 2, 1, 3])
transpose_layer_output = tf.reshape(transpose_layer_output, [batch_size * sample_length, note_size, time_lstm_state_size])

with tf.variable_scope('note_scope') as scope:
	note_layer_output, note_state = tf.nn.dynamic_rnn(cell=note_lstm, inputs=transpose_layer_output, dtype=tf.float32)

note_layer_output = tf.reshape(note_layer_output, [batch_size * sample_length * note_size, note_lstm_state_size])


# batchY_placeholder_transposed = tf.transpose(batchY_placeholder, [2, 0, 1])
output_series = tf.reshape(batchY_placeholder, [batch_size * sample_length * note_size])


# weights = tf.Variable([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 10.0]], tf.float32)
# logits_series = tf.matmul(tf.matmul(note_layer_output, W) + b, weights)
logits_series = tf.matmul(note_layer_output, W) + b
losses = []
predictions_series = tf.reshape(tf.nn.softmax(logits_series), [batch_size, sample_length, note_size, num_output_categories])


losses.append(tf.nn.sparse_softmax_cross_entropy_with_logits(
	logits=logits_series,
	labels=output_series
))
# for logits, labels in zip(logits_series, output_series):
#     losses.append(tf.nn.sparse_softmax_cross_entropy_with_logits(
#         logits=logits,
#         labels=labels
#     ))

total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)


def train_model():
    pieces = midi_io.get_pieces()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_list = []

        for epoch_idx in range(num_epochs):
            # _current_state = np.zeros((batch_size, state_size))

            x = []
            y = []
            for i in range(batch_size):
            	rand_sample = model_helpers.get_random_sample(pieces, sample_length)
            	x.append(model_helpers.transform_statematrix(rand_sample))
            	y.append(model_helpers.get_transformed_output(rand_sample))
                # x.append(model_helpers.get_random_sample(pieces, sample_length))
                # y.append(model_helpers.get_sample_output(x[-1]))
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

            _total_loss, _train_step, _predictions_series = sess.run(
                [total_loss, train_step, predictions_series], 
                feed_dict={
                    batchX_placeholder: x,
                    batchY_placeholder: y,
                    # init_state: _current_state
                })

            loss_list.append(_total_loss)
            print("LOSS FOR EPOCH #%d (mean cross entropy): %f" % (epoch_idx, _total_loss))
            if epoch_idx % 10 == 9:
                midi_io.print_predictions_2(_predictions_series)

    print("DONE")


train_model()