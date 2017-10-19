import os
import numpy as np
import tensorflow as tf
import random
# import matplotlib.pyplot as plt
# tensorflow now supports GPU computation on dev instance!!!

import midi_io
import model_helpers

num_epochs = 5000
batch_size = 5
sample_length = 128

note_size = 78
attr_size = 79 

time_lstm_state_size = 300
note_lstm_state_size = 100
note_lstm_state_size_2 = 50

num_output_categories = 2


batchX_placeholder = tf.placeholder(tf.float32, [batch_size, sample_length, note_size, attr_size])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, sample_length, note_size, num_output_categories])

time_lstm = tf.contrib.rnn.BasicLSTMCell(time_lstm_state_size)
time_state = tf.zeros([batch_size * note_size, time_lstm_state_size])

time_lstm_2 = tf.contrib.rnn.BasicLSTMCell(time_lstm_state_size)
time_state_2 = tf.zeros([batch_size * note_size, time_lstm_state_size])

note_lstm = tf.contrib.rnn.BasicLSTMCell(note_lstm_state_size)
note_state = tf.zeros([batch_size * sample_length, note_lstm_state_size])

note_lstm_2 = tf.contrib.rnn.BasicLSTMCell(note_lstm_state_size_2)
note_state_2 = tf.zeros([batch_size * sample_length, note_lstm_state_size_2])


# regular fully-connected layer
W = tf.Variable(np.random.rand(note_lstm_state_size_2, num_output_categories), dtype=tf.float32) # weights for state t-1 -> state t
b = tf.Variable(np.zeros((1, num_output_categories)), dtype=tf.float32)

time_layer_1_input = tf.transpose(batchX_placeholder, [0, 2, 1, 3])
time_layer_1_input = tf.reshape(time_layer_1_input, [batch_size * note_size, sample_length, attr_size])

time_layer_1_output, time_state = tf.nn.dynamic_rnn(cell=time_lstm, inputs=time_layer_1_input, dtype=tf.float32)

# time_layer_output shape is [batch_size * note_size, sample_length, time_lstm_state_size]

with tf.variable_scope('time_scope_2') as scope:
    time_layer_2_output, time_state_2 = tf.nn.dynamic_rnn(cell=time_lstm_2, inputs = time_layer_1_output, dtype=tf.float32)

time_layer_2_output = tf.reshape(time_layer_2_output, [batch_size, note_size, sample_length, time_lstm_state_size])


# transpose layer: switch axes
# new shape should be [note_size, batch_size * sample_length, time_lstm_state_size]
transpose_layer_output = tf.transpose(time_layer_2_output, [0, 2, 1, 3])
transpose_layer_output = tf.reshape(transpose_layer_output, [batch_size * sample_length, note_size, time_lstm_state_size])

with tf.variable_scope('note_scope') as scope:
    note_layer_1_output, note_state = tf.nn.dynamic_rnn(cell=note_lstm, inputs=transpose_layer_output, dtype=tf.float32)


with tf.variable_scope('note_scope_2') as scope:
    note_layer_2_output, note_state_2 = tf.nn.dynamic_rnn(cell=note_lstm_2, inputs=note_layer_1_output, dtype=tf.float32)


note_layer_2_output = tf.reshape(note_layer_2_output, [batch_size * sample_length * note_size, note_lstm_state_size_2])



# batchY_placeholder_transposed = tf.transpose(batchY_placeholder, [2, 0, 1])
output_series = tf.cast(tf.reshape(batchY_placeholder, [batch_size * sample_length * note_size * num_output_categories]), tf.float32)
output_series_0, output_series_1 = tf.split(output_series, 2, 0)
output_series_0 = tf.reshape(output_series_0, [batch_size * sample_length * note_size, 1])
mask = tf.concat([tf.ones(output_series_0.shape, dtype=tf.float32), output_series_0], axis=1)

# weights = tf.Variable([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 10.0]], tf.float32)
# logits_series = tf.matmul(tf.matmul(note_layer_output, W) + b, weights)
logits_series = tf.matmul(note_layer_2_output, W) + b
sigmoid_output = tf.sigmoid(tf.reshape(logits_series, [batch_size * sample_length * note_size * num_output_categories]))
sigmoid_output_0, sigmoid_output_1 = tf.split(sigmoid_output, 2, 0)

predictions_series = tf.reshape(sigmoid_output, [batch_size, sample_length, note_size, num_output_categories])
losses = -tf.log(2 * tf.multiply(sigmoid_output, output_series) - sigmoid_output - output_series + 1.0 + 1e-7)
losses = tf.reshape(losses, [batch_size * sample_length * note_size, num_output_categories])
losses = tf.multiply(mask, losses)
#losses = -tf.log(2 * tf.multiply(sigmoid_output_0, output_series_0) - sigmoid_output_0 - output_series_0 + tf.fill([batch_size * sample_length * note_size], 1.0))
# losses = []
# predictions_series = tf.reshape(tf.nn.softmax(logits_series), [batch_size, sample_length, note_size, num_output_categories])


total_loss = tf.reduce_sum(losses)
train_step = tf.train.AdadeltaOptimizer(learning_rate=0.01, rho=0.95, epsilon=0.000001).minimize(total_loss)


def train_model():
    pieces = midi_io.get_pieces()
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))    
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        loss_list = []

        for epoch_idx in range(num_epochs):
            # _current_state = np.zeros((batch_size, state_size))

            x = []
            y = []
            # print(pieces[0][0].shape)
            for i in range(batch_size):
                rand_sample = model_helpers.get_random_sample(pieces, sample_length)
                x.append(model_helpers.transform_statematrix(rand_sample))
                # y.append(model_helpers.get_transformed_output(rand_sample))
                # x.append(model_helpers.get_random_sample(pieces, sample_length))
                y.append(model_helpers.get_sample_output(rand_sample))
            x = np.array(x)
            y = np.array(y)
            # x = np.array(get_random_sample(pieces))
            # y = np.array(get_sample_output(x))

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
                midi_io.print_predictions(_predictions_series)

    print("DONE")


train_model()
