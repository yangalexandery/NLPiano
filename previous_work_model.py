import os
import numpy as np
import tensorflow as tf
import random
# import matplotlib.pyplot as plt

import midi_io
import model_helpers

num_epochs = 100
batch_size = 5
sample_length = 128

notes_size = 78
attr_size = 79 

time_lstm_state_size = 100

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, sample_length, notes_size, attr_size])

time_lstm = tf.contrib.rnn.BasicLSTMCell(time_lstm_state_size)


for i in range(sample_length):
	time_slice = tf.reshape(batchX_placeholder[:,i,:,:], [batch_size * notes_size, attr_size])

	# take notes at a time slice and split them up
	# each statematrix[batch][time][note] corresponds to a note which we want to feed to the time neuron
	# time neuron takes in a vector of size attr_size
# time_input_placeholder = 