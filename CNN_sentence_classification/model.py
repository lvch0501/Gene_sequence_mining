import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot
from CNN_sentence_classification.config import Config_glove





class CNN_Sentence_classfication:

    def __init__(self, config):
        self.sequence_length = config.sequence_length
        self.num_class = config.num_class
        self.learning_rate = config.learning_rate
        self.word_embedding = config.vec_list
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.dropout_keep_prob = config.dropout_keep_prob
        self.filter_size = config.filter_size
        self.filter_num = config.filter_num
        self.filter_stride = config.filter_stride
        self.filter_total = config.filter_total
        self.l2_reg_lambda = config.l2_reg_lambda
        # placeholder for train_input
        self.input_x = tf.placeholder(tf.int32, [None, config.sequence_length], name="input_x")
        # placeholer for train_target
        self.input_y = tf.placeholder(tf.int32, [None, self.num_class], name="input_y")
        # define trainable embedding
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            self.en_embedding = tf.get_variable(name='en_embedding', shape=[self.vocab_size, self.embedding_size],
                                                initializer=tf.constant_initializer(self.word_embedding))
            # find the embeddings for each words in train_input
            self.input_embedding = tf.nn.embedding_lookup(self.en_embedding, self.input_x)
            # expand dims to satisfy the input of CNN_network
            self.input_embedding_expanded = tf.expand_dims(self.input_embedding, -1)

        # define CNN network for each filter size
        self.pool_output_list = []
        # execute convolution for each filter_size
        for filter_size in self.filter_size:
            self.filter_shape = [filter_size, self.embedding_size, 1, self.filter_num]
            # define filter shape
            self.W = tf.Variable(tf.truncated_normal(self.filter_shape, stddev=0.1), name="filter_parameter")
            # define bias
            self.b = tf.Variable(tf.constant(0.1, shape=[self.filter_num], name="filter_bias"))
            self.conv = tf.nn.conv2d(self.input_embedding_expanded, self.W, strides=[1, self.filter_stride, 1, 1],
                                     padding="VALID", name='conv')
            # non-linearity layer
            self.relu_output = tf.nn.relu(tf.nn.bias_add(self.conv, self.b))
            # max-pool
            self.maxpool_output = tf.nn.max_pool(self.relu_output, ksize=[1, (
                        self.sequence_length - filter_size) / self.filter_stride + 1, 1, 1],
                                                 strides=[1, 1, 1, 1], padding='VALID', name='max_pool')
            # add all output into a list
            self.pool_output_list.append(self.maxpool_output)

        # reshape all output into shape=[batch, filter_num*filter_size]
        self.final_output = tf.reshape(tf.concat(self.pool_output_list, name="final_output", axis=3),
                                       shape=[-1, self.filter_total])
        # add dropout layer
        self.dropout_output = tf.nn.dropout(self.final_output, keep_prob=self.dropout_keep_prob)
        # add a full-connection layer
        with tf.name_scope("output"):
            # define full-connection layer parameters
            self.full_W = tf.get_variable(name="full_W", shape=[self.filter_total, self.num_class],
                                          initializer=tf.contrib.layers.xavier_initializer())
            # define full-connection layer bias
            self.full_b = tf.Variable(tf.constant(0.1, shape=[self.num_class]), name="full_b")
            # calculate l2_loss for W and b
            self.l2_loss = tf.nn.l2_loss(self.full_W) + tf.nn.l2_loss(self.b)

            self.full_conn_output = tf.nn.xw_plus_b(self.dropout_output, self.full_W, self.full_b,
                                                    name="full_conn_output")
            # prediction for each input
            self.prediction = tf.argmax(self.full_conn_output, axis=1, name="prediction")

        with tf.name_scope("loss"):
            self.entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.full_conn_output,
                                                                        labels=self.input_y)
            self.loss = tf.reduce_mean(self.entropy_loss) + self.l2_loss*self.l2_reg_lambda

        with tf.name_scope("accuracy"):
            correct_num = tf.equal(tf.argmax(self.input_y, axis=1), self.prediction)
            self.accuracy = tf.reduce_mean(tf.cast(correct_num, "float"), name="accuracy")

















def train():
    config = Config_glove()
    CNN_model = CNN_Sentence_classfication(config)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    x_input = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    y_target = np.array([[1, 0], [0, 1]])
    embedding = sess.run(CNN_model.en_embedding)
    x = sess.run(CNN_model.final_output, feed_dict={CNN_model.input_x: x_input, CNN_model.input_y: y_target})
    y = sess.run(CNN_model.accuracy, feed_dict={CNN_model.input_x: x_input, CNN_model.input_y: y_target})

    print('111')


if __name__ == '__main__':
    train()