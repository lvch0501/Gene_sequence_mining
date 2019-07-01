import tensorflow as tf
import numpy as np
import os
import random
from matplotlib import pyplot
from CNN_sentence_classification.config import Config_glove
from CNN_sentence_classification.config import process_MR_dataset




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
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)



with open('../test_dataset/x.txt') as file:
    data_x = file.read()
    data_x = np.array(eval(data_x))
with open('../test_dataset/y.txt') as file:
    data_y = file.read()
    data_y = np.array(eval(data_y))
total_data_num = len(data_x)
array_np = np.arange(total_data_num)
random.shuffle(array_np)
data_x, data_y = data_x[array_np], data_y[array_np]
train_data_x, train_data_y = data_x[: int(0.8*total_data_num)], data_y[: int(0.8*total_data_num)]
test_data_x, test_data_y = data_x[int(0.8*total_data_num):], data_y[int(0.8*total_data_num):]

test_acc_list = []


def train():
    config = Config_glove()
    CNN_model = CNN_Sentence_classfication(config)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch_num in range(config.epoch):
        random_index = np.arange(len(train_data_x))
        random.shuffle(random_index)
        shuffle_data_x = train_data_x[random_index]
        shuffle_data_y = train_data_y[random_index]
        train_data_num = len(shuffle_data_x)
        for i in range(0, train_data_num, config.batch_size):
            train_x = shuffle_data_x[i:min(train_data_num, i + config.batch_size)]
            train_y = shuffle_data_y[i:min(train_data_num, i + config.batch_size)]
            loss, accuracy, _ = sess.run([CNN_model.loss,CNN_model.accuracy, CNN_model.train_op], feed_dict={CNN_model.input_x: train_x,
                                                                                        CNN_model.input_y: train_y})
            print("Epoch: "+str(epoch_num)+"---------------"+"loss: "+ str(loss)+"-----"+"accuracy: "+str(accuracy))
        temp_test_acc = []
        test_data_num = len(test_data_x)
        for j in range(0, test_data_num, config.batch_size):
            test_x  = test_data_x[j: min(j+config.batch_size, test_data_num)]
            test_y = test_data_y[j: min(j+config.batch_size, test_data_num)]
            test_acc = sess.run([CNN_model.accuracy], feed_dict={CNN_model.input_x: test_x,CNN_model.input_y: test_y})
            temp_test_acc.append(test_acc)
        test_acc_per_epoch = np.mean(temp_test_acc)
        print("loss of test :"+ str(test_acc_per_epoch))
        test_acc_list.append(test_acc_per_epoch)


if __name__ == '__main__':
    train()