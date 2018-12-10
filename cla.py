import tensorflow as tf
import numpy as np
from utils import TrainingParams
import utils
from tensorflow.examples.tutorials.mnist import input_data
from cifar10_loader import CIFAR10


class AtlasClassifier:
    def __init__(self, tp: TrainingParams):
        self.tp = tp
        self.initial_weight_std = 0.1
        self.incep1_stride = 2
        self.incep2_stride = 2
        self.incep3_stride = 2
        self.num_neurons_fc1 = 255
        self.num_neurons_fc2 = 255
        self.num_neurons_fc3 = 10  # len(utils.labels)
        self.pre_incep1_filters = 32
        self.pre_incep2_filters = 32
        self.pre_incep3_filters = 32
        self.pre_incep4_filters = 32
        self.build_model()

    def fire_module(self, inputs, c1_filters, c2_filters, c3_filters, name):
        c1 = tf.layers.conv2d(inputs=inputs,
                              filters=c1_filters,
                              kernel_size=1,
                              strides=(1, 1),
                              activation=tf.nn.relu)
        c2 = tf.layers.conv2d(inputs=c1,
                              filters=c2_filters,
                              kernel_size=1,
                              strides=(1, 1),
                              activation=tf.nn.relu)
        c3 = tf.layers.conv2d(inputs=c1,
                              filters=c3_filters,
                              kernel_size=3,
                              strides=(1, 1),
                              activation=tf.nn.relu,
                              padding='same')
        return tf.concat([c2, c3], 3, name=name)

    def build_model(self):
        self.dropout_rate = tf.placeholder(dtype=tf.float32)
        self.images = tf.placeholder(tf.float32, shape=[self.tp.batch_size, self.tp.input_width, self.tp.input_height,
                                                        self.tp.input_depth])
        self.images_resized = tf.image.resize_images(images=self.images, size=[224, 224])
        self.conv1 = tf.layers.conv2d(inputs=self.images_resized, filters=96, kernel_size=7, strides=(2, 2),
                                      activation=tf.nn.relu,
                                      name="conv1")
        self.mpool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=3, strides=(2, 2), name='mpool1')
        self.fire2 = self.fire_module(self.mpool1, 16, 64, 64, "fire2")
        self.fire3 = self.fire_module(self.fire2, 16, 64, 64, "fire3")
        self.fire4 = self.fire_module(self.fire3, 32, 128, 128, "fire4")
        self.mpool4 = tf.layers.max_pooling2d(inputs=self.fire4, pool_size=3, strides=(2, 2), name="mpool4")
        self.fire5 = self.fire_module(self.mpool4, 32, 128, 128, "fire5")
        self.fire6 = self.fire_module(self.fire5, 48, 192, 192, "fire6")
        self.fire7 = self.fire_module(self.fire6, 48, 192, 192, "fire7")
        self.fire8 = self.fire_module(self.fire7, 64, 256, 256, "fire8")
        self.mpool8 = tf.layers.max_pooling2d(inputs=self.fire8, pool_size=3, strides=(2, 2), name='mpool8')
        self.fire9 = self.fire_module(self.mpool8, 32, 128, 128, "fire9")
        self.dropout9 = tf.layers.dropout(inputs=self.fire9, rate=self.tp.dropout_rate, name='dropout9')
        self.conv10 = tf.layers.conv2d(inputs=self.dropout9, filters=self.tp.num_classes, kernel_size=1, strides=(1, 1),
                                       activation=tf.nn.relu,
                                       name='conv10')
        self.avgpool10 = tf.layers.average_pooling2d(inputs=self.conv10, pool_size=12, strides=(1, 1), name='avgpool10')
        self.out = tf.nn.sigmoid(self.avgpool10, name='out')

    def train(self):
        desired_outputs = tf.placeholder(dtype=tf.float32, shape=self.out.shape, name='desired_outputs')
        # define loss
        loss_function = tf.reduce_mean(tf.square(self.out - desired_outputs) + tf.losses.get_regularization_loss())
        # define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.tp.learning_rate).minimize(loss_function)
        correct_pred = tf.equal(tf.argmax(self.out, 3), tf.argmax(desired_outputs, 3))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # for tracking the loss throughout the training process
        losses = []

        # train
        with tf.Session() as sess:
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Loop over number of epochs
            for epoch in range(self.tp.epochs):
                print('Running Epoch %g' % epoch)

                # RESET OUR BATCHES
                self.tp.training_data.shuffle_and_batch(self.tp.batch_size)
                self.tp.testing_data.shuffle_and_batch(self.tp.batch_size)

                # batches per epoch for the training data
                bpe_training = np.floor(self.tp.training_data.length / self.tp.batch_size).astype(np.int16)

                x = 1
                while x < bpe_training:
                    # Get a batch of images and labels
                    batch_xs, batch_ys, batch_strings = self.tp.training_data.get_next_batch()
                    batch_ys = np.resize(batch_ys, [batch_ys.shape[0], 1, 1, batch_ys.shape[1]])
                    # And run the training op
                    sess.run([optimizer], feed_dict={self.images: batch_xs, desired_outputs: batch_ys,
                                                     self.dropout_rate: self.tp.dropout_rate})

                    x += 1

                # batches per epoch for the testing data
                bpe_testing = np.floor(self.tp.testing_data.length / self.tp.batch_size).astype(np.int16)
                x = 1
                loss = 0
                accs = []
                while x < bpe_testing:
                    # Get a batch of images and labels
                    batch_xs, batch_ys, batch_strings = self.tp.testing_data.get_next_batch()
                    batch_ys = np.resize(batch_ys, [batch_ys.shape[0], 1, 1, batch_ys.shape[1]])

                    # And run the training op
                    loss, acc = sess.run([loss_function, accuracy],
                                            feed_dict={self.images: batch_xs, desired_outputs: batch_ys,
                                                       self.dropout_rate: self.tp.dropout_rate})

                    x += 1
                    accs.append(acc)
                print("Accuracy: %g"%(100 * np.average(np.array(accs))))
                losses.append(loss)

            # self.training_data.shuffle_and_batch(self.batch_size)
            # batch_xs, batch_ys, batch_strings = self.training_data.get_next_batch()
            # out = sess.run(self.output, feed_dict={self.corrupted_input: batch_xs})
            # plt.imshow(batch_xs[0])
            # plt.show()
            # plt.imshow(out[0])
            # plt.show()
            #
            # plt.plot(losses)
            # plt.xlabel("Epochs")
            # plt.ylabel("Loss")
            # plt.title("Loss Convergence")
            # plt.show()


if __name__ == "__main__":
    training_data = CIFAR10(['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'],
                            '/Users/james/PycharmProjects/machine_learning/src/deep_learning/vgg16_data/cifar-10-batches-py/')

    testing_data = CIFAR10(['test_batch'],
                           '/Users/james/PycharmProjects/machine_learning/src/deep_learning/vgg16_data/cifar-10-batches-py/')
    myTP = TrainingParams()
    myTP.training_data = training_data  # input_data.read_data_sets("MNIST_data/", one_hot=True)
    myTP.testing_data = testing_data
    myTP.epochs = 25
    myTP.batch_size = 20
    myTP.learning_rate = 1E-4
    myTP.dropout_rate = 0.01
    myTP.input_depth = 3
    myTP.input_width = 32#512
    myTP.input_height = 32#512
    myTP.num_classes = 10

    myModel = AtlasClassifier(myTP)
    myModel.train()
