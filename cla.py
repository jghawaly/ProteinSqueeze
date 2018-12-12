import tensorflow as tf
import numpy as np
from utils import TrainingParams, labels
from cifar10_loader import CIFAR10
from atlas_loader import AtlasLoader


class AtlasClassifier:
    def __init__(self, sess: tf.Session, tp: TrainingParams):
        self.sess = sess
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
        # self.images_resized = tf.image.resize_images(images=self.images, size=[224, 224])
        self.conv1 = tf.layers.conv2d(inputs=self.images, filters=96, kernel_size=7, strides=(2, 2),
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
        self.conv10 = tf.layers.conv2d(inputs=self.dropout9, filters=self.tp.num_classes, kernel_size=1, strides=(2, 2),
                                       activation=tf.nn.relu,
                                       name='conv10')
        self.avgpool10 = tf.layers.average_pooling2d(inputs=self.conv10, pool_size=15, strides=(2, 2), name='avgpool10')

        self.avgpool10 = tf.nn.sigmoid(self.avgpool10, name='out')

        self.logits = tf.layers.flatten(inputs=self.avgpool10, name='logits')

        # print out the network architecture
        print(self.conv1.shape)
        print(self.mpool1.shape)
        print(self.fire2.shape)
        print(self.fire3.shape)
        print(self.fire4.shape)
        print(self.mpool4.shape)
        print(self.fire5.shape)
        print(self.fire6.shape)
        print(self.fire7.shape)
        print(self.fire8.shape)
        print(self.mpool8.shape)
        print(self.fire9.shape)
        print(self.conv10.shape)
        print(self.avgpool10.shape)
        print(self.logits.shape)

    def train(self):
        # UNCOMMENT FOR SINGLE LABEL CLASSIFICAITON, SUCH AS CIFAR10
        # # contains the desired outputs
        # desired_outputs = tf.placeholder(dtype=tf.float32, shape=self.logits.shape, name='desired_outputs')
        # # the predicted labels
        # logits = tf.reshape(tf.cast(tf.argmax(self.logits, 1), tf.int32), [-1, 1])
        # # the actual labels
        # labels = tf.reshape(tf.cast(tf.argmax(desired_outputs, 1), tf.int32), [-1, 1])
        # # loss function for single label
        # loss_function = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=self.logits))
        # # Adam Optimizer
        # optimizer = tf.train.AdamOptimizer(self.tp.learning_rate).minimize(loss_function)
        # # accuracy measure
        # accuracy = tf.reduce_mean(tf.cast(tf.equal(logits, labels), dtype=tf.float32))

        # MULTI LABEL
        # contains the desired outputs
        desired_outputs = tf.placeholder(dtype=tf.float32, shape=self.logits.shape, name='desired_outputs')
        # loss function for multi label
        loss_function = tf.reduce_mean(tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=desired_outputs, logits=self.logits)))
        # Adam Optimizer
        optimizer = tf.train.AdamOptimizer(self.tp.learning_rate).minimize(loss_function)
        # accuracy measure
        # accuracy = tf.reduce_mean(tf.cast(tf.equal(self.logits, desired_outputs), dtype=tf.float32))
        # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.logits)), tf.round(desired_outputs), tf.float32))
        correction = tf.equal(tf.round(self.logits), tf.round(desired_outputs))
        accuracy = tf.reduce_mean(tf.cast(correction, tf.float32))
        # this will store losses during training
        losses = []

        # Initialize all variables
        self.sess.run(tf.global_variables_initializer())

        # Loop over number of epochs
        for epoch in range(self.tp.epochs):
            print('Running Epoch %g' % epoch)

            # shuffle batches
            self.tp.training_data.shuffle_and_batch(self.tp.batch_size)

            # batches per epoch for the training data
            bpe_training = np.floor(self.tp.training_data.length / self.tp.batch_size).astype(np.int16)

            x = 0
            while x < bpe_training:
                # Get a batch of images and labels
                batch_xs, batch_ys, batch_strings = self.tp.training_data.get_next_batch()
                batch_ys = np.squeeze(batch_ys)
                # And run the training op
                _, logits = self.sess.run([optimizer, self.logits],
                                          feed_dict={self.images: batch_xs, desired_outputs: batch_ys,
                                                     self.dropout_rate: self.tp.dropout_rate})
                x += 1

            self.tp.testing_data.shuffle_and_batch(self.tp.batch_size)

            # batches per epoch for the testing data
            bpe_testing = np.floor(self.tp.testing_data.length / self.tp.batch_size).astype(np.int16)
            x = 0
            loss = 0
            accs = []
            while x < bpe_testing:
                # Get a batch of images and labels
                batch_xs, batch_ys, batch_strings = self.tp.testing_data.get_next_batch()
                batch_ys = np.squeeze(batch_ys)
                # And run the training op
                loss, acc = self.sess.run([loss_function, accuracy],
                                          feed_dict={self.images: batch_xs, desired_outputs: batch_ys,
                                                     self.dropout_rate: self.tp.dropout_rate})

                x += 1
                accs.append(acc)
            print("Accuracy: %g" % (100.0 * np.average(np.array(accs))))
            losses.append(loss)
        return losses

    def save(self, path):
        # this is so that we can save the network
        saver = tf.train.Saver()
        # save the model
        return saver.save(self.sess, path)

    def save_new(self, path, saver):
        saver.save(self.sess, path)

    def load(self, path):
        # this is so that we can reload the network
        saver = tf.train.Saver()
        # Restore the model
        saver.restore(self.sess, path)
        print("Model loaded.")


if __name__ == "__main__":
    train_cifar10 = False

    if train_cifar10:
        with tf.Session() as sess:
            path = "C:/Users/james/PycharmProjects/machine_learning/src/deep_learning/vgg16_data/cifar-10-batches-py/"
            training_data = CIFAR10(['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'], path)
            testing_data = CIFAR10(['test_batch'], path)

            myTP = TrainingParams()
            myTP.training_data = training_data
            myTP.testing_data = testing_data
            myTP.epochs = 25
            myTP.batch_size = 100
            myTP.learning_rate = 0.0001
            myTP.dropout_rate = 0.1
            myTP.input_depth = 3
            myTP.input_width = 32
            myTP.input_height = 32
            myTP.num_classes = 10

            myModel = AtlasClassifier(sess, myTP)
            myModel.train()
    else:
        with tf.Session() as sess:
            # path = "D:/Data/all/train"
            path = "../data/train"
            # training_data = AtlasLoader(path, keys_file="D:/Data/all/train.csv")
            training_data = AtlasLoader(path, keys_file="../data/train.csv")
            testing_data = training_data

            myTP = TrainingParams()
            myTP.training_data = training_data
            myTP.testing_data = testing_data
            myTP.epochs = 25
            myTP.batch_size = 32
            myTP.learning_rate = 0.0001
            myTP.dropout_rate = 0.1
            myTP.input_depth = 1
            myTP.input_width = 512
            myTP.input_height = 512
            myTP.num_classes = len(labels)

            myModel = AtlasClassifier(sess, myTP)
            losses = myModel.train()
            myModel.save("myNetwork.ckpt")
            np.save("losses.npy", losses)
