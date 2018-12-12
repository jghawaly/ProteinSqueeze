import tensorflow as tf
import numpy as np
from utils import TrainingParams, labels, open_image
from cifar10_loader import CIFAR10
from atlas_loader import AtlasLoader
import os


class AtlasClassifier:
    def __init__(self, sess: tf.Session, tp: TrainingParams, input_data_type='atlas', training=True):
        self.is_training=training
        self.input_data_type = input_data_type
        self.sess = sess
        self.tp = tp
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
        cat = tf.concat([c2, c3], 3, name=name)

        return tf.layers.batch_normalization(inputs=cat, training=self.is_training)

    def build_model(self):
        if self.input_data_type == 'atlas':
            st = 2
            ps = 15
        elif self.input_data_type == 'cifar10':
            st = 1
            ps = 10
        else:
            raise ValueError("Bad input type, must be atlas or cifar10")

        self.dropout_rate = tf.placeholder(dtype=tf.float32)
        self.images = tf.placeholder(tf.float32, shape=[self.tp.batch_size, self.tp.input_width, self.tp.input_height,
                                                        self.tp.input_depth])

        if self.input_data_type == 'atlas':
            self.inputs = tf.cast(self.images/255, tf.float32)
        else:
            self.inputs = self.images

        conv1 = tf.layers.conv2d(inputs=self.inputs, filters=96, kernel_size=7, strides=(st, st),
                                      activation=tf.nn.relu,
                                      name="conv1")

        mpool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=3, strides=(st, st), name='mpool1')
        fire2 = self.fire_module(mpool1, 16, 64, 64, "fire2")
        fire3 = self.fire_module(fire2, 16, 64, 64, "fire3")
        fire4 = self.fire_module(fire3, 32, 128, 128, "fire4")
        mpool4 = tf.layers.max_pooling2d(inputs=fire4, pool_size=3, strides=(st, st), name="mpool4")
        fire5 = self.fire_module(mpool4, 32, 128, 128, "fire5")
        fire6 = self.fire_module(fire5, 48, 192, 192, "fire6")
        fire7 = self.fire_module(fire6, 48, 192, 192, "fire7")
        fire8 = self.fire_module(fire7, 64, 256, 256, "fire8")
        mpool8 = tf.layers.max_pooling2d(inputs=fire8, pool_size=3, strides=(st, st), name='mpool8')
        fire9 = self.fire_module(mpool8, 32, 128, 128, "fire9")
        dropout9 = tf.layers.dropout(inputs=fire9, rate=self.tp.dropout_rate, name='dropout9')
        conv10 = tf.layers.conv2d(inputs=dropout9, filters=self.tp.num_classes, kernel_size=1, strides=(2, 2),
                                       activation=tf.nn.relu,
                                       name='conv10')
        avgpool10 = tf.layers.average_pooling2d(inputs=conv10, pool_size=ps, strides=(1, 1), name='avgpool10')

        if self.input_data_type == 'atlas':
            activated = tf.nn.sigmoid(avgpool10, name='outs')
        elif self.input_data_type == 'cifar10':
            activated = tf.nn.softmax(avgpool10, name='outs')
        else:
            raise ValueError("Bad input type, must be atlas or cifar10")

        self.logits = tf.layers.flatten(inputs=activated, name='logits')

        # print out the network architecture
        print(conv1.shape)
        print(mpool1.shape)
        print(fire2.shape)
        print(fire3.shape)
        print(fire4.shape)
        print(mpool4.shape)
        print(fire5.shape)
        print(fire6.shape)
        print(fire7.shape)
        print(fire8.shape)
        print(mpool8.shape)
        print(fire9.shape)
        print(conv10.shape)
        print(avgpool10.shape)
        print(activated.shape)
        print(self.logits.shape)

    def train(self):
        lr = tf.placeholder(dtype=tf.float32, name='learning_rate')
        if self.input_data_type == 'cifar10':
            # contains the desired outputs
            desired_outputs = tf.placeholder(dtype=tf.float32, shape=self.logits.shape, name='desired_outputs')
            # the predicted labels
            logits = tf.reshape(tf.cast(tf.argmax(self.logits, 1), tf.int32), [-1, 1])
            # the actual labels
            labels = tf.reshape(tf.cast(tf.argmax(desired_outputs, 1), tf.int32), [-1, 1])
            # loss function for single label
            loss_function = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=self.logits))
            # Adam Optimizer
            optimizer = tf.train.AdamOptimizer(lr)
            # accuracy measure
            accuracy = tf.reduce_mean(tf.cast(tf.equal(logits, labels), dtype=tf.float32))
        elif self.input_data_type == 'atlas':
            # contains the desired outputs
            desired_outputs = tf.placeholder(dtype=tf.float32, shape=self.logits.shape, name='desired_outputs')
            # loss function for multi label
            loss_function = tf.reduce_mean(tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=desired_outputs, logits=self.logits)))
            # Adam Optimizer
            optimizer = tf.train.AdamOptimizer(lr)
            # how correct network is
            correctness = tf.equal(tf.round(self.logits), tf.round(desired_outputs))
            # accuracy measure
            accuracy = tf.reduce_mean(tf.cast(correctness, tf.float32))
        else:
            raise ValueError("Bad input type, must be atlas or cifar10")

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss_function)

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
                # calculate annealed learning rate
                annealed_lr = self.tp.learning_rate / np.sqrt(float(epoch)) if epoch != 0 else self.tp.learning_rate
                # And run the training op
                _, logits = self.sess.run([train_op, self.logits],
                                          feed_dict={self.images: batch_xs, desired_outputs: batch_ys,
                                                     self.dropout_rate: self.tp.dropout_rate,
                                                     lr: annealed_lr})
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

    def evaluate(self, inputs):
        if self.input_data_type == 'atlas':
            return self.sess.run([self.logits], feed_dict={self.images: inputs, self.dropout_rate: 0.0})
        elif self.input_data_type == 'cifar10':
            print("Not Yet Implemented")
            return None
        else:
            raise ValueError("Bad input type, must be atlas or cifar10")

    def evaluate_directory(self, path, threshold, string_filter=""):
        results = {}
        for filename in os.listdir(path):
            if string_filter in filename:
                img = open_image("/".join((path, filename)))
                inference = self.evaluate([img])
                results[filename] = np.sort(inference[np.where(inference >= threshold)])

        return results

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
            training_data = CIFAR10(['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'],
                                    path)
            testing_data = CIFAR10(['test_batch'], path)

            myTP = TrainingParams()
            myTP.training_data = training_data
            myTP.testing_data = testing_data
            myTP.epochs = 25
            myTP.batch_size = 100
            myTP.learning_rate = 0.001
            myTP.dropout_rate = 0.05
            myTP.input_depth = 3
            myTP.input_width = 32
            myTP.input_height = 32
            myTP.num_classes = 10

            myModel = AtlasClassifier(sess, myTP, input_data_type='cifar10')
            losses = myModel.train()
            np.save("losses_adapt.npy", losses)
    else:
        with tf.Session() as sess:
            path = "../data/train"
            training_data = AtlasLoader(path, keys_file="../data/train.csv")
            testing_data = training_data

            myTP = TrainingParams()
            myTP.training_data = training_data
            myTP.testing_data = testing_data
            myTP.epochs = 25
            myTP.batch_size = 32
            myTP.learning_rate = 0.001
            myTP.dropout_rate = 0.05
            myTP.input_depth = 1
            myTP.input_width = 512
            myTP.input_height = 512
            myTP.num_classes = len(labels)

            myModel = AtlasClassifier(sess, myTP, input_data_type='atlas')
            losses = myModel.train()
            myModel.save("/networks/myCifarModel_adapt.ckpt")
            np.save("losses_adapt.npy", losses)
