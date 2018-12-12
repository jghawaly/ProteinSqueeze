import numpy as np
import os
import cv2
import utils


class OldAtlasLoader:
    def __init__(self, files_dir, keys_file="", one_hot_labels=True, color_to_load=utils.ColorFilters.green):
        self.files_dir = files_dir
        self.keys_file = keys_file
        self.color_to_load = color_to_load

        self.images = None
        self.label_indices = None
        self.label_strings = None

        # self.image_batch = None
        # self.label_batch = None
        # self.string_batch = None
        self.shuffled_indices = None

        self.length = 0
        self.batch_iteration = 0
        self.one_hot_labels = one_hot_labels

        self.parse()

    def parse(self):
        print("Loading Data...")
        label_data = utils.read_training_output_file(self.keys_file)
        filenames = os.listdir(self.files_dir)

        my_images = []
        my_label_indices = []
        my_label_strings = []
        for filename in filenames:
            base_fname = utils.get_base_filename(filename)
            b = base_fname + "_blue.png"
            g = base_fname + "_green.png"
            r = base_fname + "_red.png"
            y = base_fname + "_yellow.png"
            bimg = utils.open_image("/".join((self.files_dir, b)))
            gimg = utils.open_image("/".join((self.files_dir, g)))
            rimg = utils.open_image("/".join((self.files_dir, r)))
            yimg = utils.open_image("/".join((self.files_dir, y)))

            img = np.dstack((bimg, gimg, rimg, yimg))

            if self.one_hot_labels:
                lbls = self.one_hot(utils.get_filename_labels(label_data, base_fname), len(utils.labels))
            else:
                lbls = utils.get_filename_labels(label_data, base_fname)
            strs = utils.int_labels_to_text(lbls)
            my_images.append(img)
            my_label_indices.append(lbls)
            my_label_strings.append(strs)

        self.images = np.array(my_images)
        self.label_indices = np.array(my_label_indices, dtype=np.int32)
        self.label_strings = np.array(my_label_strings)
        self.length = len(my_images)
        print("Done Loading Data...")

    def shuffle_and_batch(self, batch_size):
        self.batch_size = batch_size
        indices = np.arange(0, self.length, 1)
        p = np.random.permutation(self.length)
        shuffled_indices = indices[p]

        self.shuffled_indices = np.array(np.split(shuffled_indices, self.length // batch_size, axis=0))
        self.batch_iteration = 0

    def one_hot(self, inputs, num_classes):
        outputs = np.zeros((num_classes), dtype=np.int32)
        outputs[inputs] = 1
        return outputs

    def get_next_batch(self):
        next_img_batch = np.reshape(self.images[self.shuffled_indices[self.batch_iteration]], [self.batch_size, 512, 512, 1])
        next_label_batch = self.label_indices[self.shuffled_indices[self.batch_iteration]]
        next_string_batch = self.label_strings[self.shuffled_indices[self.batch_iteration]]
        self.batch_iteration += 1

        return next_img_batch, next_label_batch, next_string_batch

class AtlasLoader:
    def __init__(self, files_dir, keys_file="", one_hot_labels=True, color_to_load=utils.ColorFilters.green):
        self.files_dir = files_dir
        self.keys_file = keys_file
        self.color_to_load = color_to_load

        self.image_filenames = None
        self.label_indices = None
        self.label_strings = None

        self.shuffled_indices = None

        self.length = 0
        self.batch_iteration = 0
        self.one_hot_labels = one_hot_labels

        self.parse()

    def build_image(self, path):
        b = path + "_blue.png"
        g = path + "_green.png"
        r = path + "_red.png"
        y = path + "_yellow.png"
        bimg = utils.open_image("/".join((self.files_dir, b)))
        gimg = utils.open_image("/".join((self.files_dir, g)))
        rimg = utils.open_image("/".join((self.files_dir, r)))
        yimg = utils.open_image("/".join((self.files_dir, y)))

        return np.dstack((bimg, gimg, rimg, yimg))

    def parse(self):
        print("Loading Data...")
        label_data = utils.read_training_output_file(self.keys_file)
        filenames = os.listdir(self.files_dir)

        my_label_indices = []
        my_label_strings = []
        my_image_filenames = []
        for filename in filenames:
            base_fname = utils.get_base_filename(filename)

            if self.one_hot_labels:
                lbls = self.one_hot(utils.get_filename_labels(label_data, base_fname), len(utils.labels))
            else:
                lbls = utils.get_filename_labels(label_data, base_fname)

            strs = utils.int_labels_to_text(lbls)
            my_image_filenames.append("/".join((self.files_dir, base_fname)))
            my_label_indices.append(lbls)
            my_label_strings.append(strs)

        self.image_filenames = np.array(my_image_filenames)
        self.label_indices = np.array(my_label_indices, dtype=np.int32)
        self.label_strings = np.array(my_label_strings)
        self.length = len(my_image_filenames)
        print("Done Loading Data...")

    def shuffle_and_batch(self, batch_size):
        self.batch_size = batch_size
        indices = np.arange(0, self.length, 1)
        p = np.random.permutation(self.length)
        shuffled_indices = indices[p]

        self.shuffled_indices = np.array(np.split(shuffled_indices, self.length // batch_size, axis=0))
        self.batch_iteration = 0

    def one_hot(self, inputs, num_classes):
        outputs = np.zeros((num_classes), dtype=np.int32)
        outputs[inputs] = 1
        return outputs

    def image_batch_from_files(self, filenames):
        images = []
        for base_fname_with_path in filenames:
            images.append(self.build_image(base_fname_with_path))

        return np.array(images)

    def get_next_batch(self):
        # next_img_batch = np.reshape(self.images[self.shuffled_indices[self.batch_iteration]], [self.batch_size, 512, 512, 1])
        next_img_batch = np.reshape(self.image_batch_from_files(self.image_filenames[self.shuffled_indices[self.batch_iteration]]), [self.batch_size, 512, 512, 1])
        next_label_batch = self.label_indices[self.shuffled_indices[self.batch_iteration]]
        next_string_batch = self.label_strings[self.shuffled_indices[self.batch_iteration]]
        self.batch_iteration += 1

        return next_img_batch, next_label_batch, next_string_batch

