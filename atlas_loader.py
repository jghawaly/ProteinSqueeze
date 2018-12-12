import numpy as np
import os
import cv2
import utils


class AtlasLoader:
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
            if self.color_to_load in filename:
                img = cv2.imread("/".join((self.files_dir, filename)), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                base_fname = utils.get_base_filename(filename)
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
        # next_img_batch = self.image_batch[self.batch_iteration]
        # next_label_batch = self.label_batch[self.batch_iteration]
        # next_string_batch = self.string_batch[self.batch_iteration]
        next_img_batch = np.reshape(self.images[self.shuffled_indices[self.batch_iteration]], [self.batch_size, 512, 512, 1])
        next_label_batch = self.label_indices[self.shuffled_indices[self.batch_iteration]]
        next_string_batch = self.label_strings[self.shuffled_indices[self.batch_iteration]]
        self.batch_iteration += 1

        return next_img_batch, next_label_batch, next_string_batch

