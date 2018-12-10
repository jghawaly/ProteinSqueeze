import numpy as np
import skimage.util as u
import matplotlib.pyplot as plt

class CorruptionModels:
    blackout = 0
    gaussian = 1

class CIFAR10:
    def __init__(self, filenames, base_dir):
        self.filenames = filenames
        self.base_dir = base_dir
        self.images = None
        self.label_indices = None
        self.label_strings = None
        self.length = 0
        self.parse()

        self.image_batch = None
        self.label_batch = None
        self.string_batch = None
        self.batch_iteration = 0

    def shuffle_and_batch(self, batch_size):
        """
        random.shuffle(training_data)
        batches = [
            training_data[k:k + batch_size]
            for k in range(0, self.length, batch_size)]
        """
        # shuffle our data
        p = np.random.permutation(self.length)
        shuffled_images = self.images[p]
        shuffled_labels = self.label_indices[p]
        shuffled_strings = self.label_strings[p]

        self.image_batch = np.array(np.split(shuffled_images, self.length // batch_size, axis=0))
        self.label_batch = np.array(np.split(shuffled_labels, self.length // batch_size, axis=0))
        self.string_batch = np.array(np.split(shuffled_strings, self.length // batch_size, axis=0))
        self.batch_iteration = 0

    def get_next_batch(self):
        next_img_batch = self.image_batch[self.batch_iteration]
        next_label_batch = self.label_batch[self.batch_iteration]
        next_string_batch = self.string_batch[self.batch_iteration]
        self.batch_iteration += 1
        return next_img_batch, next_label_batch, next_string_batch

    def get_next_batch_and_corrupt(self, tp):
        i_batch, l_batch, s_batch = self.get_next_batch()
        c_batch = i_batch.copy()
        if tp.corruption_model == CorruptionModels.blackout:
            blackout_rate = tp.blackout_corruption_fraction
            # we want True to replace and False to keep
            mask = np.random.choice(a=[0, 1], p=[1.0-blackout_rate, blackout_rate], size=c_batch.shape).astype(np.bool)
            zero_batch = np.zeros(c_batch.shape).astype(type(c_batch))
            c_batch[mask] = zero_batch[mask]
        if tp.corruption_model == CorruptionModels.gaussian:
            c_batch = u.random_noise(c_batch, mode='gaussian', clip=True, mean=tp.gauss_mean, var=tp.gauss_var)

        return i_batch, l_batch, s_batch, c_batch

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def format_image(self, img, as_float=True):
        r = np.reshape(img[0:32 * 32], (32, 32))
        g = np.reshape(img[32 * 32:32 * 32 * 2], (32, 32))
        b = np.reshape(img[32 * 32 * 2:32 * 32 * 3], (32, 32))

        rgb = np.dstack((r, g, b))
        if as_float:
            return rgb.astype(np.float32)/255.0
        else:
            return rgb.astype(np.uint8)

    def parse(self):
        images = []
        label_indices = []
        label_strings = []
        for filename in self.filenames:
            # Load images and labels
            data = self.unpickle(self.base_dir + filename)
            igs = data[b'data']
            labels = self.unpickle(self.base_dir + "batches.meta")[b'label_names']
            for i in range(len(igs)):
                # Load a single image and label
                img = self.format_image(igs[i])
                lbl_index = data[b'labels'][i]
                lbl_str = labels[lbl_index]

                images.append(img)
                lbl = np.zeros((10))
                lbl[lbl_index] = 1.0
                label_indices.append(lbl)
                label_strings.append(lbl_str)
                self.length += 1

        self.images = np.array(images)
        self.label_indices = np.array(label_indices)
        self.label_strings = np.array(label_strings)
