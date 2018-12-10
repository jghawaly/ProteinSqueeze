import numpy as np
import cv2

class TrainingParams:
    training_data = None
    testing_data = None
    epochs = None
    batch_size = None
    learning_rate = None
    dropout_rate = None
    input_depth = None
    input_width = None
    input_height = None
    l2_beta = None
    num_classes = None


class ColorFilters:
    green = "green"
    blue = "blue"
    red = "red"
    yellow = "yellow"


labels = {0: "Nucleoplasm",
          1: "Nuclear membrane",
          2: "Nucleoli",
          3: "Nucleoli fibrillar center",
          4: "Nuclear speckles",
          5: "Nuclear bodies",
          6: "Endoplasmic reticulum",
          7: "Golgi apparatus",
          8: "Peroxisomes",
          9: "Endosomes",
          10: "Lysosomes",
          11: "Intermediate filaments",
          12: "Actin filaments",
          13: "Focal adhesion sites",
          14: "Microtubules",
          15: "Microtubule ends",
          16: "Cytokinetic bridge",
          17: "Mitotic spindle",
          18: "Microtubule organizing center",
          19: "Centrosome",
          20: "Lipid droplets",
          21: "Plasma membrane",
          22: "Cell junctions",
          23: "Mitochondria",
          24: "Aggresome",
          25: "Cytosol",
          26: "Cytoplasmic bodies",
          27: "Rods & rings"}


def read_training_output_file(path):
    """
    Read the true training outputs into memory
    :param path: full path to CSV file containing the data
    :return: numpy array containing string representations of the data
    """
    return np.genfromtxt(path, delimiter=",", dtype=str, skip_header=1)


def get_filename_labels(data, filename):
    """
    Returns an array of integer values containing the labels of the corresponding filename
    :param data: the numpy array containing the data
    :param filename: the filename (not path) of the image in question
    :return: array of integer labels
    """
    # find the index of the filename in the data
    index = np.where(data[:, 0]==filename)

    # if there are more than one of the same filename, then there may be an issue
    if len(index) == 1:
        return [int(s) for s in data[index][0][1].split()]
    else:
        # probably need a more descriptive exception
        raise Exception


def get_base_filename(f):
    """
    Get the base filename from the full filename, i.e. x123_green.png would be x123
    :param f: full filename
    :return: base filename
    """
    return f.split("_")[0]


def int_labels_to_text(l):
    """
    Get the text labels of the integer labels
    :param l: array of integer labels
    :return: array of text labels
    """
    return [labels[a] for a in l]

def fill_image_in_contour(img, contour):
    """
    Create an image that only contains pixels within the contour regions
    :param img: the original image
    :param contour: the contour
    :return: new image
    """
    # create an empty image the same size as the original
    m = np.zeros(img.shape, dtype=img.dtype)
    # fill the empty image with white pixels INSIDE of the contour
    contour_fill = cv2.fillPoly(m, [contour], [255, 255, 255])
    # blackout all pixels in the original image OUTSIDE of the filled contour
    filled_img = cv2.bitwise_and(img.copy(), contour_fill)
    return filled_img

def blackout_and_crop(img, contour):
    """
    Crop out a region from an image within a contour, with all pixels outside of the contour blacked out
    :param img: original image
    :param contour: contour of interest
    :return: the cropped image
    """
    blackout_img = fill_image_in_contour(img, contour)
    # we need to get the dimensions of a bounding rectangle around the contour
    x, y, w, h = cv2.boundingRect(contour)
    # crop the image using the dimensions we just found
    return blackout_img[y:y + h, x:x + w]


def format_and_resize(img, size):
    rows = img.shape[0]
    cols = img.shape[1]


if __name__ == '__main__':
    training_results_path = "D:/Data/all/train.csv"
    data = read_training_output_file(training_results_path)
    my_labels = get_filename_labels(data, '00070df0-bbc3-11e8-b2bc-ac1f6b6435d0')
    print(my_labels)
    print(int_labels_to_text(my_labels))
