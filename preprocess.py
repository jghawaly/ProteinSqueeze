import cv2
import os
import numpy as np
import utils as u


training_data_directory = "D:/Data/all/train"
testing_data_directory = "D:/Data/all/test"
new_training_data_directory = "D:/Data/all/my_train"
training_results_path = "D:/Data/all/train.csv"

training_results = u.read_training_output_file(training_results_path)

display_raw = False
display_crops = False

threshold = 25

for filename in os.listdir(training_data_directory):
    if u.ColorFilters.green in filename:
        # grab the image of the nuclei
        nuclei_img = cv2.imread("/".join((training_data_directory, filename)), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        # the filename of the protein image
        protein_filename = "".join((filename.split("_")[0], "_", u.ColorFilters.green, ".", filename.split(".")[1]))
        # grab the image of the proteins
        protein_img = cv2.imread("/".join((training_data_directory, protein_filename))).astype(np.uint8)
        # check to make sure both images exist, OpenCV will return None if they don't
        if nuclei_img is None:
            raise FileNotFoundError
        elif protein_img is None:
            raise FileNotFoundError
        else:
            # if the user wants to display the raw unprocessed nuclei image
            if display_raw:
                print(u.int_labels_to_text(u.get_filename_labels(training_results, u.get_base_filename(protein_filename))))
                cv2.imshow("Img", nuclei_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            # threshold the nuclei from the images, we are thresholding the nuclei because they are usually more
            # visible than the proteins
            ret, thresh = cv2.threshold(nuclei_img.copy(), threshold, 255, cv2.THRESH_BINARY)
            # find the contours of the nuclei
            nuclei_img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # this will contain the new filtered contours
            filtered_contours = []
            # area of the image in square pixels
            nuclei_img_total_area = nuclei_img.shape[0] * nuclei_img.shape[1]
            # loop through every contour
            for c in contours:
                # find the area of this contour
                area = cv2.contourArea(c)
                # if the area fills greater than 0.01% of the image
                if area >= 0.0001 * nuclei_img_total_area:
                    # keep the contour
                    filtered_contours.append(c)
            # overwrite contours to save memory
            contours = filtered_contours
            # next step is to find contours that contain enough visual information to use
            c_s = []
            s = []
            # loop through every contour if there are some
            if len(contours) > 0:
                for c in contours:
                    # get image with only the region inside of the contour
                    drawn_img = u.fill_image_in_contour(protein_img, c)
                    # get the sum of all of the pixels values in the new image
                    sum = cv2.sumElems(drawn_img)
                    # append the current contour to a list
                    c_s.append(c)
                    # append the current sum to a list
                    s.append(sum)

                # convert the two list to numpy arrays
                c_s = np.array(c_s)
                s = np.array(s)[:,0]

                # calculated the mean of the sum list
                mean = np.mean(s)
                # get only contours that have pixel sum values greater than the mean of all of the sums
                contours = c_s[np.where(s >= mean)]
                i = 0
                for c in contours:
                    new_cell = u.fill_image_in_contour(protein_img, c)
                    #cropped_cell = u.blackout_and_crop(protein_img, c)
                    new_filename = "".join((new_training_data_directory, os.sep, u.get_base_filename(protein_filename), "_green", "_", str(i), ".png"))
                    cv2.imwrite(new_filename, new_cell)
                    if display_crops:
                        print(u.int_labels_to_text(
                            u.get_filename_labels(training_results, u.get_base_filename(protein_filename))))

                        cv2.imshow(protein_filename, new_cell)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    i += 1


