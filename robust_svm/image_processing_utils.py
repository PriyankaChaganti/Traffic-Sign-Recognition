import cv2
from data_set import AM
from os.path import join
import numpy as np
from robust_svm.process_images import read_feature_file
from settings import images_path, project_path, temp_folder,training_data_folder,hog3_path

def get_cropped_image(dataset_path, annotation):
    """
    Crops the images based on annotations
    :param dataset_path:The path to the directory that holds Images.(Example:../data/training_data/Images)
    :param annotation:Annotations of the given image.
    :return:cropped_image
    """
    # Get the x,y co-ordinates of region of interest
    ROIY1 = int(annotation[AM.ROIy1])
    ROIX1 = int(annotation[AM.ROIx1])
    ROIY2 = int(annotation[AM.ROIy2])
    ROIX2 = int(annotation[AM.ROIx2])
    # Get the image filename
    image_filename = annotation[AM.Filename]
    image_path = join(dataset_path, image_filename)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Copy the image into a new variable
    cropped_image = img[ROIY1:ROIY2, ROIX1:ROIX2].copy()
    return cropped_image


def highlight_invariant_threshold(image):
    """
    Processes image by
    1.Converting the cropped image from rgb to hsv color space using cv2
    2.Applying thresholding.
    3.Applying bit wise operators on the thresholded images to get final image.
    :param image:RGB image that has to be preprocessed
    :return:output_image
    """
    # Convert the image from BGR color space to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)

    # Threshold for blue color. 235-255 represents blue in hue(H of HSV space)
    blue_low = np.array([235, 0, 0], dtype=np.uint8)
    blue_high = np.array([255, 255, 255], dtype=np.uint8)
    image_in_blue = cv2.inRange(hsv_image, blue_low, blue_high)

    # Threshold for red color. 0-10 represents red in hue(H of HSV space)
    red_low = np.array([0, 0, 0])
    red_high = np.array([10, 255, 255])
    image_in_red = cv2.inRange(hsv_image, red_low, red_high)

    # Threshold image for a minimum saturation(S in HSV space)
    saturation_low = np.array([0, 40, 0], dtype=np.uint8)
    saturation_high = np.array([255, 255, 255], dtype=np.uint8)
    image_with_saturated_colors = cv2.inRange(hsv_image, saturation_low, saturation_high)

    # Threshold image for a minimum value (V in HSV space)
    value_low = np.array([0, 0, 30], dtype=np.uint8)
    value_high = np.array([255, 255, 230], dtype=np.uint8)
    image_with_minimum_value = cv2.inRange(hsv_image, value_low, value_high)

    # Copy the image
    output_image = image_in_blue.copy()
    cv2.bitwise_or(image_in_blue, image_in_red, output_image)

    image_with_saturated_red_n_blue = None
    cv2.bitwise_and(output_image, image_with_saturated_colors, output_image)

    # Image with with minimum saturation of red and blue and value
    cv2.bitwise_and(output_image, image_with_minimum_value, output_image)
    return output_image


def hole_fill(image):
    """
    Fills holes by morphological closing and opening. Morphological closing reduces the
    radius of white regions by the kernel size. Any white region that has radius less
    than the kernel radius disappear removing small noisy white chunks. Morphological
    opening increases the size of white regions by the kernel size. Any white region
    that did not disappear during closing will return to the original size during opening
    :param image:The image on which morphological operations are to be done.
    :return:image_opened
    """
    # Create a 2x2 cell with every value set to 1
    kernel = np.ones((2, 2), np.uint8)
    # Morphological closing
    image_closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Morphological opening
    image_opened = cv2.morphologyEx(image_closed, cv2.MORPH_OPEN, kernel, iterations=2)
    return image_opened


def grow_region(image):
    """
    Takes a thresholded image as input and generates a binary image
    that has a single contiguous region
    :param image:Thresholded image.
    :return:binary_image
    """
    # Get image width, height, and center
    image = image.copy()
    width = np.size(image, 1)
    height = np.size(image, 0)
    xc = int(width/2)
    yc = int(height/2)

    # Get seeds in the image
    seeds = get_seeds(xc, yc, width, height, image)

   # Initialize a numpy array with image size
    binary_image = np.zeros((height, width, 1), np.uint8)

    for (x, y) in seeds:
        children = []

        while True:
            neighbors = get_neighboring_points(x, y, width, height)

            for i, j in neighbors:
                if image[j, i] > 0:
                    children.append((i, j))
                    image[j, i] = 0
            if len(children) > 0:
                (x, y) = children[-1]
                binary_image[y][x] = 255
                del(children[-1])
            else:
                break

    return binary_image


def get_seeds(xc, yc, x_lim, y_lim, image):
    """
    Starts at the image center and searches along the row and column to find
    a pixel that has non zero values i.e white pixel. This function is used to get a list of
    seeds for region growing.

    Note: Top left corner of the image represents the origin (0,0)
    :param xc:
    :param yc:
    :param x_lim:
    :param y_lim:
    :param image:
    :return:
    """
    seeds = []
    # Traverse along the x-axis from the center to the right to find a pixel with non-zero value
    for i in range(xc, x_lim):
        if image[yc][i] > 0:
            seeds.append((i, yc))
            break

    # Traverse along the x-axis from the center to the left to find a pixel with non-zero value
    for i in range(xc, 0, -1):
        if image[yc][i] > 0:
            seeds.append((i, yc))
            break

    # Traverse along the y-axis from the center to the bottom to find a pixel with non-zero value
    for i in range(yc, y_lim):
        if image[i][xc] > 0:
            seeds.append((xc, i))
            break

    # Traverse along the y-axis from the center to the top to find a pixel with non-zero value
    for i in range(yc, 0, -1):
        if image[i][xc] > 0:
            seeds.append((xc, i))
            break

    # Increment x,y with of the following tuples to get the diagonal pixels
    # Ex: Pt(0, 0) + Pt(1, 1) = Pt(1, 1) i.e top right neighboring pixel of Pt(0, 0)
    # Pt(2, 3) +  Pt(-1, 0) = Pt(1, 3) i.e bottom left neighboring pixel of Pt(2, 3)
    diagonal_directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for xd, yd in diagonal_directions:
        # Pt(xp, yp) represents current pixel in each iteration of the while loop
        xp = xc
        yp = yc
        # Traverse within the boundaries of the image
        while 0 < xp < x_lim and 0 < yp < y_lim:
            # Add the pixel to the list of seeds
            if image[yp][xp] > 0:
                seeds.append((xp, yp))
            # Traverse diagonally
            xp = xp + xd
            yp = yp + yd

    return seeds


def get_neighboring_points(x, y, x_lim, y_lim):
    """
    Returns the eight immediate neighbors of the pixel (x,y).
    :param x:
    :param y:
    :param x_lim:
    :param y_lim:
    :return:
    """
    neighbors = []

    steps = [-1, 0, 1]
    for i in steps:
        for j in steps:
            if not (i == 0 and j == 0):
                if 0 < x+i < x_lim and 0 < y+j < y_lim:
                    neighbors.append((x+i, y+j))

    return neighbors


if __name__ == "__main__":
    #Test the function read_image_annotations
    features_folder_path = join(training_data_folder,hog3_path)
    data_class_id='00000'
    image_file_name ='00000_00000.ppm'
    hog_feature_data= read_feature_file(features_folder_path, data_class_id, image_file_name)
    print(hog_feature_data)

    sample_annotation = ['00001_00029.ppm', '193', '191', '16', '17', '177', '4174', '14']
    data_set = '00014'
    dataset_path = join(training_data_folder,images_path, data_set)
    image_filename = sample_annotation[0]
    image_path = join(dataset_path, image_filename)

    cropped_image = get_cropped_image(dataset_path, sample_annotation)
    thresholded_image = highlight_invariant_threshold(cropped_image)
    filled_image = hole_fill(thresholded_image)
    region_grown_image = grow_region(filled_image)

    # Save the original image in temp_space directory
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cv2.imwrite(join(temp_folder, image_filename.replace('.ppm', '.png')), original_image)

    # Save the cropped image in temp_space directory
    cropped_image_name = image_filename.replace('.ppm', '_cropped.png')
    cv2.imwrite(join(temp_folder, cropped_image_name), cropped_image)

    # Save the thresholded image in temp_space directory
    thresholded_image_name = image_filename.replace('.ppm', '_thresholded.png')
    cv2.imwrite(join(temp_folder, thresholded_image_name), thresholded_image)

    # Save the filled image in temp_space directory
    filled_image_name = image_filename.replace('.ppm', '_hole_filled.png')
    cv2.imwrite(join(temp_folder, filled_image_name), filled_image)

    # Save the region grown image in temp_space directory
    region_grown_image_name = image_filename.replace('.ppm', '_region_grown.png')
    cv2.imwrite(join(temp_folder, region_grown_image_name), region_grown_image)
