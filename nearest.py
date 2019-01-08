
#!/usr/bin/env python
# Program that performs knn training and testing

"""
Euclidean distance between the 2 images
"""

from collections import defaultdict
from sortdata import Sorting
import numpy as np

K = 11 


def train(image_list):
    return image_list #return list of training images as model


def test(image_list, model_images):
    def calculate_distance(image1, image2):
        dist = 0
        for f_image1, f_image2 in zip(image1.features, image2.features):
        	dist += (f_image1 - f_image2) ** 2 #Calculates the Euclidean distance between 2 images
        return 0.0 + dist

    # testing data-----
    for test_image in image_list:
        # Traversing all the model images and calculating the distance
        #For some value of K
        least_dist = Sorting(K)
        for model_image in model_images:
            curr_dist = calculate_distance(test_image, model_image)
            least_dist.insert(curr_dist, model_image)

        # Get a voting from all the K nearest neighbors
        model_dict = defaultdict(lambda: 0)
        for index in range(K):
            curr_img = least_dist.get(index)
            model_dict[curr_img.orientation] += 1

        max_orientation = max(model_dict, key=model_dict.get)
        test_image.pred_orientation = max_orientation
