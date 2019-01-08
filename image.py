#!/usr/bin/env python
#Object representation of an image - from the data provided in the files
class Image:

    def __init__(self, line_str):
        temp_arr = line_str.split()
        self.id = temp_arr[0] # Integer ID
        temp_arr = [int(i) for i in temp_arr[1:]] # Integer orientation
        self.orientation = temp_arr[0]
        self.features = tuple(temp_arr[1:]) # Tuple of features
        self.mini_features = self.features[0:100] # Taking less features 192
        self.pred_orientation = None # Predicted orientation
        self.output_vector = self.compute_output_vector(self.orientation) # For 90 degree orientation, output vector will be (0, 1, 0, 0)


    @staticmethod
    def compute_output_vector(orientation):
        out_vector_list = [0] * 4
        index = orientation / 90
        out_vector_list[index] += 1.0 
        return tuple(out_vector_list)

