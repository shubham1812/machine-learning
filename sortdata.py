#!/usr/bin/env python
# Data structure for storing sorted list (in ascending order) of a fixed size


class Sorting:
    def __init__(self, max_length):
        self.max_length = max_length  # The maximum size of the list
        self.list = list()  # List of tuples where each tuple is of the form (key, val)


    def insert(self, key, value):
        if len(self.list) == 0 or key < self.list[len(self.list) - 1]:
            index = 0
            for item in self.list:
                if item[0] >= key:
                    break
                index += 1
            self.list.insert(index, (key, value)) #Inserts the key-value tuple in the list sorted by the key
            if len(self.list) > self.max_length:
                del self.list[-1] #If length exceeds, remove the last element

    def get(self, index):
        if len(self.list) >= index:
            val = self.list[index][1]
            return val # vallue at index
