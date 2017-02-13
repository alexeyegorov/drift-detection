""" Reservoir as a component of volatility detector """

# Authors: Wenjun Bai <vivianbai.cn@gmail.com>
#          Shu Shang <ignatius.sun@gmail.com>
#          Duyen Phuc Nguyen <nguyenduyenphuc@gmail.com>
# License: BSD 3 clause

import numpy as np
from buffer import calculate_stddev


class Reservoir:
    def __init__(self, size):
        """Initialize the reservoir with a given size.

        Parameters
        ----------
        size : int
            Size of the reservoir, the reservoir is initialized with zeros. The number of elements of the
            reservoir equals to this size.
        """
        self.size = size
        self.elements = []
        self.element_total = 0
        self.e_index = 0
        self.rand = np.random
        for i in range(self.size):
            self.elements.append(0.0)

    def add_element(self, input_value):
        """Add an element to the reservoir. As the sliding window slides, the oldest entry in the
        buffer is dropped from a buffer and moved into the reservoir, then the reservoir stores
        the dropped entry by randomly replacing one of its stored samples.

        Parameters
        ----------
        input_value: real value
            The input value to the buffer, randomly replacing one element in the reservoir
            The type of input value should be real value like int, float... because in this
            method the value will be used to calculate the statistics "total"
        """
        if self.e_index < self.size:
            self.elements[self.e_index] = input_value
            self.element_total += input_value
            self.e_index += 1
        else:
            index_remove = int(self.rand.rand() * self.e_index)
            self.element_total -= self.elements[index_remove]
            self.elements[index_remove] = input_value
            self.element_total += input_value

    def get_reservoir_mean(self):
        """Calculate the mean of the elements stored in reservoir"""
        return self.element_total / self.e_index

    def get_stddev(self):
        """Calculate the standard deviation of the elements stored in reservoir"""
        stddev = calculate_stddev(self.elements, self.get_reservoir_mean())
        if stddev == 0:
            return 0.00000000001
        else:
            return stddev

    def get_count(self):
        """Get the number of elements in the reservoir, this statistics is monitored by e_index"""
        return self.e_index

    def check_full(self):
        if self.e_index == self.size:
            return True
        else:
            return False

    def clear(self):
        self.elements[:] = []
        for i in range(self.size):
            self.elements.append(0.0)
        self.element_total = 0
        self.e_index = 0

    def check_is_clear(self):
        return self.e_index == 0
