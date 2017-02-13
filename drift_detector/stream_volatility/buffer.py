""" Buffer as a component of volatility detector """

# Authors: Wenjun BAI <vivianbai.cn@gmail.com>
#          Shu SHANG <ignatius.sun@gmail.com>
#          Duyen Phuc Nguyen <nguyenduyenphuc@gmail.com>
# License: BSD 3 clause

import math
import numpy as np


def calculate_stddev(times, mean):
    count = 0
    sum = 0
    for d in times:
        if d > 0:
            count += 1
            sum += math.pow(d - mean, 2)
    if count == 0:
        return 0
    else:
        return math.sqrt(sum / count)


class Buffer:
    def __init__(self, size):
        """Initialize the buffer with the given size

        Parameters
        ----------
        size : int
            Size of the buffer, the buffer is initialized with zeros.
        """
        self.buffer = []
        self.size = size
        self.sliding_index = 0
        self.is_full = False
        self.total = 0
        for i in range(self.size):
            self.buffer.append(0.0)

    def add(self, value):
        """Add an element into the buffer

        Parameters
        ----------
        value : real value
            Input value, the old one in the buffer is removed, using a sliding index
            Total size of buffer is 32 by default.

        Returns
        -------
        removed: if the buffer is full, else return -1.
        """
        if self.sliding_index == self.size:
            self.is_full = True
            self.sliding_index = 0
            self.buffer[:] = []
            for i in range(self.size):
                self.buffer.append(0.0)

        removed = self.buffer[self.sliding_index]
        self.total -= removed

        self.buffer.append(value)
        self.sliding_index += 1
        self.total += value

        if self.is_full:
            return removed
        else:
            return -1

    def get_mean(self):
        """Calculate the mean value of the buffer"""
        if self.is_full:
            return self.total / self.size
        else:
            return self.total / self.sliding_index

    def get_stddev(self):
        """Calculate the standard deviation"""
        stddev = calculate_stddev(self.buffer, self.get_mean())
        if stddev == 0:
            return 0.00000000001
        else:
            return stddev

    def clear(self):
        """Clear the buffer, reset the parameters"""
        self.buffer[:] = []
        for i in range(self.size):
            self.buffer.append(0.0)
        self.sliding_index = 0
        self.is_full = False
        self.total = 0
