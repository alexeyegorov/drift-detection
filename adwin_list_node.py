import numpy
import math

class AdwinListNode(object):

    def __init__(self, max_bucket):
        self.max_bucket = max_bucket
        self.bucket_size = 0
        self.next = None
        self.previous = None
        self.sum = []
        self.variance = []
        for i in range (self.max_bucket+1):
            self.sum.append(0.0)
            self.variance.append(0.0)

    def add_bucket(self, value, variance):
        self.sum[self.bucket_size] = value
        self.variance[self.bucket_size] = variance
        self.bucket_size += 1

    def compress_bucket(self, delete_num = 1):
        for k in range(delete_num, self.max_bucket+1):
            self.sum[k-delete_num] = self.sum[k]
            self.variance[k-delete_num] = self.variance[k]
        for k in range(1, delete_num+1):
            self.sum[self.max_bucket - k + 1] = 0.0
            self.variance[self.max_bucket - k + 1] = 0.0

        self.bucket_size -= delete_num