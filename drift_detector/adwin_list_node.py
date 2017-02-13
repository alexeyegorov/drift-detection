"""Node implementation of adwin list data structure"""


# Authors: Wenjun Bai <vivianbai.cn@gmail.com>
#          Shu Shang <ignatius.sun@gmail.com>
#          Duyen Phuc Nguyen <nguyenduyenphuc@gmail.com>
# License: BSD 3 clause


class AdwinListNode(object):
    """Implementation of a node of adwin list"""

    def __init__(self, max_number_of_buckets):
        """Init a node with a given parameter number_of_buckets

        Parameters
        ----------
        max_number_of_buckets : In each row, the max number of buckets
        """
        self.max_number_of_buckets = max_number_of_buckets
        self.size = 0
        self.next = None
        self.prev = None
        self.sum = []
        self.variance = []
        for i in range(self.max_number_of_buckets + 1):
            self.sum.append(0.0)
            self.variance.append(0.0)

    def insert_bucket(self, value, variance):
        """Insert a bucket at the end

        Parameters
        ----------
        value: the totally size of the new one
        variance : the variance of the new one
        """
        self.sum[self.size] = value
        self.variance[self.size] = variance
        self.size += 1

    def drop_bucket(self, n=1):
        """Drop the older portion of the bucket

        Parameters
        ----------
        n :number data of drop bucket
        """
        for k in range(n, self.max_number_of_buckets + 1):
            self.sum[k - n] = self.sum[k]
            self.variance[k - n] = self.variance[k]
        for k in range(1, n + 1):
            self.sum[self.max_number_of_buckets - k + 1] = 0.0
            self.variance[self.max_number_of_buckets - k + 1] = 0.0
        self.size -= n
