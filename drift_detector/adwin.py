"""ADaptive sliding WINdow  Algorithm (ADWIN)."""

# Authors: Wenjun Bai <vivianbai.cn@gmail.com>
#          Shu Shang <ignatius.sun@gmail.com>
#          Duyen Phuc Nguyen <nguyenduyenphuc@gmail.com>
# License: BSD 3 clause

import math

from drift_detector.adwin_list import AdwinList


class Adwin(object):
    """The Adwin algorithm is a change detector and estimator.
    It keeps a sliding (variable-length) window with the most
    recently read example,with the property that the window
    has the maximal length statistically consistent with the
    hypothesis that "there has been no change in the average
    value inside the window".

    References
    ----------
    A. Bifet, R. Gavalda. (2007). "Learning from Time-Changing
    Data with Adaptive Windowing". Proceedings of the 2007 SIAM
    International Conference on Data Mining 443-448.
    http://www.lsi.upc.edu/~abifet/Timevarying.pdf

    A. Bifet, J. Read, B.Pfahringer.G. Holmes, I. Zliobaite.
    (2013). "CD-MOA: Change Detection Framework for Massive Online
    Analysis". Springer Berlin Heidelberg 8207(9): 443-448.
    https://sites.google.com/site/zliobaitefiles/cdMOA-CR.pdf?attredirects=0
    """

    def __init__(self, delta=0.01):
        """Init the buckets

        Parameters
        ----------
        delta : float
            confidence value.
        """

        self.mint_clock = 1.0
        self.min_window_length = 16
        self.delta = delta
        self.max_number_of_buckets = 5
        self.bucket_list = AdwinList(self.max_number_of_buckets)
        self.mint_time = 0.0
        self.min_clock = self.mint_clock
        self.mdbl_error = 0.0
        self.mdbl_width = 0.0
        self.last_bucket_row = 0
        self.sum = 0.0
        self.width = 0.0
        self.variance = 0.0
        self.bucket_number = 0

    def get_estimation(self):
        """Get the estimation value"""
        if self.width > 0:
            return self.sum / float(self.width)
        else:
            return 0

    def set_input(self, value):
        """Add new element and reduce the window

        Parameters
        ----------
        value : new element

        Returns
        -------
        boolean: the return value of the method check_drift(), true if a drift was detected.
        """
        self.insert_element(value)
        self.compress_buckets()
        return self.check_drift()

    def length(self):
        """Get the length of window"""
        return self.width

    def insert_element(self, value):
        """insert new bucket"""
        self.width += 1
        self.bucket_list.head.insert_bucket(float(value), 0.0)
        self.bucket_number += 1
        if self.width > 1:
            self.variance += (self.width - 1) * (value - self.sum / (self.width - 1)) \
                             * (value - self.sum / (self.width - 1)) / self.width
        self.sum += value

    def compress_buckets(self):
        """
        Merge buckets.
        Find the number of buckets in a row, if the row is full, then merge the two buckets.
        """
        i = 0
        cont = 0
        cursor = self.bucket_list.head
        next_node = None
        while True:
            k = cursor.size
            if k == self.max_number_of_buckets + 1:
                next_node = cursor.next
                if next_node is None:
                    self.bucket_list.add_to_tail()
                    next_node = cursor.next
                    self.last_bucket_row += 1
                n1 = self.bucket_size(i)
                n2 = self.bucket_size(i)
                u1 = cursor.sum[0] / n1
                u2 = cursor.sum[1] / n2
                internal_variance = n1 * n2 * (u1 - u2) * (u1 - u2) / (n1 + n2)
                next_node.insert_bucket(cursor.sum[0] + cursor.sum[1],
                                        cursor.variance[0] + cursor.variance[1] + internal_variance)
                self.bucket_number -= 1
                cursor.drop_bucket(2)
                if next_node.size <= self.max_number_of_buckets:
                    break
                else:
                    break
            cursor = cursor.next
            i += 1
            if cursor is None:
                break

    def check_drift(self):
        """
        Reduce the window, detecting if there is a drift.

        Returns
        -------
        change : boolean value
        Result of whether the window has changed.
        """

        change = False
        exit = False
        cursor = None
        self.mint_time += 1
        if self.mint_time % self.min_clock == 0 and self.width > self.min_window_length:
            reduce_width = True
            while reduce_width:
                reduce_width = False
                exit = False
                n0 = 0.0
                n1 = float(self.width)
                u0 = 0.0
                u1 = float(self.sum)
                cursor = self.bucket_list.tail
                i = self.last_bucket_row
                while True:
                    for k in range(cursor.size):
                        if i == 0 and k == cursor.size - 1:
                            exit = True
                            break
                        n0 += self.bucket_size(i)
                        n1 -= self.bucket_size(i)
                        u0 += cursor.sum[k]
                        u1 -= cursor.sum[k]
                        min_length_of_subwindow = 5
                        if n0 >= min_length_of_subwindow and n1 >= min_length_of_subwindow and self.cut_expression(n0,
                                                                                                                   n1,
                                                                                                                   u0,
                                                                                                                   u1):
                            reduce_width = True
                            change = True
                            if self.width > 0:
                                self.delete_element()
                                exit = True
                                break
                    cursor = cursor.prev
                    i -= 1
                    if exit or cursor is None:
                        break
        return change

    def delete_element(self):
        """delete the bucket at the tail of window"""
        node = self.bucket_list.tail
        n1 = self.bucket_size(self.last_bucket_row)
        self.width -= n1
        self.sum -= node.sum[0]
        u1 = node.sum[0] / n1
        incVariance = float(
            node.variance[0] + n1 * self.width * (u1 - self.sum / self.width) * (u1 - self.sum / self.width)) / (
                          float(n1 + self.width))
        self.variance -= incVariance
        node.drop_bucket()
        self.bucket_number -= 1
        if node.size == 0:
            self.bucket_list.remove_from_tail()
            self.last_bucket_row -= 1

    def cut_expression(self, n0_, n1_, u0, u1):
        """Expression calculation"""
        n0 = float(n0_)
        n1 = float(n1_)
        n = float(self.width)
        diff = float(u0 / n0) - float(u1 / n1)
        v = self.variance / self.width
        dd = math.log(2.0 * math.log(n) / self.delta)
        min_length_of_subwindow = 5
        m = (float(1 / (n0 - min_length_of_subwindow + 1))) + (float(1 / (n1 - min_length_of_subwindow + 1)))
        eps = math.sqrt(2 * m * v * dd) + float(2 / 3 * dd * m)
        if math.fabs(diff) > eps:
            return True
        else:
            return False

    def bucket_size(self, Row):
        return int(math.pow(2, Row))
