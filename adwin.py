import numpy
import math
from Adwin.adwin_list import AdwinList

class Adwin(object):
       def __init__(self, delta=0.01, min_clock=32, min_length_window=10, max_bucket=5):
           """
                   :param delta: confidence value
                   :param max_bucket: max number of buckets which have same number of original data in one row
                   :param min_clock: min number of new data for starting to reduce window and detect change
                   :param min_length_window: start to reduce the window and detect the change
           """
           self.delta = delta
           self.min_clock = self.min_clock
           self.min_length_window = min_length_window
           self.max_bucket = max_bucket
           self.list_bucket_row = AdwinList(self.max_bucket)
           self.time = 0.0
           self.sum = 0.0
           self.width = 0.0
           self.variance = 0.0
           self.bucket_number=0
           self.last_bucket_row = 0
           # last_bucket_row: the max number of merge

       def update(self, value):
           self.insert_element(value)
           self.compress_bucket()
           return self.check_drift()

       # insert new bucket
       def insert_element(self, value):
           self.width += 1
           self.list_bucket_row.head.add_bucket(float(value), 0.0)
           self.bucket_number += 1

           if self.width > 1:
               self.variance += (self.width - 1) * (value - self.sum / (self.width - 1)) * (
                    value - self.sum / (self.width - 1)) / self.width
           self.sum += value

       # merge the two bucket to a new one
       def compress_bucket(self):
           i = 0
           cursor = self.list_bucket_row.head
           next_node = None

           while True:
               k = cursor.bucket_row_size
               if k == self.max_bucket + 1:
                   next_node = cursor.next
                   if next_node is None:
                       self.list_bucket_row.add_to_tail()
                       next_node = cursor.next
                       self.last_bucket_row += 1
                   n1 = self.bucket_size(i)
                   n2 = self.bucket_size(i)
                   u1 = cursor.sum[0] / n1
                   u2 = cursor.sum[1] / n2
                   internal_variance = n1 * n2 * (u1 - u2) * (u1 - u2) / (n1 + n2)
                   next_node.add_bucket(cursor.sum[0] + cursor.sum[1],
                                        cursor.variance[0] + cursor.variance[1] + internal_variance)
                   self.bucket_number -= 1
                   cursor.compress_bucket(2)
                   if next_node.size <= self.max_bucket:
                       break
                   else:
                       break
               cursor = cursor.next
               i += 1
               if cursor is None:
                   break

       def bucket_size(self, row):
           return int(math.pow(2, row))

       # check whether the window change
       def check_drift(self):
           change = False
           quit = False
           cursor = None
           self.time += 1

           if self.time % self.min_clock == 0 and self.width > self.min_length_window:
               reduce_width = True
               while reduce_width:
                   reduce_width = False
                   quit = False
                   n0 = 0.0
                   n1 = float(self.width)
                   u0 = 0.0
                   u1 = float(self.sum)
                   cursor = self.list_bucket_row.tail
                   i = self.last_bucket_row

                   while True:
                       # while quit = True & cursor <> None
                       for k in range(cursor.bucket_row_size):
                           if i == 0 and k == cursor.bucket_row_size - 1:
                               quit = True
                               break
                           n0 += self.bucket_size(i)
                           n1 -= self.bucket_size(i)
                           u0 += cursor.sum[k]
                           u1 -= cursor.sum[k]
                           mintMinWinLength = 5
                           if n0 >= mintMinWinLength and n1 >= mintMinWinLength \
                                   and self.cut_expression(n0, n1, u0, u1):
                               reduce_width = True
                               change = True
                               if self.width > 0:
                                   self.delete_element()
                                   quit = True
                                   break
                       cursor = cursor.prev
                       i -= 1
                       if quit or cursor is None:
                           break
           return change

       # remove the bucket from the tail of window
       def delete_element(self):
           node = self.list_bucket_row.tail
           delete_num = self.bucket_size(self.last_bucket_row)
           self.width -= delete_num
           self.sum -= node.sum[0]
           mean = node.sum[0] / delete_num
           internal_variance = float(node.variance[0] + delete_num * self.width * (mean - self.sum / self.width) *
                                     (mean - self.sum / self.width)) / (float(delete_num + self.width))
           self.variance -= internal_variance
           node.compress_bucket()
           # delete the bucket
           self.bucket_number -= 1
           if node.bucker_row_size == 0:
               self.list_bucket_row.remove_from_tail()
               self.last_bucket_row -= 1

       def cut_expression(self, n0, n1, u0, u1):
           n0 = float(n0)
           n1 = float(n1)
           n = float(self.width)
           dd = math.log(2.0 * math.log(n) / self.delta)
           v = self.variance / self.width
           diff = float(u0 / n0) - float(u1 / n1)
           mintMinWinLength = 5
           m = (float(1 / ((n0 - mintMinWinLength + 1)))) + (float(1 / ((n1 - mintMinWinLength + 1))))
           epsilon = math.sqrt(2 * m * v * dd) + float(2 / 3 * dd * m)

           if math.fabs(diff) > epsilon:
               return True
           else:
               return False









