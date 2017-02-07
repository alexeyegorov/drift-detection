import numpy
import math
from Adwin.adwin_list_node import AdwinListNode


class AdwinList(object):

    def __init__(self, max_bucket):
        self.head = None
        self.tail = None
        self.count = 0
        self.max_bucket = max_bucket
        self.add_to_head()

    def add_to_head(self):
        temp = AdwinListNode(self.max_bucket)
        if self.head is not None:
            temp.next = self.head
            self.head.prev = temp
        self.head = temp
        if self.tail is None:
            self.tail = self.head
        self.count += 1

    def add_to_tail(self):
        temp = AdwinListNode(self.max_bucket)
        if self.tail is not None:
            temp.previous = self.tail
            self.tail.next = temp
        self.tail = temp
        if self.head is None:
            self.head = self.tail
        self.count += 1

    def remove_from_tail(self):
        temp = self.tail
        self.tail = self.tail.prev
        if self.tail is None:
            self.head = None
        else:
            self.tail.next = None
        self.count -= 1