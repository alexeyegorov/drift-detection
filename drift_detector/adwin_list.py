"""Implementation of an adwin list"""

# Authors: Wenjun Bai <vivianbai.cn@gmail.com>
#          Shu Shang <ignatius.sun@gmail.com>
#          Duyen Phuc Nguyen <nguyenduyenphuc@gmail.com>
# License: BSD 3 clause


from drift_detector.adwin_list_node import AdwinListNode


class AdwinList(object):
    def __init__(self, max_number_bucket):
        """Init a adwin list with a given parameter max_number_buckets

        Parameters
        ----------
        max_number_bucket : max number of elements in the bucket
        """
        self.head = None
        self.tail = None
        self.count = 0
        self.max_number_bucket = max_number_bucket
        self.add_to_head()

    def add_to_tail(self):
        """add a node at the tail of adwin list, used in the initialization of an AdwinList"""
        temp = AdwinListNode(self.max_number_bucket)
        if self.tail is not None:
            temp.prev = self.tail
            self.tail.next = temp
        self.tail = temp
        if self.head is None:
            self.head = self.tail
        self.count += 1

    def add_to_head(self):
        """Add a node to the head of an AdwinList"""
        temp = AdwinListNode(self.max_number_bucket)
        if self.head is not None:
            temp.next = self.head
            self.head.prev = temp
        self.head = temp
        if self.tail is None:
            self.tail = self.head
        self.count += 1

    def remove_from_head(self):
        """Remove the head node of an AdwinList"""
        temp = self.head
        self.head = self.head.next
        if self.head is not None:
            self.head.prev = None
        else:
            self.tail = None
        self.count -= 1

    def remove_from_tail(self):
        """Remove the tail node of an AdwinList"""
        temp = self.tail
        self.tail = self.tail.prev
        if self.tail is None:
            self.head = None
        else:
            self.tail.next = None
        self.count -= 1
