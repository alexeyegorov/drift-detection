import numpy
import math
from detector.ad_win_list_node import AdWinListNode

class AdWinList(object):

    def __init__(self, M):
        self.head = None
        self.tail = None
        self.count = 0
        self.M = M
        self.addToHead()

    def addToTail(self):
        temp = AdWinListNode(self.M)
        if self.tail is not None:
            temp.prev = self.tail
            self.tail.next = temp
        self.tail = temp
        if self.head is None:
            self.head = self.tail
        self.count += 1

    def removeFromHead(self):
        temp = self.head
        self.head = self.head.next
        if self.head is not None:
            self.head.prev = None
        else:
            self.tail = None
        self.count -= 1

    def addToHead(self):
        temp = AdWinListNode(self.M)
        if self.head is not None:
            temp.next = self.head
            self.head.prev = temp
        self.head = temp
        if self.tail is None:
            self.tail = self.head
        self.count += 1

    def removeFromTail(self):
        temp = self.tail
        self.tail = self.tail.prev
        if self.tail is None:
            self.head = None
        else:
            self.tail.next = None
        self.count -= 1
