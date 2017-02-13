import numpy
import math

class AdWinListNode(object):

    def __init__ (self, M):
        self.M = M
        self.size = 0
        self.next = None
        self.prev = None
        self.sum = []
        self.variance = []
        for i in range (self.M+1):
            self.sum.append(0.0)
            self.variance.append(0.0)

    def addBack(self, value, var):

        self.sum[self.size] = value
        self.variance[self.size] = var
        self.size += 1

    def dropFront (self, n = 1):
        for k in range(n, self.M+1):
            self.sum[k-n] = self.sum[k]
            self.variance[k-n] = self.variance[k]
        for k in range(1, n+1):
            self.sum[self.M - k + 1] = 0.0
            self.variance[self.M - k + 1] = 0.0

        self.size -= n
