# coding=utf-8
""" Drift detection method based in DDM method of Joao Gama SBIA 2004. """

# Authors: Wenjun Bai <vivianbai.cn@gmail.com>
#          Shu Shang <ignatius.sun@gmail.com>
#          Duyen Phuc Nguyen <nguyenduyenphuc@gmail.com>
# License: BSD 3 clause

import sys
import math


class DDM:
    """
    The drift detection method (DDM) controls the number of errors
    produced by the learning model during prediction. It compares
    the statistics of two windows: the first contains all the data,
    and the second contains only the data from the beginning until
    the number of errors increases.
    Their method doesn't store these windows in memory.
    It keeps only statistics and a window of recent errors data.".

    References
    ---------
    Gama, J., Medas, P., Castillo, G., Rodrigues, P.:
    "Learning with drift detection". In: Bazzan, A.L.C., Labidi,
    S. (eds.) SBIA 2004. LNCS (LNAI), vol. 3171, pp. 286–295. Springer, Heidelberg (2004)
    """

    def __init__(self):
        self.m_n = 1
        self.m_p = 1
        self.m_s = 0
        self.m_psmin = sys.float_info.max
        self.m_pmin = sys.float_info.max
        self.m_smin = sys.float_info.max
        self.change_detected = False
        self.is_initialized = True
        self.estimation = 0.0
        self.is_warning_zone = False

    def set_input(self, prediction):
        """
        The number of errors in a sample of n examples is modelled by a binomial distribution.
        For each point t in the sequence that is being sampled, the error rate is the probability
        of mis-classifying p(t), with standard deviation s(t).
        DDM checks two conditions:
        1) p(t) + s(t) > p(min) + 2 * s(min) for the warning level
        2) p(t) + s(t) > p(min) + 3 * s(min) for the drift level

        Parameters
        ----------
        prediction : new element, it monitors the error rate

        Returns
        -------
        change_detected : boolean
                    True if a change was detected.
        """
        if self.change_detected is True or self.is_initialized is False:
            self.reset()
            self.is_initialized = True

        self.m_p += (prediction - self.m_p) / float(self.m_n)
        self.m_s = math.sqrt(self.m_p * (1 - self.m_p) / float(self.m_n))

        self.m_n += 1
        self.estimation = self.m_p
        self.change_detected = False

        if self.m_n < 30:
            return False

        if self.m_p + self.m_s <= self.m_psmin:
            self.m_pmin = self.m_p;
            self.m_smin = self.m_s;
            self.m_psmin = self.m_p + self.m_s;

        if self.m_p + self.m_s > self.m_pmin + 3 * self.m_smin:
            self.change_detected = True
        elif self.m_p + self.m_s > self.m_pmin + 2 * self.m_smin:
            self.is_warning_zone = True
        else:
            self.is_warning_zone = False

        return self.change_detected

    def reset(self):
        """reset the DDM drift detector"""
        self.m_n = 1
        self.m_p = 1
        self.m_s = 0
        self.m_psmin = sys.float_info.max
        self.m_pmin = sys.float_info.max
        self.m_smin = sys.float_info.max
