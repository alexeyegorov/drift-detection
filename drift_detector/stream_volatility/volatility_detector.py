# coding=utf-8
""" Relative Stream Volatility Detector """

# Authors: Wenjun Bai <vivianbai.cn@gmail.com>
#          Shu Shang <ignatius.sun@gmail.com>
#          Duyen Phuc Nguyen <nguyenduyenphuc@gmail.com>
# License: BSD 3 clause

from drift_detector.stream_volatility.buffer import Buffer
from drift_detector.stream_volatility.reservoir import Reservoir


class VolatilityDetector:
    """
    A drift detector is a detector that monitors the changes of stream volatility.
    Stream Volatility is the rate of changes of the detected changes given by a drift detector like Adwin.
    We can see this kind of detector as a drift detector the a set of given drifts and we call it volatility detector.

    A volatility detector takes the output of a drift detector and outputs an alarm if there is a change in the rate
    of detected drifts.

    The implementation uses two components: a buffer and a reservoir.
    The buffer is a sliding window that keeps the most recent samples of drift intervals acquired from
    a drift detection technique. The reservoir is a pool that stores previous samples which ideally represent
    the overall state of the stream.

    References
    ----------
    Huang, D.T.J., Koh, Y.S., Dobbie, G., Pears, R.: Detecting volatility shift in data streams.
    In: 2014 IEEE International Conference on Data Mining (ICDM), pp. 863–868 (2014)

    """
    def __init__(self, drift_detector, size):
        """
        Initialize a drift detector

        Parameters
        ----------
        drift_detector: type drift_detector
                    The volatility detector takes the output of a drift detector.
                    The corresponding drift detector is passed here to monitor its outputs.
        size: int
            Size of the reservoir and buffer by default.
        """
        self.drift_detector = drift_detector
        self.sample = 0
        self.reservoir = Reservoir(size)
        self.buffer = Buffer(size)
        self.confidence = 0.05
        self.recent_interval = []
        self.timestamp = 0
        self.vol_drift_found = False
        self.drift_found = False
        self.pre_drift_point = -1
        self.rolling_index = 0
        for i in range(size * 2 + 1):
            self.recent_interval.append(0.0)

    def set_input(self, input_value):
        """
        Main part of the algorithm, takes the drifts detected by a drift detector.

        Parameters
        ----------
        input_value: real value
                The input value of the volatility detector, the value should be real values and should be the output
                of some drift detector.

        Returns
        -------
        vol_drift_found: true if a drift of stream volatility was found.
        """
        self.sample += 1
        self.drift_found = self.drift_detector.set_input(input_value)
        if self.drift_found:
            self.timestamp += 1
            if self.buffer.is_full:
                result_buffer = self.buffer.add(self.timestamp)
                self.reservoir.add_element(result_buffer)
            else:
                self.buffer.add(self.timestamp)
            interval = self.timestamp
            self.recent_interval[self.rolling_index] = interval
            self.rolling_index += 1
            if self.rolling_index == self.reservoir.size * 2:
                self.rolling_index = 0
            self.timestamp = 0
            self.pre_drift_point = self.sample
            if self.buffer.is_full and self.reservoir.check_full():
                relative_var = self.buffer.get_stddev() / self.reservoir.get_stddev()
                if relative_var > (1.0 + self.confidence) or relative_var < (1.0 - self.confidence):
                    self.buffer.clear()
                    # self.severity_buffer[:] = []
                    self.vol_drift_found = True
                else:
                    self.vol_drift_found = False
        else:
            self.timestamp += 1
            self.vol_drift_found = False

        return self.vol_drift_found
