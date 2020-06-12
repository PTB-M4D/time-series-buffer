from collections import deque
import numpy as np


class TimesSeriesBuffer:
    """
    Custom buffer class, that allows to save streams of time-series with uncertainty 
    in timestamps and values. Acts like a FIFO buffer.
    """

    def __init__(self, maxlen=10, return_type="array"):
        """Initialize a FIFO buffer.
        
        Parameters
        ----------
            maxlen: int (default: 10)
                maximum length of the buffer, directly handed over to deque

            return_type: str (default: array)

                * list: return data as list of tuples of (time, time_unc, val, val_unc)
                * array: float-array of shape (N, 4) where rows correspond to (time, time_unc, val, val_unc)
                * uarray: ufloat-array of shape (N, 2) where rows correspond to (time, val)
                * arrays: four float-arrays of shape (N, 1)
                * uarrays: two ufloat-arrays of shape (N, 1)
        
        """
        self.buffer = deque(maxlen=maxlen)
        self.return_type = return_type

    def __len__(self):
        return len(self.buffer)

    def add(self, data=None, time=None, time_unc=None, val=None, val_unc=None):
        """Append one or more new datapoints to the buffer. 
        A datapoint consists of the tuple (time, time_uncertainty, value, value_uncertainty).
        
        Parameters
        ----------
            data: iterable of iterables with shape (N, M) (default: None)
                If given, all other kwargs are ignored.
                
                * M==2 (pairs): assumed to be like (time, value)
                * M==3 (triple): assumed to be like (time, value, value_unc)
                * M==4 (4-tuple): assumed to be like (time, time_unc, value, value_unc)

            time: float, or iterable of float/ufloat (default: None)
                Timestamp(s) to be added.

            time_unc: float, or iterable of float (default: None)
                Uncertainty(ies) of the timestamp(s) to be added.

            val: (iterable of) float/ufloat (default: None)
                Value(s) to be added.

            val_unc: (iterable of) float (default: None)
                Uncertainty(ies) of the value(s) to be added.
    
            time, time_unc, val, val_unc need to be of same shape, but uncertainties can be omitted. 
        
        """
        # define list of supported iterable types
        it_types = [list, tuple, np.ndarray]

        # time series is given as rows of tuples
        if isinstance(data, it_types):

            if len(data[0]) == 2:  # pair
                for (t, v) in data:
                    self.buffer.append((t, np.nan, v, np.nan))

            elif len(data[0]) == 3:  # triple
                for (t, v, uv) in data:
                    self.buffer.append((t, np.nan, v, uv))

            elif len(data[0]) == 4:  # 4-tuple
                for (t, ut, v, uv) in data:
                    self.buffer.append((t, ut, v, uv))

        # time series is given as iterable of floats
        elif isinstance(time, it_types) and isinstance(val, it_types):
            if isinstance(unc, it_types):
                for t, v, u in zip(time, val, unc):
                    self.buffer.append((t, v, u))
            else:
                for t, v in zip(time, val):
                    self.buffer.append((t, v, np.nan))

    def pop(self, n_samples=1):
        """
        Return the next `n_samples` from the left side of the buffer.

        View the latest `n` additions to the buffer. Returns the same format that
        :py:func:`append_multi` accepts.
        
        Parameters
        ----------
            n: int (default: 1)
                How many datapoints to return.
        
        Return
        ------
            Depends on return_type, see :py:func:`show` for details?
        """

        # take the next samples from the beginning of buffer
        n_pop = min(n_samples, len(self.buffer))
        next_samples = [self.buffer.popleft() for i in range(n_pop)]

        # return as numpy array
        return np.array(next_samples)

    def show(self, n_samples=1):
        """
        View the latest `n` additions to the buffer. Returns the same format that
        :py:func:`append_multi` accepts.
        """

        # take the next samples from the beginning of buffer
        n_buffer = len(self.buffer)
        n_pop = min(n_samples, n_buffer)
        next_samples = [self.buffer[n_buffer - (i + 1)] for i in range(n_pop)]

        # return as numpy array
        return np.vstack(next_samples)
