from time_series_buffer import TimeSeriesBuffer
import numpy as np
from uncertainties import unumpy


maxlen = 50
N = 10

def test_add_array():

    # init the buffer
    tsb = TimeSeriesBuffer(maxlen=maxlen)
    counter = 0

    # fill buffer with arrays of different shape
    for M in [2, 3, 4]:
        data = np.random.random((N, M))

        # add data to buffer
        tsb.add(data=data)

        # check if number of buffer elements increased accordingly
        counter += len(data)
        assert len(tsb) == min(counter, maxlen)

        # check if latest element of buffer corresponds to latest element of data
        # (because M changes, we only check if the timestamp (index=0) matches)
        assert tsb.buffer[-1][0] == data[-1,0]

def test_add_uarray():
    tsb = TimeSeriesBuffer(maxlen=maxlen)

    M = 4
    data = np.random.random((N, M))
    udata = unumpy.uarray(data[:,[0,2]], data[:,[1,3]])
    
    # add data to buffer
    tsb.add(data=udata)

    # check if number of buffer elements increased accordingly
    assert len(tsb) == min(N, maxlen)

    # check if latest element of buffer corresponds to latest element of data
    assert tsb.buffer[-1] == tuple(data[-1,:])


def test_add_arrays():
    tsb = TimeSeriesBuffer(maxlen=maxlen)

    M = 4
    data = np.random.random((N, M))
    t = unumpy.uarray(data[:,0], data[:,1])
    v = unumpy.uarray(data[:,2], data[:,3])
    
    # add data to buffer
    tsb.add(time=t, val=v)

    # check if number of buffer elements increased accordingly
    assert len(tsb) == min(N, maxlen)

    # check if latest element of buffer corresponds to latest element of data
    assert tsb.buffer[-1] == tuple(data[-1,:])


def test_add_uarrays():
    tsb = TimeSeriesBuffer(maxlen=maxlen)

    M = 4
    data = np.random.random((N, M))
    t = unumpy.uarray(data[:,0], data[:,1])
    v = unumpy.uarray(data[:,2], data[:,3])
    
    # add data to buffer
    tsb.add(time=t, val=v)

    # check if number of buffer elements increased accordingly
    assert len(tsb) == min(N, maxlen)

    # check if latest element of buffer corresponds to latest element of data
    assert tsb.buffer[-1] == tuple(data[-1,:])

def test_add_mixed_types():
    # mix float, ufloat, array and uarray
    pass


def test_error_on_shape_mismatch():
    pass


def test_pop_size_and_order():
    # check output length and buffer size after pop
    # check if really oldest elements are received
    pass


def test_show_size_and_order():
    # check output length and buffer size after pop
    # check if newest elements are received and order is as expected
    pass


def test_return_formats():
    #
    pass
