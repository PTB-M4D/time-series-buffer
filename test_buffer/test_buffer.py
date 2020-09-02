import numpy as np
import pytest
import uncertainties
from uncertainties import unumpy

from time_series_buffer import TimeSeriesBuffer

maxlen = 50
N = 10
n_samples = 7


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
        assert tsb.buffer[-1][0] == data[-1, 0]


def test_add_uarray():
    tsb = TimeSeriesBuffer(maxlen=maxlen)

    M = 4
    data = np.random.random((N, M))
    udata = unumpy.uarray(data[:, [0, 2]], data[:, [1, 3]])

    # add data to buffer
    tsb.add(data=udata)

    # check if number of buffer elements increased accordingly
    assert len(tsb) == min(N, maxlen)

    # check if latest element of buffer corresponds to latest element of data
    assert tsb.buffer[-1] == tuple(data[-1, :])


def test_add_arrays():
    tsb = TimeSeriesBuffer(maxlen=maxlen)

    M = 4
    data = np.random.random((N, M))
    t = data[:, 0]
    ut = data[:, 1]
    v = data[:, 2]
    uv = data[:, 3]

    # add data to buffer in different combinations
    # and check if latest element of buffer corresponds to latest element of data
    ignore = tsb.empty_unc

    # no uncertainty information
    tsb.add(time=t, val=v)
    assert tsb.buffer[-1] == (data[-1, 0], ignore, data[-1, 2], ignore)

    # typical uncertainty information
    tsb.add(time=t, val=v, val_unc=uv)
    assert tsb.buffer[-1] == (data[-1, 0], ignore, data[-1, 2], data[-1, 3])

    # full uncertainty information
    tsb.add(time=t, time_unc=ut, val=v, val_unc=uv)
    assert tsb.buffer[-1] == (data[-1, 0], data[-1, 1], data[-1, 2], data[-1, 3])

    # untypical uncertainty information
    tsb.add(time=t, time_unc=ut, val=v)
    assert tsb.buffer[-1] == (data[-1, 0], data[-1, 1], data[-1, 2], ignore)


def test_add_uarrays():
    tsb = TimeSeriesBuffer(maxlen=maxlen)

    M = 4
    data = np.random.random((N, M))
    t = unumpy.uarray(data[:, 0], data[:, 1])
    v = unumpy.uarray(data[:, 2], data[:, 3])

    # add data to buffer
    tsb.add(time=t, val=v)

    # check if number of buffer elements increased accordingly
    assert len(tsb) == min(N, maxlen)

    # check if latest element of buffer corresponds to latest element of data
    assert tsb.buffer[-1] == tuple(data[-1, :])


def test_add_mixed_types():
    # mix float, ufloat, array and uarray
    pass


def test_error_on_shape_mismatch():
    tsb = TimeSeriesBuffer(maxlen=maxlen)

    M = 4
    data = np.random.random((N, M))
    t = data[:, 0]
    ut = data[:, 1]
    v = data[:, 2]
    uv = data[:, 3]

    with pytest.raises(ValueError):
        tsb.add(time=t, time_unc=ut[:-2], val=v, val_unc=uv)

    with pytest.raises(ValueError):
        tsb.add(time=t, time_unc=ut, val=v[:-2], val_unc=uv)

    with pytest.raises(ValueError):
        tsb.add(time=t, time_unc=ut, val=v, val_unc=uv[:-2])


def test_pop_size_and_order():
    n_samples = 7

    M = 4
    tsb = TimeSeriesBuffer(maxlen=maxlen)
    data = np.random.random((N, M))
    tsb.add(data=data)

    # pop some buffer elements
    length_before_pop = len(tsb)
    result = tsb.pop(n_samples=n_samples)
    length_after_pop = len(tsb)

    # check output length and buffer size after pop
    if n_samples > length_before_pop:
        assert 0 == length_after_pop
    else:
        assert length_before_pop - n_samples == length_after_pop

    # check if really oldest elements are received
    assert np.all(result == data[0:n_samples, :])


def test_show_size_and_order():
    n_samples = 7

    M = 4
    tsb = TimeSeriesBuffer(maxlen=maxlen, return_type="array")
    data = np.random.random((N, M))
    tsb.add(data=data)

    # return some buffer elements without pop
    length_before_show = len(tsb)
    result = tsb.show(n_samples=n_samples)
    length_after_show = len(tsb)

    # check that show does not alter the buffer
    assert length_before_show == length_after_show

    # check if newest elements are received and order is as expected
    assert np.all(result == data[-n_samples:, :])


def test_show_all():

    M = 4
    tsb = TimeSeriesBuffer(maxlen=maxlen, return_type="array")
    data = np.random.random((N, M))
    tsb.add(data=data)

    # return some buffer elements without pop
    result = tsb.show(n_samples=-1)

    # check that show does not alter the buffer
    assert len(result) == N

    # check if newest elements are received and order is as expected
    assert np.all(result == data)


def test_return_format_list():
    M = 4
    tsb = TimeSeriesBuffer(maxlen=maxlen, return_type="list")
    data = np.random.random((N, M))
    tsb.add(data=data)

    result = tsb.show(n_samples=n_samples)
    assert isinstance(result, list)
    assert isinstance(result[0], tuple)
    assert isinstance(result[0][0], float)


def test_return_format_array():
    M = 4
    tsb = TimeSeriesBuffer(maxlen=maxlen, return_type="array")
    data = np.random.random((N, M))
    tsb.add(data=data)

    result = tsb.show(n_samples=n_samples)
    assert isinstance(result, np.ndarray)
    assert isinstance(result[0], np.ndarray)
    assert isinstance(result[0][0], float)


def test_return_format_arrays():
    M = 4
    tsb = TimeSeriesBuffer(maxlen=maxlen, return_type="arrays")
    data = np.random.random((N, M))
    tsb.add(data=data)

    t, ut, v, uv = tsb.show(n_samples=n_samples)
    assert isinstance(t, np.ndarray)
    assert isinstance(ut, np.ndarray)
    assert isinstance(v, np.ndarray)
    assert isinstance(uv, np.ndarray)
    assert isinstance(t[0], float)
    assert isinstance(ut[0], float)
    assert isinstance(v[0], float)
    assert isinstance(uv[0], float)


def test_return_format_uarray():
    M = 4
    tsb = TimeSeriesBuffer(maxlen=maxlen, return_type="uarray")
    data = np.random.random((N, M))
    tsb.add(data=data)

    result = tsb.show(n_samples=n_samples)
    assert isinstance(result, np.ndarray)
    assert isinstance(result[0], np.ndarray)
    assert isinstance(result[0, 0], uncertainties.core.Variable)
    assert isinstance(result[0, 1], uncertainties.core.Variable)


def test_return_format_uarrays():
    M = 4
    tsb = TimeSeriesBuffer(maxlen=maxlen, return_type="uarrays")
    data = np.random.random((N, M))
    tsb.add(data=data)

    t, v = tsb.show(n_samples=n_samples)
    assert isinstance(t, np.ndarray)
    assert isinstance(v, np.ndarray)
    assert isinstance(t[0], uncertainties.core.Variable)
    assert isinstance(v[0], uncertainties.core.Variable)


def test_pop_empty_array():
    tsb = TimeSeriesBuffer(maxlen=maxlen, return_type="array")
    result = tsb.pop()

    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 4)
    assert result.size == 0


def test_pop_empty_arrays():
    tsb = TimeSeriesBuffer(maxlen=maxlen, return_type="arrays")
    t, ut, v, uv = tsb.show(n_samples=n_samples)

    assert isinstance(t, np.ndarray)
    assert isinstance(ut, np.ndarray)
    assert isinstance(v, np.ndarray)
    assert isinstance(uv, np.ndarray)
    assert t.size == 0
    assert ut.size == 0
    assert v.size == 0
    assert uv.size == 0


def test_pop_empty_uarray():
    tsb = TimeSeriesBuffer(maxlen=maxlen, return_type="uarray")
    result = tsb.show(n_samples=n_samples)

    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 2)
    assert result.size == 0


def test_pop_empty_uarrays():
    tsb = TimeSeriesBuffer(maxlen=maxlen, return_type="uarrays")
    t, v = tsb.show(n_samples=n_samples)

    assert isinstance(t, np.ndarray)
    assert isinstance(v, np.ndarray)
    assert t.size == 0
    assert v.size == 0
