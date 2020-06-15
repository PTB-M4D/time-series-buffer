from time_series_buffer import TimeSeriesBuffer
import numpy as np 


def add_array():
    tsb = TimeSeriesBuffer(maxlen=50)

    N = 10

    for M in [2,3,4]:
        new_data = np.random.random((N, M))
        tsb.add(data=new_data)
        print(tsb)

def add_uarray():
    pass

def add_arrays():
    pass

def add_uarrays():
    pass

def add_mixed_types():
    # mix float, ufloat, array and uarray
    pass

def error_on_shape_mismatch():
    pass

def pop_size_and_order():
    # check output length and buffer size after pop
    # check if really oldest elements are received
    pass

def show_size_and_order():
    # check output length and buffer size after pop
    # check if newest elements are received and order is as expected
    pass

def test_return_formats():
    # 
    pass
