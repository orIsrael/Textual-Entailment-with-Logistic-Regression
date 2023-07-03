import time


def timing(f):
    def wrap(*args, **kwargs):
        s = time.time()
        ret = f(*args, **kwargs)
        e = time.time()
        print('{:s} function took {:.3f} seconds'.format(f.__name__, e - s))

        return ret
    return wrap
