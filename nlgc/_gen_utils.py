# author: proloy Das <proloy@umd.edu>


class LazyProperty:
    "http://blog.pythonisito.com/2008/08/lazy-descriptors.html"
    def __init__(self, func):
        self._func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def __get__(self, obj, klass=None):
        if obj is None:
            return None
        result = obj.__dict__[self.__name__] = self._func(obj)
        return result