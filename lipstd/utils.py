from collections.abc import Iterable
import torch


# Wrapper class, adapted to Python 3.8 from https://stackoverflow.com/questions/9057669/how-can-i-intercept-calls-to-pythons-magic-methods-in-new-style-classes/9059858#9059858

class MetaClass(type):  # create proxies for wrapped object's double-underscore attributes
    def __init__(cls, name, bases, dct):

        def make_proxy(name):
            def proxy(self, *args):
                return getattr(self._obj, name)
            return proxy

        type.__init__(cls, name, bases, dct)
        if cls.__wraps__:
            ignore = set("__%s__" % n for n in cls.__ignore__.split())
            for name in dir(cls.__wraps__):
                if name.startswith("__"):
                    if name not in ignore and name not in dct:
                        setattr(cls, name, property(make_proxy(name)))


class Wrapper(object, metaclass=MetaClass):
    """Wrapper class that provides proxy access to an instance of some
       internal instance."""

    __wraps__ = None
    __ignore__ = "class mro new init setattr getattr getattribute"

    def __init__(self, obj):
        if self.__wraps__ is None:
            raise TypeError("base class Wrapper may not be instantiated")
        elif isinstance(obj, self.__wraps__):
            self._obj = obj
        else:
            raise ValueError("wrapped object must be of %s" % self.__wraps__)

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._obj, name)


def to_one_hot(x, size):
    x_one_hot = x.new_zeros(x.size(0), size)
    x_one_hot.scatter_(1, x.unsqueeze(-1).long(), 1).float()

    return x_one_hot


def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)) and not torch.is_tensor(el):
            yield from flatten(el)
        else:
            yield el
