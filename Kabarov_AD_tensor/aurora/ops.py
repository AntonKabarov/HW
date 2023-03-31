from .tensor import Tensor


def add(a, b):
    return a.add(b)


def dot(a, b):
    return a.dot(b)


def mul(a, b):
    return a.mul(b)


def norm(a, ord,p=1):
    return a.norm(ord,p)


def sub(a, b):
    return a.sub(b)


def ones(shape):
    lst = [1] * shape[0]
    return Tensor(lst)


def randn(shape):
    from random import gauss
    return Tensor([gauss(0, 1) for _ in range(shape[0])])


def tensor(data):
    lst = data
    return Tensor(lst)

def rmul(a,b):
    return a.rmul(b)
    
def zeros(shape):
    lst = [0] * shape[0]
    return Tensor(lst)
