class Tensor:
    def __init__(self, data):
        import array
        for x in data:
            if isinstance(x, int) or isinstance(x, float):
                continue
            raise TypeError('now we should support only 1d tensors')

        self.__data = array.array('f', data)
        self.__shape = (len(data), )

    @property
    def shape(self):
        return self.__shape

    def __repr__(self):
        return f"Tensor([{', '.join([f'{x:.4f}' for x in self.__data])}])"

    def isclose(self, other):
        from math import isclose
        return all([isclose(a, b, abs_tol=1e-5) for a, b in zip(self.__data, other.__data)])

    def add(self, other):
    	return Tensor([p+q for p,q in zip(self.__data,other.__data)])

    def dot(self, other):
        return sum([p*q for p,q in zip(self.__data,other.__data)])

    def mul(self, scalar):
        return Tensor([i*scalar for i in self.__data])
    
    def rmul(self, scalar):
        return Tensor([scalar*i for i in self.__data])

    def norm(self, ord=2,p=1):
        import math
        res = 0
        if ord != 1 and ord != 2 and ord != 3:
            raise ValueError('illegal ord value')
        elif ord ==3:
         res = pow(sum([abs(i)**p for i in self.__data]),1. / p)
        elif ord ==2:
         res = math.sqrt(sum([i*i for i in self.__data]))
        else:
         res =sum([abs(i) for i in self.__data])
        return res

    def sub(self, other):
        return Tensor([p-q for p,q in zip(self.__data,other.__data)])

    def __add__(self, other):
        return self.add(other)

    def __mul__(self, other):
        return self.mul(other)

    def __neg__(self):
        return self.mul(-1)

    def __rmul__(self, other):
        return self.rmul(other)

    def __sub__(self, other):
        return self.sub(other)

    def __matmul__(self, other):
        return self.dot(other)

    def __eq__(self, other):
        return self.isclose(other)
