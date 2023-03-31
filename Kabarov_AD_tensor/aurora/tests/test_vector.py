from ..tensor import Tensor
from ..ops import *


def test_shape():
    assert Tensor([]).shape == (0,)
    assert Tensor([0]).shape == (1,)
    assert Tensor([0, 0, 0, 0, 0]).shape == (5,)


def test_add():
    a = Tensor([1, 2, 3])
    b = Tensor([2, 3, 4])
    a_plus_b = Tensor([3, 5, 7])

    assert a.add(b) == a_plus_b
    assert b.add(a) == a_plus_b
    assert a.add(b) == b.add(a)


def test_magic_add():
    a = Tensor([1, 2, 3])
    b = Tensor([2, 3, 4])
    a_plus_b = Tensor([3, 5, 7])

    assert a + b == a_plus_b
    assert b + a == a_plus_b
    assert a.add(b) == b.add(a)


def test_mul():
    a = Tensor([1, 2, 3])
    double_a = Tensor([2, 4, 6])

    assert a.mul(2) == double_a


def test_magic_mul():
    a = Tensor([1, 2, 3])
    double_a = Tensor([2, 4, 6])

    assert a * 2 == double_a
    assert 2 * a == double_a
    assert a * 2 == 2 * a


def test_magic_neg():
    a = Tensor([1, 2, 3])
    minus_a = Tensor([-1, -2, -3])

    assert a * (-1) == minus_a
    assert -1 * a == minus_a
    assert -a == minus_a


def test_dot():
    import math

    a = Tensor([1, 2, 3])
    b = Tensor([1, 2, 3])

    assert math.isclose(a.dot(b), 14)

    a = Tensor([1, 2, 3])
    b = Tensor([2, 3, 4])

    assert math.isclose(a.dot(b), 20)
    assert math.isclose(b.dot(a), 20)


def test_magic_matmul():
    import math

    a = Tensor([1, 2, 3])
    b = Tensor([1, 2, 3])

    assert math.isclose(a @ b, 14)

    a = Tensor([1, 2, 3])
    b = Tensor([2, 3, 4])

    assert math.isclose(a @ b, 20)
    assert math.isclose(b @ a, 20)


def test_norm_l1():
    a = Tensor([1, 2, 3])
    b = Tensor([-3, 4])

    assert a.norm(ord=1) == 6
    assert b.norm(ord=1) == 7


def test_norm_l2():
    a = Tensor([1])
    b = Tensor([-3, 4])

    assert a.norm(ord=2) == 1
    assert b.norm(ord=2) == 5


def test_sub():
    a = Tensor([2, 3, 4])
    b = Tensor([1, 2, 3])

    a_minus_b = Tensor([1, 1, 1])
    assert a.sub(b) == a_minus_b
    assert a - b == a_minus_b


def test_tensor():
    assert tensor([]) == Tensor([])
    assert tensor([1]) == Tensor([1])
    assert tensor([1, 2, 3, 4]) == Tensor([1, 2, 3, 4])


def test_ones():
    assert Tensor([]) == ones((0,))
    assert Tensor([1]) == ones((1,))
    assert Tensor([1, 1, 1, 1]) == ones((4,))


def test_zeros():
    assert Tensor([]) == zeros((0,))
    assert Tensor([0]) == zeros((1,))
    assert Tensor([0, 0, 0, 0, 0]) == zeros((5,))


def test_ops_add():
    a = Tensor([1, 2, 3])
    b = Tensor([2, 3, 4])
    a_plus_b = Tensor([3, 5, 7])

    assert add(a, b) == a_plus_b
    assert add(b, a) == a_plus_b
    assert add(a, b) == add(b, a)


def test_axioms():
    import random

    test_shape = (random.randint(5, 10),)
    alpha = random.gauss(0, 1)
    beta = random.gauss(0, 1)

    a, b, c = randn(test_shape), randn(test_shape), randn(test_shape)
    zero = zeros(test_shape)

    assert a + b == b + a
    assert (a + b) + c == a + (b + c)
    assert a + zero == a
    assert a + (-a) == zero

    assert alpha * (beta * a) == (alpha * beta) * a
    assert (alpha + beta) * a == alpha * a + beta * a
    assert alpha * (a + b) == alpha * a + alpha * b
    assert 1 * a == a


def test_stress_axioms():
    for _ in range(100):
        test_axioms()
