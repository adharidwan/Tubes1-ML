import numpy as np


class Tensor:

    def __init__(self, data, requires_grad=True, _prev=(), _op=""):
        self.data          = np.array(data, dtype=np.float64)
        self.grad          = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._prev         = list(_prev)
        self._backward     = lambda: None
        self._op           = _op

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, _prev=(self, other), _op="add")

        def _backward():
            if self.requires_grad:
                self.grad  += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, _prev=(self, other), _op="mul")

        def _backward():
            if self.requires_grad:
                self.grad  += _unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(self.data  * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        out = Tensor(-self.data, _prev=(self,), _op="neg")

        def _backward():
            if self.requires_grad:
                self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return self + (-other)

    def __rsub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return other + (-self)

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return self * other ** -1

    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return other * self ** -1

    def __pow__(self, exp):
        assert isinstance(exp, (int, float)), "Exponent must be a scalar."
        out = Tensor(self.data ** exp, _prev=(self,), _op=f"pow{exp}")

        def _backward():
            if self.requires_grad:
                self.grad += exp * (self.data ** (exp - 1)) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        out = Tensor(self.data @ other.data, _prev=(self, other), _op="matmul")

        def _backward():
            if self.requires_grad:
                self.grad  += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     _prev=(self,), _op="sum")

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / n

    def exp(self):
        out = Tensor(np.exp(self.data), _prev=(self,), _op="exp")

        def _backward():
            if self.requires_grad:
                self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(np.clip(self.data, 1e-15, None)),
                     _prev=(self,), _op="log")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad / np.clip(self.data, 1e-15, None)

        out._backward = _backward
        return out

    def clip(self, a_min, a_max):
        out = Tensor(np.clip(self.data, a_min, a_max), _prev=(self,), _op="clip")

        def _backward():
            if self.requires_grad:
                mask = (self.data >= a_min) & (self.data <= a_max)
                self.grad += out.grad * mask.astype(np.float64)

        out._backward = _backward
        return out

    def maximum(self, val):
        out = Tensor(np.maximum(self.data, val), _prev=(self,), _op="maximum")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (self.data > val).astype(np.float64)

        out._backward = _backward
        return out

    def backward(self):
        topo    = []
        visited = set()

        def build(node):
            if id(node) not in visited:
                visited.add(id(node))
                for parent in node._prev:
                    build(parent)
                topo.append(node)

        build(self)

        self.grad = np.ones_like(self.data)  # dL/dL = 1

        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def numpy(self):
        return self.data

    def item(self):
        return self.data.item()

    def __repr__(self):
        return (f"Tensor(data={self.data}, grad={self.grad}, "
                f"op='{self._op}')")

def _unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, (g_dim, s_dim) in enumerate(zip(grad.shape, shape)):
        if s_dim == 1 and g_dim != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad