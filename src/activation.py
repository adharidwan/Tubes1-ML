from tensor import Tensor

class Linear:
    def __call__(self, x: Tensor) -> Tensor:
        return x


class ReLU:
    def __call__(self, x: Tensor) -> Tensor:
        return x.maximum(0)


class Sigmoid:
    def __call__(self, x: Tensor) -> Tensor:
        return Tensor(1.0, requires_grad=False) / (Tensor(1.0, requires_grad=False) + (-x).exp())


class Tanh:
    def __call__(self, x: Tensor) -> Tensor:
        e_pos = x.exp()
        e_neg = (-x).exp()
        return (e_pos - e_neg) / (e_pos + e_neg)


class Softmax:
    def __call__(self, x: Tensor) -> Tensor:
        x_shifted = x - Tensor(x.data.max(axis=-1, keepdims=True), requires_grad=False)
        exp_x = x_shifted.exp()
        return exp_x / exp_x.sum(axis=-1, keepdims=True)