import numpy as np
from tensor import Tensor


class Linear:

    def __init__(self, in_features: int, out_features: int, init: str = None, seed: int = None, **kwargs):
        self.in_features  = in_features
        self.out_features = out_features

        rng = np.random.default_rng(seed)

        if init == "zero":
            W_data = np.zeros((in_features, out_features))

        elif init == "uniform":
            low  = kwargs.get("low",  -1.0)
            high = kwargs.get("high",  1.0)
            W_data = rng.uniform(low, high, (in_features, out_features))

        elif init == "normal":
            mean = kwargs.get("mean", 0.0)
            std  = np.sqrt(kwargs.get("var", 1.0))
            W_data = rng.normal(mean, std, (in_features, out_features))

        elif init == "xavier":
            limit  = np.sqrt(6.0 / (in_features + out_features))
            W_data = rng.uniform(-limit, limit, (in_features, out_features))

        elif init == "he":
            std    = np.sqrt(2.0 / in_features)
            W_data = rng.normal(0.0, std, (in_features, out_features))

        else:
            raise ValueError(f"Unknown init method: '{init}'")

        self.W = Tensor(W_data)                      
        self.b = Tensor(np.zeros(out_features))        

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b

    def backward(self, a_prev: np.ndarray, dC_dz: np.ndarray) -> np.ndarray:
        dC_dW = self.dC_dW(a_prev, dC_dz)
        dC_db = self.dC_db(dC_dz)
        dC_da_prev = self.dC_da_prev(dC_dz)

        self.W.grad += dC_dW
        self.b.grad += dC_db
        return dC_da_prev

    # z = a_prev * w + b

    # dz/dW = a_prev
    # dC/dW = dz/dW * dC/dz
    def dC_dW(self, a_prev: np.ndarray, dC_dz: np.ndarray) -> np.ndarray:
        return a_prev.T @ dC_dz

    # dz/db = 1
    # dC/db = dz/db * dC/dz
    def dC_db(self, dC_dz: np.ndarray) -> np.ndarray:
        return dC_dz.sum(axis=0)

    # dz/da_prev = w
    # dC/da_prev = dz/da_prev * dC/dz
    def dC_da_prev(self, dC_dz: np.ndarray) -> np.ndarray:
        return dC_dz @ self.W.data.T

    def parameters(self):
        return [self.W, self.b]

    def __repr__(self):
        return f"Linear({self.in_features} -> {self.out_features})"