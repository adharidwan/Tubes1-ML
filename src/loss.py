from tensor import Tensor

class MSE:
    # L = (1/n) * Σ (y_i - ŷ_i)^2
    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return ((y_pred - y_true) ** 2).mean()


class BinaryCrossEntropy:
    # L = -(1/n) * Σ [ y_i log(ŷ_i) + (1 - y_i) log(1 - ŷ_i) ]
    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        y_pred = y_pred.clip(1e-12, 1 - 1e-12)
        return -(y_true * y_pred.log() + (Tensor(1.0, requires_grad=False) - y_true) * (Tensor(1.0, requires_grad=False) - y_pred).log()).mean()


class CategoricalCrossEntropy:
    # L = -(1/n) * Σ Σ (y_ij log(ŷ_ij))
    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        y_pred = y_pred.clip(1e-12, 1.0)
        return -(y_true * y_pred.log()).mean()