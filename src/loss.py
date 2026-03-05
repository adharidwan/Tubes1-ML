import numpy as np

class MSE:
    # L = (1/n) * Σ (y_i - ŷ_i)^2

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # dL/dŷ = (2/n)(ŷ - y)
        n = y_true.shape[0]
        return (2 / n) * (y_pred - y_true)


class BinaryCrossEntropy:
    # L = -(1/n) * Σ [ y_i log(ŷ_i) + (1 - y_i) log(1 - ŷ_i) ]

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)

        return -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # dL/dŷ = (ŷ - y) / (ŷ(1-ŷ) * n)
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)

        n = y_true.shape[0]

        return (y_pred - y_true) / (y_pred * (1 - y_pred) * n)


class CategoricalCrossEntropy:
    # L = -(1/n) * Σ Σ (y_ij log(ŷ_ij))

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)

        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # dL/dŷ = -(y / ŷ) / n
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)

        n = y_true.shape[0]

        return -y_true / y_pred / n