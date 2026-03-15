import numpy as np


class BackwardRules:

    @staticmethod
    def dC_da(loss_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n = y_true.shape[0]
        if loss_name == "mse":
            # 2(y_hat - y)/n
            return 2 * (y_pred - y_true) / n
        if loss_name == "bce":
            # (y_hat - y)/(y_hat(1-y_hat)n)
            eps = 1e-12
            y_pred = np.clip(y_pred, eps, 1.0 - eps)
            return (y_pred - y_true) / (y_pred * (1 - y_pred) * n)
        if loss_name == "cce":
            # (y_hat - y)/n (softmax + CCE combined)
            return (y_pred - y_true) / n

        raise ValueError(f"Unknown loss for backward rule: '{loss_name}'")

    @staticmethod
    def da_dz(activation_name: str, z: np.ndarray, y_hat: np.ndarray = None) -> np.ndarray:
        if activation_name == "linear":
            # da/dz = 1
            return np.ones_like(z)
        if activation_name == "relu":
            # da/dz = 1(z > 0)
            return (z > 0).astype(np.float64)
        if activation_name == "sigmoid":
            # da/dz = s * (1 - s), where s = sigmoid(z)
            s = 1.0 / (1.0 + np.exp(-z))
            return s * (1-s)
        if activation_name == "tanh":
            # da/dz = 1 - tanh(z)^2
            t = np.tanh(z)
            return 1.0 - t * t
        if activation_name == "softmax":
            # kalau CCE, combined shortcut gradient udah di-handle dC_da
            return np.ones_like(z)

        raise ValueError(f"Unknown activation for backward rule: '{activation_name}'")
