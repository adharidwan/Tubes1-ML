import numpy as np

class Linear:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def derivate(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
    
class ReLU:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0,x)
    
    def derivate(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, 0.0)
    
class Sigmoid:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def derivate(self, x:np.ndarray) -> np.ndarray:
        sigmoid = self(x)
        return sigmoid * (1 - sigmoid)

class Tanh:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def derivate(self, x:np.ndarray) -> np.ndarray:
        t = np.tanh(x)
        return 1 - t**2
    
class Softmax:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)