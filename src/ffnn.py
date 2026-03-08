import numpy as np
from tensor import Tensor
from layer import Linear
from activation import Linear as LinearAct, ReLU, Sigmoid, Tanh, Softmax
from loss import MSE, BinaryCrossEntropy, CategoricalCrossEntropy

ACTIVATIONS = {
    "linear":  LinearAct,
    "relu":    ReLU,
    "sigmoid": Sigmoid,
    "tanh":    Tanh,
    "softmax": Softmax,
}

LOSSES = {
    "mse": MSE,
    "bce": BinaryCrossEntropy,
    "cce": CategoricalCrossEntropy,
}

class FFNN:

    def __init__(self, layer_sizes: list, activations: list, loss: str = None, init: str = None, seed: int = None, **init_kwargs):

        assert len(activations) == len(layer_sizes) - 1, \
            "Atleast one activation per layer transition mpruy."

        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer_seed = None if seed is None else seed + i
            self.layers.append(
                Linear(layer_sizes[i], layer_sizes[i + 1],
                       init=init, seed=layer_seed, **init_kwargs)
            )

        self.activations = [ACTIVATIONS[a]() for a in activations]

        self.loss_fn = LOSSES[loss]()

        self.parameters = []
        for layer in self.layers:
            self.parameters.extend(layer.parameters())

    def forward(self, x: np.ndarray) -> Tensor:
        if isinstance(x, Tensor):
            out = x
        else:
            out = Tensor(x, requires_grad=False)
            
        for layer, activation in zip(self.layers, self.activations):
            out = layer.forward(out)   
            out = activation(out)      
        return out

    def __repr__(self): #debug
        lines = ["FFNN("]
        for i, (layer, act) in enumerate(zip(self.layers, self.activations)):
            lines.append(f"  ({i}) {layer} -> {act.__class__.__name__}")
        lines.append(f"  loss={self.loss_fn.__class__.__name__}")
        lines.append(")")
        return "\n".join(lines)