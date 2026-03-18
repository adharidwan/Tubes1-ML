import json

import matplotlib.pyplot as plt
import numpy as np

from tensor import Tensor
from layer import Linear
from activation import Linear as LinearAct, ReLU, Sigmoid, Tanh, Softmax
from loss import MSE, BinaryCrossEntropy, CategoricalCrossEntropy
from backward import BackwardRules

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

    def __init__(self, layer_sizes: list, activations: list, loss: str = None, init: str = None, seed: int = None, regularization: str = None, lambda_reg: float = 0.01, backprop_mode: str = "autograd", **init_kwargs):

        assert len(activations) == len(layer_sizes) - 1, \
            "Atleast one activation per layer transition mpruy."
        if loss not in LOSSES:
            raise ValueError(f"Unknown loss: '{loss}'. Available losses: {list(LOSSES.keys())}")

        unknown_activations = [a for a in activations if a not in ACTIVATIONS]
        if unknown_activations:
            raise ValueError(
                f"Unknown activations: {unknown_activations}. "
                f"Available activations: {list(ACTIVATIONS.keys())}"
            )

        if regularization not in {None, "l1", "l2"}:
            raise ValueError("regularization must be one of: None, 'l1', 'l2'.")

        self.layer_sizes = list(layer_sizes)
        self.activation_names = list(activations)
        self.loss_name = loss
        self.init = init
        self.seed = seed
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.backprop_mode = backprop_mode
        self.init_kwargs = dict(init_kwargs)

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

        self.rng = np.random.default_rng(seed)

    def forward(self, x: np.ndarray) -> Tensor:
        if isinstance(x, Tensor):
            out = x
        else:
            out = Tensor(x, requires_grad=False)
            
        self._cache = {"inputs": [], "pre_acts": []}
        for layer, activation in zip(self.layers, self.activations):
            self._cache["inputs"].append(out.data.copy())
            z = layer.forward(out)
            self._cache["pre_acts"].append(z.data.copy())
            out = activation(z)
        return out

    def __repr__(self): #debug
        lines = ["FFNN("]
        for i, (layer, act) in enumerate(zip(self.layers, self.activations)):
            lines.append(f"  ({i}) {layer} -> {act.__class__.__name__}")
        lines.append(f"  loss={self.loss_fn.__class__.__name__}")
        lines.append(")")
        return "\n".join(lines)

    def plot_weight_distribution(self, layers: list[int]):
        i = 0
        for i in range(len(self.layers)):
            if i in layers:
                plt.hist(self.layers[i].W.data.flatten(), bins='auto') # 'auto' chooses an optimal number of bins
                plt.title("Weight Distribution in Layer " + str(i+1))
                plt.xlabel("Weight")
                plt.ylabel("Frequency")
                plt.tight_layout()
                plt.show()

    def plot_grad_distribution(self, layers: list[int]):
        i = 0
        for i in range(len(self.layers)):
            if i in layers:
                plt.hist(self.layers[i].W.grad.flatten(), bins='auto') # 'auto' chooses an optimal number of bins
                plt.title("Gradient Distribution in Layer " + str(i+1))
                plt.xlabel("Gradient")
                plt.ylabel("Frequency")
                plt.tight_layout()      
                plt.show()

    def save(self, path):
        metadata = {
            "layer_sizes": self.layer_sizes,
            "activations": self.activation_names,
            "loss": self.loss_name,
            "init": self.init,
            "seed": self.seed,
            "regularization": self.regularization,
            "lambda_reg": self.lambda_reg,
            "backprop_mode": self.backprop_mode,
            "init_kwargs": self.init_kwargs,
        }

        arrays = {"metadata": np.array(json.dumps(metadata), dtype=object)}
        for idx, layer in enumerate(self.layers):
            arrays[f"W_{idx}"] = layer.W.data
            arrays[f"b_{idx}"] = layer.b.data

        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        checkpoint = np.load(path, allow_pickle=True)
        metadata = json.loads(str(checkpoint["metadata"].item()))

        model = cls(
            layer_sizes=metadata["layer_sizes"],
            activations=metadata["activations"],
            loss=metadata["loss"],
            init=metadata["init"],
            seed=metadata["seed"],
            regularization=metadata["regularization"],
            lambda_reg=metadata["lambda_reg"],
            backprop_mode=metadata.get("backprop_mode", "autograd"),
            **metadata["init_kwargs"],
        )

        for idx, layer in enumerate(model.layers):
            layer.W.data = checkpoint[f"W_{idx}"].copy()
            layer.b.data = checkpoint[f"b_{idx}"].copy()

        return model

    def _step(self, lr: float):
        for param in self.parameters:
            param.data -= lr * param.grad

    def _prepare_inputs(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, n_features).")
        if X.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"Expected {self.layer_sizes[0]} input features, got {X.shape[1]}."
            )
        return X

    def _prepare_targets(self, y):
        y = np.asarray(y, dtype=np.float64)
        output_dim = self.layer_sizes[-1]

        if y.ndim == 1:
            if output_dim == 1:
                y = y.reshape(-1, 1)
            elif self.loss_name == "cce":
                class_indices = y.astype(int)
                if np.any(class_indices < 0) or np.any(class_indices >= output_dim):
                    raise ValueError("Class labels are out of range for the output dimension.")
                y = np.eye(output_dim, dtype=np.float64)[class_indices]
            else:
                raise ValueError(
                    "1D targets are only supported for single-output models or categorical cross entropy."
                )

        if y.ndim != 2:
            raise ValueError("y must be a 1D or 2D array.")
        if y.shape[1] != output_dim:
            raise ValueError(
                f"Expected target dimension {output_dim}, got {y.shape[1]}."
            )
        return y

    def _compute_total_loss(self, y_true, y_pred):
        targets = Tensor(y_true, requires_grad=False)
        return self.loss_fn(targets, y_pred) + self.regularization_loss()

    def regularization_loss(self):
        if self.regularization is None:
            return Tensor(0.0, requires_grad=False)
        reg = None
        for layer in self.layers:
            if self.regularization == "l1":
                term = layer.W.abs().sum()
            else: # l2
                term = (layer.W ** 2).sum()
            reg = term if reg is None else reg + term
        return reg * self.lambda_reg

    def _backward(self, loss, y_true: np.ndarray, y_pred) -> None:
        if self.backprop_mode == "autograd":
            loss.backward()
        else:
            self._manual_backward(y_true, y_pred.numpy())
            self._add_regularization_grads()

    def _manual_backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        dC_da = BackwardRules.dC_da(self.loss_name, y_true, y_pred)

        for l in reversed(range(len(self.layers))):
            dC_dz = self._compute_dC_dz(layer_idx=l, dC_da=dC_da, y_pred=y_pred)
            dC_da = self._backward_one_layer(layer_idx=l, dC_dz=dC_dz)

    def _compute_dC_dz(self, layer_idx: int, dC_da: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        z = self._cache["pre_acts"][layer_idx]
        da_dz = BackwardRules.da_dz(
            self.activation_names[layer_idx], z, y_pred
        )
        return dC_da * da_dz
    
    def _backward_one_layer(self, layer_idx: int, dC_dz: np.ndarray) -> np.ndarray:
        a_in = self._cache["inputs"][layer_idx]
        return self.layers[layer_idx].backward(a_in, dC_dz)

    def _add_regularization_grads(self) -> None:
        if self.regularization is None:
            return

        for layer in self.layers:
            if self.regularization == "l1":
                layer.W.grad += self.lambda_reg * np.sign(layer.W.data)
            else:  # l2
                layer.W.grad += self.lambda_reg * 2.0 * layer.W.data

    def _compute_val_loss(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        y_pred = self.forward(X_val)
        return self._compute_total_loss(y_val, y_pred).item()

    def fit(self, X, y, batch_size=32, lr=0.01, epochs=100, verbose=1, validation_data=None):
        X = self._prepare_inputs(X)
        y = self._prepare_targets(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if lr <= 0:
            raise ValueError("lr must be positive.")
        if epochs <= 0:
            raise ValueError("epochs must be positive.")

        X_val, y_val = None, None
        if validation_data is not None:
            X_val = self._prepare_inputs(validation_data[0])
            y_val = self._prepare_targets(validation_data[1])

        history = {"train_loss": [], "val_loss": []}

        n_samples = X.shape[0]
        effective_batch_size = min(batch_size, n_samples)
        n_batches = int(np.ceil(n_samples / effective_batch_size))

        for epoch in range(epochs):
            indices = self.rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            batch_losses = []

            for batch_idx, start in enumerate(range(0, n_samples, effective_batch_size)):
                end = start + effective_batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                for param in self.parameters:
                    param.zero_grad()
                predictions = self.forward(X_batch)
                loss = self._compute_total_loss(y_batch, predictions)
                self._backward(loss, y_batch, predictions)
                self._step(lr)
                batch_losses.append(loss.item())

                if verbose:
                    self._print_progress(
                        epoch, epochs, batch_idx + 1, n_batches,
                        train_loss=float(np.mean(batch_losses)),
                        val_loss=None,
                        done=False,
                    )

            epoch_loss = float(np.mean(batch_losses))
            history["train_loss"].append(epoch_loss)

            val_loss = None
            if X_val is not None:
                val_loss = float(self._compute_val_loss(X_val, y_val))
            history["val_loss"].append(val_loss)

            if verbose:
                self._print_progress(
                    epoch, epochs, n_batches, n_batches,
                    train_loss=epoch_loss,
                    val_loss=val_loss,
                    done=True,
                )

        return history

    @staticmethod
    def _print_progress(epoch, epochs, batch, n_batches, train_loss, val_loss, done):
        bar_width = 30
        filled = int(bar_width * batch / n_batches)
        bar = "=" * filled + (">" if filled < bar_width else "") + "." * (bar_width - filled - (1 if filled < bar_width else 0))
        suffix = f"train_loss: {train_loss:.6f}"
        if val_loss is not None:
            suffix += f" - val_loss: {val_loss:.6f}"
        line = f"\rEpoch {epoch + 1}/{epochs} [{bar}] {batch}/{n_batches} - {suffix}"
        end = "\n" if done else ""
        print(line, end=end, flush=True)

    def predict(self, X, return_proba=False):
        X = self._prepare_inputs(X)
        proba = self.forward(X).numpy()

        if return_proba:
            return proba

        if self.loss_name == "mse":
            return proba
        elif self.loss_name == "bce":
            return (proba >= 0.5).astype(int).flatten()
        else: # cce
            return np.argmax(proba, axis=-1)