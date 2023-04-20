"""
Author: Sophia Sanborn, Sagnik Bhattacharya
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: github.com/sophiaas, github.com/sagnibak
"""

import numpy as np
from abc import ABC, abstractmethod

from neural_networks.activations import initialize_activation
from neural_networks.weights import initialize_weights
from collections import OrderedDict

from typing import Callable, List, Literal, Tuple, Union

from .utils import convolution as conv


class Layer(ABC):
    """Abstract class defining the `Layer` interface."""

    def __init__(self):
        self.activation = None

        self.n_in = None
        self.n_out = None

        self.parameters = {}
        self.cache = {}
        self.gradients = {}

        super().__init__()

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        pass

    def clear_gradients(self) -> None:
        self.cache = OrderedDict({a: [] for a, b in self.cache.items()})
        self.gradients = OrderedDict(
            {a: np.zeros_like(b) for a, b in self.gradients.items()}
        )

    def forward_with_param(
        self, param_name: str, X: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Call the `forward` method but with `param_name` as the variable with
        value `param_val`, and keep `X` fixed.
        """

        def inner_forward(param_val: np.ndarray) -> np.ndarray:
            self.parameters[param_name] = param_val
            return self.forward(X)

        return inner_forward

    def _get_parameters(self) -> List[np.ndarray]:
        return [b for a, b in self.parameters.items()]

    def _get_cache(self) -> List[np.ndarray]:
        return [b for a, b in self.cache.items()]

    def _get_gradients(self) -> List[np.ndarray]:
        return [b for a, b in self.gradients.items()]


def initialize_layer(
    name: str,
    activation: str = None,
    weight_init: str = None,
    n_out: int = None,
    kernel_shape: Tuple[int, int] = None,
    stride: int = None,
    pad: int = None,
    mode: str = None,
    keep_dim: str = "first",
) -> Layer:
    """Factory function for layers."""
    if name == "fully_connected":
        return FullyConnected(
            n_out=n_out, activation=activation, weight_init=weight_init,
        )

    elif name == "conv2d":
        return Conv2D(
            n_out=n_out,
            activation=activation,
            kernel_shape=kernel_shape,
            stride=stride,
            pad=pad,
            weight_init=weight_init,
        )

    elif name == "pool2d":
        return Pool2D(kernel_shape=kernel_shape, mode=mode, stride=stride, pad=pad)

    elif name == "flatten":
        return Flatten(keep_dim=keep_dim)

    else:
        raise NotImplementedError("Layer type {} is not implemented".format(name))


class FullyConnected(Layer):
    """A fully-connected layer multiplies its input by a weight matrix, adds
    a bias, and then applies an activation function.
    """

    def __init__(
        self, n_out: int, activation: str, weight_init="xavier_uniform"
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)

        # instantiate the weight initializer
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int]) -> None:
        """Initialize all layer parameters (weights, biases)."""
        self.n_in = X_shape[1]

        ### BEGIN YOUR CODE ###

        W = self.init_weights((self.n_in, self.n_out))
        
        b = np.zeros((1, self.n_out))


        self.parameters = OrderedDict({"W": W, "b": b})
        self.cache: OrderedDict = OrderedDict({"Z": [], "X": []}) # cache for backprop
        self.gradients: OrderedDict = OrderedDict({"W": np.zeros([W.shape[0], W.shape[1]]), "b": np.zeros([b.shape[0], b.shape[1]])})  # parameter gradients initialized to zero
                                           # MUST HAVE THE SAME KEYS AS `self.parameters`

        ### END YOUR CODE ###

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: multiply by a weight matrix, add a bias, apply activation.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        Parameters
        ----------
        X  input matrix of shape (batch_size, input_dim)

        Returns
        -------
        a matrix of shape (batch_size, output_dim)
        """
        # initialize layer parameters if they have not been initialized
        if self.n_in is None:
            self._init_parameters(X.shape)

        ### BEGIN YOUR CODE ###
        
        # perform an affine transformation and activation
        out = X @ self.parameters["W"] + self.parameters["b"]
       
        
        # store information necessary for backprop in `self.cache`
        self.cache["Z"] = out
        self.cache["X"] = X

        # activation after caching

        Z = self.activation(out)

        ### END YOUR CODE ###

        return Z

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for fully connected layer.
        Compute the gradients of the loss with respect to:
            1. the weights of this layer (mutate the `gradients` dictionary)
            2. the bias of this layer (mutate the `gradients` dictionary)
            3. the input of this layer (return this)

        Parameters
        ----------
        dLdY  derivative of the loss with respect to the output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        derivative of the loss with respect to the input of this layer
        shape (batch_size, input_dim)
        """
        ### BEGIN YOUR CODE ###
        
        # unpack the cache

        W = self.parameters["W"]
        b = self.parameters["b"]
        X = self.cache["X"]
        Z = self.cache["Z"]
        
        # compute the gradients of the loss w.r.t. all parameters as well as the
        # input of the layer
        

        dLdZ = self.activation.backward(Z, dLdY)
        dX = dLdZ @ W.T
        # dW = dLdZ.T @ X
        dW = X.T @ dLdZ
        db = np.sum(dLdZ, axis=0, keepdims=True)

        #print("\n================================================")
        #print("dLdY", dLdY.shape, "W", W.shape, "b", b.shape, "X", X.shape, "Z", Z.shape, "dLdZ", dLdZ.shape, "dX", dX.shape, "dW", dW.shape, "db", db.shape)
        #print("================================================")

        self.gradients["W"] = dW
        self.gradients["b"] = db

        # store the gradients in `self.gradients`
        # the gradient for self.parameters["W"] should be stored in
        # self.gradients["W"], etc.

        #print("SIZE", dY.shape, dX.shape)

        ### END YOUR CODE ###

        return dX


class Conv2D(Layer):
    """Convolutional layer for inputs with 2 spatial dimensions."""

    def __init__(
        self,
        n_out: int,
        kernel_shape: Tuple[int, int],
        activation: str,
        stride: int = 1,
        pad: str = "same",
        weight_init: str = "xavier_uniform",
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = pad

        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int, int, int]) -> None:
        """Initialize all layer parameters and determine padding."""
        self.n_in = X_shape[3]

        W_shape = self.kernel_shape + (self.n_in,) + (self.n_out,)
        W = self.init_weights(W_shape)
        b = np.zeros((1, self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b})
        self.cache = OrderedDict({"Z": [], "X": []})
        self.gradients = OrderedDict({"W": np.zeros_like(W), "b": np.zeros_like(b)})

        if self.pad == "same":
            self.pad = ((W_shape[0] - 1) // 2, (W_shape[1] - 1) // 2)
        elif self.pad == "valid":
            self.pad = (0, 0)
        elif isinstance(self.pad, int):
            self.pad = (self.pad, self.pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for convolutional layer. This layer convolves the input
        `X` with a filter of weights, adds a bias term, and applies an activation
        function to compute the output. This layer also supports padding and
        integer strides. Intermediates necessary for the backward pass are stored
        in the cache.

        Parameters
        ----------
        X  input with shape (batch_size, in_rows, in_cols, in_channels)

        Returns
        -------
        output feature maps with shape (batch_size, out_rows, out_cols, out_channels)
        """
        if self.n_in is None:
            self._init_parameters(X.shape)

        W = self.parameters["W"]
        b = self.parameters["b"]

        # DIMENSION INITIALIZATION
        # in_channels = self.n_in, out_channels = self.n_out
        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_shape = (kernel_height, kernel_width)

        # implement a convolutional forward pass

        out_rows = int(((in_rows - kernel_height + 2 * self.pad[0]) / self.stride)) + 1
        out_cols = int(((in_cols - kernel_width + 2 * self.pad[1]) / self.stride)) + 1

        output = np.zeros([n_examples, out_rows, out_cols, out_channels])

        padded_X, _ = conv.pad2d(X, self.pad, self.stride, kernel_shape)

        # for each example
        for i in range(n_examples):
            for row in range(out_rows):
                for col in range(out_cols):
                    # start and end col index
                    start_col = col * self.stride
                    end_col = start_col + kernel_width
                    # start and end row index
                    start_row = row * self.stride
                    end_row = start_row + kernel_height
                    # X window
                    X_window = padded_X[i, start_row:end_row, start_col:end_col, :]
                    #print("\n====================================")
                    #print("WINDOW", X_window.shape, "STRIDE:", self.stride, "START/END", start_col, "/", end_col)
                    #print(X_window)
                    #print("====================================")
                    for f in range(out_channels):
                        #TODO: is W the right filter array?
                        filter = W[:, :, :, f]
                        #print("\n====================================")
                        #try:
                        #    print("WINDOW * FILTER:", (X_window * filter).shape, "NP.SUM", np.sum(X_window * filter).shape, "b", b.shape)
                        #except:
                        #    print(X_window.shape)
                        #print("====================================")
                        result = np.sum(X_window * filter) + b[:, f]
                        output[i, row, col, f] = result

        # cache any values required for backprop
        
        self.cache["Z"] = output
        self.cache["X"] = X

         ### END YOUR CODE ###

        return self.activation(output)


    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for conv layer. Computes the gradients of the output
        with respect to the input feature maps as well as the filter weights and
        biases.

        Parameters
        ----------
        dLdY  derivative of loss with respect to output of this layer
              shape (batch_size, out_rows, out_cols, out_channels)

        Returns
        -------
        derivative of the loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, in_channels)
        """
        ### BEGIN YOUR CODE ###

        # Load Cache
        X = self.cache["X"]
        Z = self.cache["Z"]
        W = self.parameters["W"]
        b = self.parameters["b"]


        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_height, kernel_width, in_channels, out_channels = W.shape
        kernel_shape = (kernel_height, kernel_width)
        _, out_rows, out_cols, _ = Z.shape

       
        # dLdY ~ (16, 16, 16, 32)
        # 16 - batch size, 16 - out_rows, 16 - out_cols, 32 - out_channels
        # backwards activation
        dZ = self.activation.backward(Z, dLdY)        

        dB = dZ.sum(axis=(0, 1, 2)).reshape(1, -1)
        dW = np.zeros(W.shape)

        print("\n===============================")
        print("dLdY", dLdY.shape, "dZ", dZ.shape, "Z", Z.shape, "W", W.shape, "X", X.shape, "dB", dB.shape, "b", b.shape, "dW", dW.shape)
        print("=================================")
            

        # Dilate each dLdY by stride
        if self.stride != 1:
            for i in range(1, self.stride):
                dLdY = np.insert(dLdY, range(1, dLdY.shape[1], i), 0, axis=1)
                dLdY = np.insert(dLdY, range(1, dLdY.shape[2], i), 0, axis=2)
        
        # Pad dLdY
        padded_dLdY = np.pad(X, [(0,0), (kernel_height - 1, kernel_height - 1), (kernel_width - 1, kernel_width - 1), (0,0)])

        # Flip W
        W_flip = np.flip(W, axis=0)
        W_flip = np.flip(W_flip, axis=1)

        output = np.zeros([n_examples, in_rows, in_cols, out_channels])

        dW = np.zeros(W.shape)

        # dLdX ~ (16, 3
        for i in range(n_examples):
            for row in range(in_rows):
                for col in range(in_cols):
                    # start and end col index
                    start_col = col
                    end_col = start_col + kernel_width
                    # start and end row index
                    start_row = row * self.stride
                    end_row = start_row + kernel_height
                    # X window
                    window = padded_dLdY[i, start_row:end_row, start_col:end_col, :]
                    
                    for f in range(out_channels):
                        #TODO: is W the right filter array?
                        filter = W_flip[:, :, :, f]
                        result = np.sum(window * filter)
                        output[i, row, col, f] = result

   
        
        
        
        
        self.gradients["W"] = output
        self.gradients["b"] = dB

        return output

        #print("\n===============================")
        #print("dLdY", dLdY.shape, "stride", self.stride, "padded_dLdY", padded_dLdY.shape, "padding", self.pad)
        #print("=================================")


        return dX

class Pool2D(Layer):
    """Pooling layer, implements max and average pooling."""

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        mode: str = "max",
        stride: int = 1,
        pad: Union[int, Literal["same"], Literal["valid"]] = 0,
    ) -> None:

        if type(kernel_shape) == int:
            kernel_shape = (kernel_shape, kernel_shape)

        self.kernel_shape = kernel_shape
        self.stride = stride

        if pad == "same":
            self.pad = ((kernel_shape[0] - 1) // 2, (kernel_shape[1] - 1) // 2)
        elif pad == "valid":
            self.pad = (0, 0)
        elif isinstance(pad, int):
            self.pad = (pad, pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

        self.mode = mode

        if mode == "max":
            self.pool_fn = np.max
            self.arg_pool_fn = np.argmax
        elif mode == "average":
            self.pool_fn = np.mean

        self.cache = {
            "out_rows": [],
            "out_cols": [],
            "X_pad": [],
            "p": [],
            "pool_shape": [],
        }
        self.parameters = {}
        self.gradients = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: use the pooling function to aggregate local information
        in the input. This layer typically reduces the spatial dimensionality of
        the input while keeping the number of feature maps the same.

        As with all other layers, please make sure to cache the appropriate
        information for the backward pass.

        Parameters
        ----------
        X  input array of shape (batch_size, in_rows, in_cols, channels)

        Returns
        -------
        pooled array of shape (batch_size, out_rows, out_cols, channels)
        """
        ### BEGIN YOUR CODE ###

        # implement the forward pass

        # cache any values required for backprop

        ### END YOUR CODE ###

        return X_pool

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for pooling layer.

        Parameters
        ----------
        dLdY  gradient of loss with respect to the output of this layer
              shape (batch_size, out_rows, out_cols, channels)

        Returns
        -------
        gradient of loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, channels)
        """
        ### BEGIN YOUR CODE ###

        # perform a backward pass

        ### END YOUR CODE ###

        return dX

class Flatten(Layer):
    """Flatten the input array."""

    def __init__(self, keep_dim: str = "first") -> None:
        super().__init__()

        self.keep_dim = keep_dim
        self._init_params()

    def _init_params(self):
        self.X = []
        self.gradients = {}
        self.parameters = {}
        self.cache = {"in_dims": []}

    def forward(self, X: np.ndarray, retain_derived: bool = True) -> np.ndarray:
        self.cache["in_dims"] = X.shape

        if self.keep_dim == -1:
            return X.flatten().reshape(1, -1)

        rs = (X.shape[0], -1) if self.keep_dim == "first" else (-1, X.shape[-1])
        return X.reshape(*rs)

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        in_dims = self.cache["in_dims"]
        dX = dLdY.reshape(in_dims)
        return dX