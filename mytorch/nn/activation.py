import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        shifted_Z = Z - np.max(Z, axis=self.dim, keepdims=True)
        exp_Z = np.exp(shifted_Z)
        
        self.A = exp_Z / np.sum(exp_Z, axis=self.dim, keepdims=True)
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        dim = self.dim
        A_moved = np.moveaxis(self.A, dim, -1)
        dLdA_moved = np.moveaxis(dLdA, dim, -1)
        original_moved_shape = A_moved.shape
        
        C = A_moved.shape[-1]
        A_2d = A_moved.reshape(-1, C)
        dLdA_2d = dLdA_moved.reshape(-1, C)
        
        dLdZ_2d = A_2d * dLdA_2d - A_2d * np.sum(dLdA_2d * A_2d, axis=-1, keepdims=True)
        dLdZ_moved = dLdZ_2d.reshape(original_moved_shape)
        dLdZ = np.moveaxis(dLdZ_moved, -1, dim)
        
        return dLdZ
 

    