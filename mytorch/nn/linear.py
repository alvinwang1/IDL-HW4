import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # Store original shape and input A
        self.input_shape = A.shape
        self.A = A
        
        batch_size = int(np.prod(self.input_shape[:-1])) if len(self.input_shape[:-1]) > 0 else 1
        in_features = self.input_shape[-1]
        A_2d = A.reshape(batch_size, in_features)
        Z_2d = A_2d @ self.W.T + self.b
        out_shape = tuple(self.input_shape[:-1]) + (self.W.shape[0],)
        Z = Z_2d.reshape(out_shape)
        
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # Compute gradients
        batch_size = int(np.prod(self.input_shape[:-1])) if len(self.input_shape[:-1]) > 0 else 1
        out_features = self.W.shape[0]
        in_features = self.W.shape[1]
        
        dLdZ_2d = dLdZ.reshape(batch_size, out_features)
        A_2d = self.A.reshape(batch_size, in_features)
        self.dLdW = dLdZ_2d.T @ A_2d
        self.dLdb = np.sum(dLdZ_2d, axis=0)
        dLdA_2d = dLdZ_2d @ self.W
        self.dLdA = dLdA_2d.reshape(self.input_shape)
        
        return self.dLdA
