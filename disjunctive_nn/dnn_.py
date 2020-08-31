from functools import reduce

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


sigmoid = torch.nn.Sigmoid()


class Polytope():
    """
    Convex polytope constructed as the intersection of m half-spaces

    Parameters
    ----------
    dim : int
        Input data dimension (number of features)
    m : int
        Number of half-spaces per polytope
    """

    def __init__(self, dim: int, m: int):
        self.dim = dim
        self.m = m
        
        self.cuts = torch.rand([1, m], requires_grad=True)
        self.w = torch.rand([dim, m], requires_grad=True)
        
        self.params = [self.cuts, self.w]

    def value(self, X):
        polytope = torch.matmul(X, self.w) + self.cuts
        polytope = sigmoid(polytope)
        polytope = torch.prod(polytope, axis=1)  # intersection of half-spaces
        return polytope



class DisjunctiveNormalNetwork(BaseEstimator, ClassifierMixin):
    """
    Neural network consisting of one layer of polytopes combined in an OR gate

    Parameters
    ----------
    n_polytopes : int
        Number of polytops
    m : int
        Number of half-spaces per polytope
    """

    def __init__(self, n_polytopes: int, m: int):
        """
        Network initialization

        Parameters
        ----------
        n_polytopes : int
            Number of polytopes
        m : int
            Number of half-spaces per polytope
        """

        self.n_polytopes = n_polytopes
        self.m = m

        self.polytopes = None
        
    
    def forward(self, X):
        """
        Implementation of network's forward pass

        The network's output is 1 in the areas covered by the polytopes and 0 otherwise.
        This is expressed as the sum of boolean variables where each variable corresponds to a polytope.

        y = A + B + C + ...

        Using the DeMorgan rule, this is equivalent to
        y = (A'B'C'...)'

        Expressed numerically:
        y = 1 - (1-A)(1-B)(1-C)...
        """

        result = 1.0
        for cube in self.polytopes:
            result *= 1.0 - cube.value(X)
        
        result = 1 - result
        return torch.stack([1 - result, result]).T
    
    
    def fit(self, X, y, epochs=1000, lr=0.01, batch_size=100, verbose=True):
        """
        Fit Neural Hyoercube Network with Adam optimizer and Cross-Entropy loss

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        
        Returns
        -------
        self : object
            Returns self.
        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y).type(torch.LongTensor)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Initialize hyper-cubes for the first time
        dim = X.shape[1]
        if self.polytopes is None:
            self.polytopes = [Polytope(dim, self.m) for c in range(self.n_polytopes)]

        # Define loss function and optimizer
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            params=list(reduce(lambda a, b: a + b, [cube.params for cube in self.polytopes])),
            lr=lr)
        
        # Run optimizer
        for epoch in range(epochs):
            inputs, labels = X, y

            optimizer.zero_grad()

            outputs = self.forward(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            if verbose:
                if epoch % (epochs // 10) == 0:
                    acc = np.mean(np.argmax(outputs.detach().numpy(), axis=1) == labels.detach().numpy())
                    print(f"Epoch: {epoch}, Accuracy: {acc}")

        return self
    
                    
    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        X = torch.from_numpy(X.astype(np.float32))

        return self.forward(X).detach().numpy()[:, 1]

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        X = torch.from_numpy(X.astype(np.float32))

        return (self.predict_proba(X) >= 0.5).astype(int)
