"""Script to compute the principal components PC of a matrix."""


import jax
import jax.numpy as jnp
import jax.random as random
from jax.numpy.linalg import svd
from matplotlib import pyplot as plt

jax.config.update("jax_enable_x64", True)


def pca(X_train):
    """Compute the principle components of the dataset X_train.

    Limitations:
        - Currently only for datasets with fewer examples than features.

    Args:
        X_train: The training data of shape (n, p). Here, n is the number of training
            samples and p is the feature dimension. 
    
    Returns:
        A tuple of functions and the singular values. The first functions transforms
            new data to the latent space and the second reverses this.

    Raises:
        ValueError: If the number of training samples is larger than the number of 
            features.
    """
    if X_train.shape[0] > X_train.shape[1]:
        raise ValueError(
            f"Unsupported shape. The data dimension needs to smaller than the feature "
            f"dimension, got {X_train.shape[0]} for the data dimension and "
            f"{X_train.shape[1]} for the feature dimension."
        )

    # Compute covariance matrix in sample space (n, n) and convert to double precision
    XXt = (X_train @ X_train.transpose()).astype(jnp.float64)

    # Eigendecomposition of XXt to obtain eigenvectors, executed in double precision
    U, SS_T, _ = svd(XXt)

    # Convert back to single precision
    U = U.astype(jnp.float32)
    SS_T = SS_T.astype(jnp.float32)

    # S of shape (n, n)
    singular_values = jnp.sqrt(SS_T)
    S_inv = jnp.diag(singular_values ** (-1))

    # W of shape (p, n)
    W = X_train.transpose() @ U @ S_inv

    def transform(X):
        """Transforms data into latent space.

        Args:
            X: needs to be a tensor of shape (batch, p) where p is the dimension 
                of the feature space.

        Returns:
            A tensor of shape (batch, n), the data in latent space. 
        """
        return X @ W

    def recover(T):
        """Transforms latent data back to the original space.

        Args:
            T: a tensor of shape (batch, n) representing data in latent space.
        
        Returns:
            The data in the original space, a tensor of shape (batch, p), where
            p is the feature dimension.
        """
        return T @ W.transpose()

    return transform, recover, singular_values


if __name__ == "__main__":

    # Set the shape
    n, h, w, d = 3, 2, 2, 1
    # n, h, w, d = 250, 128, 128, 82
    shape = (n, h * w * d)
    n, p = shape

    # Create a random key
    key = random.PRNGKey(0)

    # Generate a random tensor with values from a uniform distribution
    X_raw = random.normal(key, shape, dtype=jnp.float32)

    # Print the memory consuption of X_raw
    print(f"Memory size of JAX array: {X_raw.nbytes/1_000_000_000} gigabytes")

    # Compute the PCA associated transformations
    transform, recover,singular_values = pca(X_train=X_raw)

    # demonstrate that they work
    T = transform(X_raw)
    X_recovered = recover(T)
    print(X_recovered - X_raw)

    plt.plot(singular_values)
    plt.show()