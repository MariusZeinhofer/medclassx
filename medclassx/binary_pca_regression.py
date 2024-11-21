"""Script to compute binary (two-class) PCA regression."""

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax.numpy.linalg import svd
from matplotlib import pyplot as plt
from medclassx.pca import pca
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

jax.config.update("jax_enable_x64", True)


def binary_pca_regression(X_patient, X_control, cut_off=2):
    """PCA on combined data and separate classes in pca space with logistic regression.

    Limitations:
        - Currently only for datasets with fewer examples than features, as it calls
        the pca method that has this limitation.

    Args:
        X_patient: The patient data of shape (n_pat, p). Here, n_pat is the number of
            patient samples and p is the feature dimension.

        X_control: The control data of shape (n_cont, p). Here, n_cont is the number of
            control samples and p is the feature dimension.

        cut_off: The number of principle coponents to consider. Default is to use 5.

    Returns:
        A list of dictionaries and the latents T. Each dictionary corresponds to one 
            principle component with associated principle component number (pc_number),
            principle component (pc), singular value (sv), variance accounted for
            (vaf), and class separation value (p value of suitable t-test).

    Raises:
        ValueError: If the number of training samples is larger than the number of
            features.
    """
    if not X_patient.shape[1] == X_control.shape[1]:
        raise ValueError(
            f"Patients and control shapes do not match. Got patient voxel dimension: "
            f"{X_patient.shape[1]} and control voxel dimension: {X_control.shape[1]}."
        )

    # assemble data matrix
    X = jnp.concatenate((X_control, X_patient), axis=0)  # question: view or copy?

    # generate labels
    y_1 = jnp.zeros(shape=(len(X_control),), dtype=int)
    y_2 = jnp.ones(shape=(len(X_patient),), dtype=int)
    y = jnp.concatenate((y_1, y_2), axis=0)

    # compute the PCA of the data matrix
    _, _, singular_values, W = pca(X)

    # just keep the data up to to the cut off value
    singular_values  # = singular_values[0:cut_off]
    W = W[:, 0:cut_off]

    # compute the latent data up to the cut-off value
    T = X @ W

    results = []
    # iterate over the latent data
    for i in range(0, cut_off):
        # instantiate a logistic regression model
        model = LogisticRegression(max_iter=200)

        # fit it using the i-th column of T
        model.fit(T[:, i : i + 1], y)

        # prediction on train data
        y_pred = model.predict(T[:, i : i + 1])

        # compute accuracy
        accuracy = accuracy_score(y, y_pred)

        # compute variance accounted for by the i-th pc
        vaf = singular_values[i] / sum(singular_values)

        # z-score normalized principle component
        pcz = (W[:, i] - jnp.mean(W[:, i]))/jnp.std(W[:, i])

        # collect data
        results.append(
            {
                "pc_number": i,
                "sv": singular_values[i],
                "vaf": vaf,
                "accuracy": accuracy,
                "pc": W[:, i],
                "pcz": W[:, i],
                #"pcz": pcz,
            }
        )

    return results, T


if __name__ == "__main__":
    # Set the shape
    n_pat, h, w, d = 3, 2, 2, 2
    shape_pat = (n_pat, h * w * d)

    n_con, h, w, d = 2, 2, 2, 2
    shape_con = (n_pat, h * w * d)

    p = shape_pat[1]

    # Create a random key
    key_pat = random.PRNGKey(0)
    key_con = random.PRNGKey(1)

    # Generate a random patient and control data
    X_pat = random.normal(key_pat, shape_pat, dtype=jnp.float32)
    X_con = random.normal(key_con, shape_con, dtype=jnp.float32)

    results = binary_pca_regression(X_control=X_con, X_patient=X_pat, cut_off=6)

    print(f"results: {results}")
