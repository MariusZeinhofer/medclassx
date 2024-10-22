"""Based on the compute_pca.py and visualize_pca.py derives the PCA pattern."""

from pathlib import Path

import jax.numpy as jnp
import nibabel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# choose pcs
pcs_idx = [2, 5]

# root data path
root = r"examples\exp_aprinois_2\out"

# the latent data, remember 0 based counting
T = jnp.load(Path(root + r"\latent_data.npy"))[:, [p-1 for p in pcs_idx]]

print(T.shape)

# load the principle components
pcs = [nibabel.load(Path(root + "\zscore_pc_" + f"{pc}" + ".nii")) for pc in pcs_idx]

# convert to a list of numpy arrays
pcs = [pc.get_fdata() for pc in pcs]
print(pcs[0].shape)

# the first 30 patients are healthy controls, the last 30 are PSP diagnosed
y_1 = jnp.zeros(shape=(30,), dtype=int)
y_2 = jnp.ones(shape=(30,), dtype=int)
y = jnp.concatenate((y_1, y_2), axis=0)

# Set up logistic regression
model = LogisticRegression(max_iter=200)

# uses L-BFGS per default
model.fit(T, y)

# Access the trained parameters
weights = model.coef_
intercept = model.intercept_

# Print the weights and intercept
print("Weights (coefficients):")
print(weights)
print(weights.shape)
print("Intercept:")
print(intercept)

# Make predictions
y_pred = model.predict(T)

# Evaluate the model
accuracy = accuracy_score(y, y_pred)
conf_matrix = confusion_matrix(y, y_pred)

# Print the evaluation results
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)

# derive pattern using logistic weights
pattern = sum([weights[0, i] * pcs[i] for i in range(0, len(pcs))])
jnp.save(Path(root + "\pattern.npy"), pattern)