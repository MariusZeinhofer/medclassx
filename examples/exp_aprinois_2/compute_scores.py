"""Computes the scores of the train set with respect to the derived pattern"""

from pathlib import Path

import jax.numpy as jnp
import nibabel
from matplotlib import pyplot as plt
from medclassx.mask_trafo import mask_vector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# root data path
root = r"examples\exp_aprinois_2\out"

# load pattern
pattern = jnp.load(Path(root + "\pattern.npy"))
a, b, c = pattern.shape

# flatten pattern
pattern = jnp.reshape(pattern, shape=(-1,))
print(pattern.shape)

# mask pattern
mask_path = Path(r"data\aprinois_nuk_data\mask_for_scanvp.nii")
nifti_mask = nibabel.load(mask_path)
mask = nifti_mask.get_fdata()
mask = mask.reshape(-1)
pattern, unmask = mask_vector(pattern, mask)
print(pattern.shape)

# load dataset
X = jnp.load(Path(r"examples\exp_aprinois_2\out\X.npy"))
X_control = X[0:30, :]
X_patient = X[30:,:]

# compute scores
S_control = jnp.array([jnp.sum(pattern * x) for x in X_control])
s_mean = jnp.mean(S_control)
s_std = jnp.std(S_control)
S_control_zscore = (S_control - s_mean) / s_std
print(S_control_zscore)
print("---------")

S_patient = jnp.array([jnp.sum(pattern * x) for x in X_patient])
S_patient_zscore = (S_patient - s_mean) / s_std
print(S_patient_zscore)




# Example data
# Example data
numbers = jnp.arange(1, 61)  # 1 to 10
z_scores_control = S_control_zscore[0:30]
z_scores_patient = S_patient_zscore[0:30]
z_scores = jnp.concatenate((z_scores_control, z_scores_patient), axis=0)


# Create the bar plot
fig, ax = plt.subplots()

bars = ax.bar(numbers, z_scores, width=0.5, color='blue')

# Adding labels and title
ax.set_ylabel('z scores')
ax.set_title('Score Expression PSP vs HC')
#ax.set_ylim([0, 1])  # Set the y-axis limits

# Display the values on top of the bars
for bar in bars:
    height = bar.get_height()
    #ax.annotate(f'{height:.2f}',
    #            xy=(bar.get_x() + bar.get_width() / 2, height),
    #            xytext=(0, 3),  # 3 points vertical offset
    #            textcoords="offset points",
    #            ha='center', va='bottom')

# Show the plot
plt.show()

exit()



# the latent data produced by the PCA in psp-hc-pc.py, transfer to 0 based counting
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

