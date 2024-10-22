"""Computes the scores of the train set with respect to the derived pattern"""

from pathlib import Path

import jax.numpy as jnp
import nibabel
from matplotlib import pyplot as plt
from medclassx.mask_trafo import mask_vector

outfolder = Path("examples/exp_lateNC/out")



# load pattern
pattern = jnp.load(outfolder / "pattern.npy")
a, b, c = pattern.shape

# flatten pattern
pattern = jnp.reshape(pattern, shape=(-1,))
print(pattern.shape)

# mask pattern
mask_path = outfolder / "mask_resampled.nii"
nifti_mask = nibabel.load(mask_path)
mask = nifti_mask.get_fdata()
mask = mask.reshape(-1)
pattern, unmask = mask_vector(pattern, mask)
print(pattern.shape)

# load dataset
X = jnp.load(outfolder / "X.npy")
X_control = X[0:9, :]
X_patient = X[9:,:]

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
numbers = jnp.arange(1, 19)  # 1 to 10
z_scores_control = S_control_zscore[0:9]
z_scores_patient = S_patient_zscore[0:9]
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