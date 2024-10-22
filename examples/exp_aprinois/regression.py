# Step 1: Import necessary libraries
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# the latent data produced by the PCA in psp-hc-pc.py
T = jnp.load(Path(r"examples\exp_aprinois\out\latent_data.npy"))[:, 1:2]

# the first 30 patients are healthy controls, the last 30 are PSP diagnosed
y_1 = jnp.zeros(shape=(30,), dtype=int)
y_2 = jnp.ones(shape=(30,), dtype=int)
y = jnp.concatenate((y_1, y_2), axis=0)

# Step 3: Split the dataset into training and testing sets
T_train, T_test, y_train, y_test = train_test_split(T, y, test_size=0.5, random_state=0)

# Step 4: Create and train the logistic regression model
model = LogisticRegression(max_iter=200)

# uses L-BFGS per default
model.fit(T_train, y_train)

# Access the trained parameters
weights = model.coef_
intercept = model.intercept_

# Print the weights and intercept
print("Weights (coefficients):")
print(weights)
print("Intercept:")
print(intercept)

# Step 5: Make predictions
y_pred = model.predict(T_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the evaluation results
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
