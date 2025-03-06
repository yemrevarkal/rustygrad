import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Generate 1000 samples for 3 normal features (x1, x2, x3)
n_samples = 100
x1 = np.random.normal(0, 1, n_samples)
x2 = np.random.normal(5, 2, n_samples)
x3 = np.random.normal(-3, 1.5, n_samples)

# Create hidden features (polynomial relationships)
hidden_1 = x1 ** 2  # Square of x1
hidden_2 = x2 * x3  # Product of x2 and x3

# Build the target variable (linear combination with noise)
noise = np.random.normal(0, 0.5, n_samples)  # White noise
y = 3 * x1 + 2 * x2 - 1.5 * x3 

# Create a DataFrame with only the normal features and target
df = pd.DataFrame({
    "x1": x1,
    "x2": x2,
    "x3": x3,
    "y": y
})

# Save to CSV
df.to_csv("src/synthetic_data.csv", index=False)

print("Dataset created and saved as synthetic_data.csv")

