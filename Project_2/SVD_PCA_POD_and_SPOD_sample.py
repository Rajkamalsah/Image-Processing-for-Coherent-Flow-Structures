# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:41:13 2024

@author: Raj
"""
#%%SVD sample code
import numpy as np

import matplotlib.pyplot as plt
# Create a matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Perform Singular Value Decomposition
U, S, VT = np.linalg.svd(A)

print("U matrix:\n", U)
print("Singular values:\n", S)
print("V^T matrix:\n", VT)





#%% PCA sample code
import numpy as np
from sklearn.decomposition import PCA
# Create the same matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
# Perform PCA
pca = PCA(n_components=3)
pca.fit(A)

# Get the principal components
principal_components = pca.components_

# Get the explained variance
explained_variance = pca.explained_variance_

# Get the transformed data
transformed_data = pca.transform(A)

print("Principal components:\n", principal_components)
print("Explained variance:\n", explained_variance)
print("Transformed data:\n", transformed_data)

#%% POD
import numpy as np
import matplotlib.pyplot as plt
# Define the matrix
X=np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])


# Compute the mean of each column
mean_X = np.mean(X, axis=0)

# Subtract the mean from each column
X_prime = X - mean_X

# Compute the covariance matrix
C = np.dot(X_prime, X_prime.T) / X.shape[1]

# Compute the eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(C)

# The eigenvectors are the POD modes
POD_modes = eigenvectors

# Print the POD modes and their corresponding energy contents
print("POD Modes:")
print(POD_modes)
print("Energy Contents:")
print(eigenvalues)


#%%SPOD

data_matrix=np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Number of snapshots
N = data_matrix.shape[1]

# Choose a window size M
M = N // 2  # For example, we choose M as half of N

# Preallocate memory for SPOD modes
SPOD_modes = np.zeros((data_matrix.shape[0], M, N - M + 1))

# Preallocate memory for time coefficients
time_coeffs = np.zeros((M, N - M + 1))

# Loop over all snapshots
for i in range(N - M + 1):
    # Extract snapshot
    snapshot = data_matrix[:, i:i+M]

    # Compute cross-spectral density matrix
    R = np.dot(snapshot, snapshot.T) / M

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(R)

    # Sort the eigenvalues in descending order and get the indices
    idx = eigenvalues.argsort()[::-1]

    # Sort the eigenvalues and eigenvectors according to the sorted indices
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    # Take only the first M eigenvectors
    eigenvectors = eigenvectors[:,:M]

    # Compute SPOD modes
    SPOD_modes[:, :, i] = eigenvectors

    # Compute time coefficients for each mode
    for j in range(M):
        for k in range(M):
            time_coeffs[j, i] += np.dot(eigenvectors[:, j].T, snapshot[:, k])

# Print the SPOD modes
print("SPOD Modes:")
print(SPOD_modes)

# Print the time coefficients
print("Time Coefficients:")
print(time_coeffs)

# Choose a specific mode
mode_index = 0  # Replace with your desired mode index

# Get the SPOD mode and Fourier coefficients for the chosen mode
spod_mode = SPOD_modes[:, mode_index, :]
fourier_coeff = time_coeffs[mode_index, :]

# Reshape the SPOD mode into a 2D array for plotting
# Here, we're assuming that your original data matrix represents a 2D field
# You might need to adjust this depending on the nature of your data
spod_mode_2d = spod_mode.reshape((int(np.sqrt(spod_mode.size)), -1))

# Plot the SPOD mode as a contour plot
plt.figure(figsize=(6, 5))
plt.contourf(spod_mode_2d, cmap='viridis')
plt.colorbar()
plt.title('SPOD Mode')
plt.show()

# Plot the temporal evolution of the SPOD mode
plt.figure(figsize=(6, 5))
plt.plot(np.abs(fourier_coeff))
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Temporal Evolution of SPOD Mode')
plt.grid(True)
plt.show()
