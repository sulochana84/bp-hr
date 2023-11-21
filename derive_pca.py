"""PCA calculation"""
import numpy as np
def  pca_redchannel(red_channel):
    """this function retuns the PCA recostructed image """
    #assign the red channel to X.
    X = red_channel

    #Specifiy the number of principal components
    num_components = 10
    # Step-1
    X_meaned = X - np.mean(X, axis=0)

   # Step-2
    cov_mat = np.cov(X_meaned, rowvar=False)

   # Step-3
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

   # Step-
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

   # Step-5
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    # Step-6
    X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()


    # Step-7
    X_reconstructed = np.dot(X_reduced, eigenvector_subset.T)
    return X_reconstructed