# PACKAGE: DO NOT EDIT THIS CELL
import numpy as np
import timeit
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from ipywidgets import interact
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA as SKPCA


def normalize(X):
    """Normalize the given dataset X
    Args:
        X: ndarray, dataset
    
    Returns:
        (Xbar, mean, std): tuple of ndarray, Xbar is the normalized dataset
        with mean 0 and standard deviation 1; mean and std are the 
        mean and standard deviation respectively.
    
    Note:
        You will encounter dimensions where the standard deviation is
        zero, for those when you do normalization the normalized data
        will be NaN. Handle this by setting using `std = 1` for those 
        dimensions when doing normalization.
    """
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std_filled = std.copy()
    std_filled[std==0] = 1.
    Xbar = ((X-mu)/std_filled)
    return Xbar, mu, std

def eig(S):
    """Compute the eigenvalues and corresponding eigenvectors 
        for the covariance matrix S.
    Args:
        S: ndarray, covariance matrix
    
    Returns:
        (eigvals, eigvecs): ndarray, the eigenvalues and eigenvectors

    Note:
        the eigenvals and eigenvecs should be sorted in descending
        order of the eigen values
    """
    eigvals, eigvecs = np.linalg.eig(S)
    k = np.argsort(eigvals)[::-1]
    return eigvals[k], eigvecs[:,k]

def projection_matrix(B):
    """Compute the projection matrix onto the space spanned by `B`
    Args:
        B: ndarray of dimension (D, M), the basis for the subspace
    
    Returns:
        P: the projection matrix
    """
    return (B @ np.linalg.inv(B.T @ B) @ B.T)

def PCA(X, num_components):
    """
    Args:
        X: ndarray of size (N, D), where D is the dimension of the data,
           and N is the number of datapoints
        num_components: the number of principal components to use.
    Returns:
        X_reconstruct: ndarray of the reconstruction
        of X from the first `num_components` principal components.
    """
    # first perform normalization on the digits so that they have zero mean and unit variance
    # Then compute the data covariance matrix S
    S = 1.0/len(X) * np.dot(X.T, X)

    # Next find eigenvalues and corresponding eigenvectors for S
    eig_vals, eig_vecs = eig(S)

    # find indices for the largest eigenvalues, use them to sort the eigenvalues and 
    # corresponding eigenvectors. Take a look at the documenation fo `np.argsort` 
    # (https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html),
    # which might be useful
    eig_vals, eig_vecs = eig_vals[:num_components], eig_vecs[:, :num_components]

    # dimensionality reduction of the original data
    B = np.real(eig_vecs)
    # Z = X.T.dot(W)
    # reconstruct the images from the lower dimensional representation
    reconst = (projection_matrix(B) @ X.T)
    return reconst.T

def mse(predict, actual):
    """Helper function for computing the mean squared error (MSE)"""
    return np.square(predict - actual).sum(axis=1).mean()


def show_pca_digits(i=1):
    """Show the i th digit and its reconstruction"""
    plt.figure(figsize=(4,4))
    actual_sample = X[i].reshape(28,28)
    reconst_sample = (reconst[i, :] * std + mu).reshape(28, 28)
    plt.imshow(np.hstack([actual_sample, reconst_sample]), cmap='gray')
    plt.show()

if __name__ == "__main__":
    images, labels = fetch_openml('mnist_784', version=1, return_X_y=True)

    images = np.array(images)
    labels = np.array(labels)
    
    NUM_DATAPOINTS = 1000
    X = (images.reshape(-1, 28 * 28)[:NUM_DATAPOINTS]) / 255.
    Xbar, mu, std = normalize(X)
    
    # for num_component in range(1, 20):
    #     # We can compute a standard solution given by scikit-learn's implementation of PCA
    #     pca = SKPCA(n_components=num_component, svd_solver='full')
    #     sklearn_reconst = pca.inverse_transform(pca.fit_transform(Xbar))
    #     reconst = PCA(Xbar, num_component)
    #     np.testing.assert_almost_equal(reconst, sklearn_reconst)
    #     print(np.square(reconst - sklearn_reconst).sum())
        
    loss = []
    reconstructions = []
    
    # iterate over different numbers of principal components, and compute the MSE
    for num_component in range(1, 100):
        reconst = PCA(Xbar, num_component)
        error = mse(reconst, Xbar)
        reconstructions.append(reconst)
        # print('n = {:d}, reconstruction_error = {:f}'.format(num_component, error))
        loss.append((num_component, error))

    reconstructions = np.asarray(reconstructions)
    reconstructions = reconstructions * std + mu # "unnormalize" the reconstructed image
    loss = np.asarray(loss)
    
    # fig, ax = plt.subplots()
    # ax.plot(loss[:,0], loss[:,1]);
    # ax.axhline(100, linestyle='--', color='r', linewidth=2)
    # ax.xaxis.set_ticks(np.arange(1, 100, 5));
    # ax.set(xlabel='num_components', ylabel='MSE', title='MSE vs number of principal components');
    # plt.figure(figsize=(4,4))
    # plt.imshow(images[0].reshape(28,28), cmap='gray')
    show_pca_digits(0)