# imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.linalg as la
import matplotlib.cm as cm
from matplotlib import rc
import time
from IPython import display

np.random.seed(42)

# Choose a GMM with 3 components

# means
m = np.zeros((3,2))
m[0] = np.array([1.2, 0.4])
m[1] = np.array([-4.4, 1.0])
m[2] = np.array([4.1, -0.3])

# covariances
S = np.zeros((3,2,2))
S[0] = np.array([[0.8, -0.4], [-0.4, 1.0]])
S[1] = np.array([[1.2, -0.8], [-0.8, 1.0]])
S[2] = np.array([[1.2, 0.6], [0.6, 3.0]])

# mixture weights
w = np.array([0.3, 0.2, 0.5])

"""
    Genetate Data
"""
def generate_data():
    N_split = 200 # number of data points per mixture component
    N = N_split*3 # total number of data points
    x = []
    y = []
    for k in range(3):
        x_tmp, y_tmp = np.random.multivariate_normal(m[k], S[k], N_split).T 
        x = np.hstack([x, x_tmp])
        y = np.hstack([y, y_tmp])

    data = np.vstack([x, y])
    return data, x, y, N

"""
    Visualization dataset
"""
def visulization_data(m, S, x, y):
    X, Y = np.meshgrid(np.linspace(-10,10,100), np.linspace(-10,10,100))
    pos = np.dstack((X, Y))

    mvn = multivariate_normal(m[0,:].ravel(), S[0,:,:])
    xx = mvn.pdf(pos)

    # plot the dataset
    plt.figure()
    plt.title("Mixture components")
    plt.plot(x, y, 'ko', alpha=0.3)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    # plot the individual components of the GMM
    plt.plot(m[:,0], m[:,1], 'or')

    for k in range(3):
        mvn = multivariate_normal(m[k,:].ravel(), S[k,:,:])
        xx = mvn.pdf(pos)
        plt.contour(X, Y, xx,  alpha = 1.0, zorder=10)
        
    # plot the GMM 
    plt.figure()
    plt.title("GMM")
    plt.plot(x, y, 'ko', alpha=0.3)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    # build the GMM
    gmm = 0
    for k in range(3):
        mix_comp = multivariate_normal(m[k,:].ravel(), S[k,:,:])
        gmm += w[k]*mix_comp.pdf(pos)
        
    plt.contour(X, Y, gmm,  alpha = 1.0, zorder=10)

    plt.show()

"""
    Train GMM in EM
"""

if __name__ == "__main__":
    K = 3 # number of clusters
    data, x, y, N = generate_data()
    X, Y = np.meshgrid(np.linspace(-10,10,100), np.linspace(-10,10,100))
    pos = np.dstack((X, Y))
    
    means = np.zeros((K,2))
    covs = np.zeros((K,2,2))
    for k in range(K):
        means[k] = np.random.normal(size=(2,))
        covs[k] = np.eye(2)

    weights = np.ones((K,1))/K
    print("Initial mean vectors (one per row):\n" + str(means))
    #EDIT THIS FUNCTION
    NLL = [] # log-likelihood of the GMM
    gmm_nll = 0
    for k in range(K):
        gmm_nll += weights[k]*multivariate_normal.pdf(mean=means[k,:], cov=covs[k,:,:], x=data.T)
    NLL += [-np.sum(np.log(gmm_nll))]

    # plt.figure()
    # plt.plot(x, y, 'ko', alpha=0.3)
    # plt.plot(means[:,0], means[:,1], 'oy', markersize=25)

    # for k in range(K):
    #     rv = multivariate_normal(means[k,:], covs[k,:,:])
    #     plt.contour(X, Y, rv.pdf(pos), alpha = 1.0, zorder=10)
        
    # plt.xlabel("$x_1$")
    # plt.ylabel("$x_2$")
    # plt.show()
    
    r = np.zeros((K,N)) # will store the responsibilities

    for em_iter in range(100):    
        means_old = means.copy()
        
        # E-step: update responsibilities
        for k in range(K):
            r[k] = weights[k]*multivariate_normal.pdf(mean=means[k,:], cov=covs[k,:,:], x=data.T)  
            
        r = r/np.sum(r, axis=0) 
            
        # M-step
        N_k = np.sum(r, axis=1)

        for k in range(K): 
            # update means
            means[k] = np.sum(r[k]*data, axis=1)/N_k[k]
            
            # update covariances
            diff = data - means[k:k+1].T
            _tmp = np.sqrt(r[k:k+1])*diff
            covs[k] = np.inner(_tmp, _tmp)/N_k[k]
            
        # weights
        weights = N_k/N 
        
        # log-likelihood
        gmm_nll = 0
        for k in range(K):
            gmm_nll += weights[k]*multivariate_normal.pdf(mean=means[k,:].ravel(), cov=covs[k,:,:], x=data.T)
        NLL += [-np.sum(np.log(gmm_nll))]
        
        # plt.figure() 
        # plt.plot(x, y, 'ko', alpha=0.3)
        # plt.plot(means[:,0], means[:,1], 'oy', markersize=25)
        # for k in range(K):
        #     rv = multivariate_normal(means[k,:], covs[k])
        #     plt.contour(X, Y, rv.pdf(pos), alpha = 1.0, zorder=10)
            
        # plt.xlabel("$x_1$")
        # plt.ylabel("$x_2$")
        # plt.text(x=3.5, y=8, s="EM iteration "+str(em_iter+1))
        
        if la.norm(NLL[em_iter+1]-NLL[em_iter]) < 1e-6:
            print("Converged after iteration ", em_iter+1)
            break
    
    # plot final the mixture model
    # plt.figure() 
    gmm = 0
    for k in range(3):
        mix_comp = multivariate_normal(means[k,:].ravel(), covs[k,:,:])
        gmm += weights[k]*mix_comp.pdf(pos)

    # plt.plot(x, y, 'ko', alpha=0.3)
    # plt.contour(X, Y, gmm,  alpha = 1.0, zorder=10)    
    # plt.xlim([-8,8])
    # plt.ylim([-6,6])
    # plt.show()
    
    plt.figure()
    plt.semilogy(np.linspace(1,len(NLL), len(NLL)), NLL)
    plt.xlabel("EM iteration");
    plt.ylabel("Negative log-likelihood");

    idx = [0, 1, 9, em_iter+1]

    for i in idx:
        plt.plot(i+1, NLL[i], 'or')
        
    plt.show()