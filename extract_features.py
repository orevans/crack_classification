import numpy as np
import matplotlib.pyplot as plt
import glob, sys

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.preprocessing import image

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import k_means
from scipy.stats import pearsonr

from autoencoder import read_images

def main():
    # Load model
    model = load_model('autoencoder_epochs_400.h5')

    # Create new model that outputs at the encoder layer
    model = Model(model.inputs, model.layers[6].output)

    # Read in data
    L = 28
    X = read_images("images/").astype('float32')
    X = np.reshape(X, (len(X), L, L, 1))

    # Get encoder layer features and flatten
    encoder = model.predict(X)
    encoder = encoder.reshape(X.shape[0],128)

    # Extract gamma and omega values from data
    images = glob.glob("images/" + '*.png')
    gamma = [float(x.split('_')[1]) for x in images]
    omega = [float(x.split('_')[3][:-4]) for x in images]
    omega = np.asarray(omega).reshape(-1, 1)
    gamma = np.asarray(gamma).reshape(-1, 1)

    # K-means clustering
    #n_clusters = 15
    #_clusters = k_means(encoder, n_clusters)
    #_labels = _clusters[1]
    #print(np.bincount(_clusters[1]))

    #n = 15
    #fig, ax = plt.subplots(1,15)
    #for i in range(n):
    #    ax[i].scatter(omega[_labels == i],gamma[_labels == i],alpha=0.1)

    #plt.show()

    # Try PCA
    _pca = PCA(n_components=2)
    _pca.fit(encoder.transpose())
    print(_pca.explained_variance_ratio_)

    # Plot PCA components vs. parameters
    comp  = _pca.components_[0,:].reshape(-1, 1)

    for i in range(2):
        comp = _pca.components_[i,:].reshape(-1, 1)
        regr = LinearRegression().fit(omega,comp)
        print(regr.score(gamma,comp))
        print(pearsonr(gamma,comp))

    #fig, ax = plt.subplots()
    #ax.scatter(omega,comp,alpha=0.05)
    #ax.set_ylim(comp.min(),comp.max())
    #ax.grid(True)
    #plt.show()




if __name__ == '__main__':
    main()
