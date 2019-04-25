import numpy as np
import matplotlib.pyplot as plt
import glob, sys
from scipy.misc import imread

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.preprocessing import image

def convert_image(img_path):
    """Convert png to array"""
    # Load image and convert to array
    img_arr = imread(img_path, mode='L')
    return img_arr

def read_images(img_dir):
    """Load images into a single array and preprocess"""
    # Find all .jpg files
    images = glob.glob(img_dir + '*.png')

    print("Loading images...")
    x = []
    for im in images:
        x.append(convert_image(im))

    return np.asarray(x)

def main():
    L = int(sys.argv[1])

    input_img = Input(shape=(L, L, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    print(encoder.summary())
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Read in data
    X = read_images("images/").astype('float32')
    X = np.reshape(X, (len(X), L, L, 1))

    # and normalize
    X /= 255

    # split into train and test data
    X_train, X_test = X[:9500], X[9501:]

    history_callback = autoencoder.fit(X_train, X_train, batch_size=128, epochs=400, verbose=1, validation_split=0.1)
    #decoded_imgs = autoencoder.predict(X_test)

    #n = 10
    #plt.figure(figsize=(20, 4))
    #for i in range(1, n+1):
        # display original
        #    ax = plt.subplot(2, n, i)
        #    plt.imshow(X_test[i].reshape(L, L))
        #          plt.gray()
        #    ax.get_xaxis().set_visible(False)
        #    ax.get_yaxis().set_visible(False)

        # display reconstruction
        #    ax = plt.subplot(2, n, i + n)
        #    plt.imshow(decoded_imgs[i].reshape(L, L))
        #    plt.gray()
        #    ax.get_xaxis().set_visible(False)
        #    ax.get_yaxis().set_visible(False)
        #plt.show()

    # Save model
    autoencoder.save('autoencoder_epochs_400.h5')
    del autoencoder

    # Load model
    #model = load_model('models/autoencoder.h5')

    # Test model
    #score = model.evaluate(X_test, X_test)
    #print("Final accuracy: {}".format(score[1]))
