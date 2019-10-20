import numpy as np
import matplotlib.pyplot as plt
import glob, sys
from scipy.misc import imread
from sklearn.preprocessing import OneHotEncoder

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
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

def categorize(param, num_classes):
    """Return parameters (in range [0,1]) in categories based on uniform
    segmentation of interval into num_classes."""
    cat = param*0.
    bins = np.array([float(k)/num_classes for k in range(num_classes)])
    cat = np.digitize(param, bins) - 1
    cat = cat.astype('int32')
    print(cat)

    return cat

def main():
    L = int(sys.argv[1])
    input_shape = (28, 28, 1)
    num_classes = 5

    # Set up model with batch normalization
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

    # Extract gamma and omega values from data
    images = glob.glob("images/" + '*.png')
    omega = [float(x.split('_')[3][:-4]) for x in images]
    omega = np.asarray(omega).reshape(-1, 1)
    gamma = [float(x.split('_')[1]) for x in images]
    gamma = np.asarray(gamma)#.reshape(-1, 1)

    # Make sure gamma is in [0,1]
    gamma += 2.
    gamma /= 4.

    gamma_cat = categorize(gamma, num_classes)
    gamma_cat = to_categorical(gamma_cat, num_classes)

    # Read in data
    X = read_images("images/").astype('float32')
    X = np.reshape(X, (len(X), L, L, 1))

    # and normalize
    X /= 255

    # split into train and test data
    X_train, X_test = X[:29000], X[29001:]
    y_train, y_test = gamma_cat[:29000], gamma_cat[29001:]

    # Fit model
    history_callback = model.fit(X_train, y_train, batch_size=128, epochs=100, verbose=1, validation_split=0.2)
    # Load model
    y_predict = model.predict(X_test)

    # Test model
    score = model.evaluate(X_test, y_test)
    print("Final accuracy: {}".format(score[1]))

if __name__ == '__main__':
    main()
