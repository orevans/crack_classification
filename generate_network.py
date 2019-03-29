import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

def initiation_probabilities(grid, gamma):
    """Return probability matrix based on distances to nearest fracture"""
    # Compute distance matrix
    d = distance_transform_edt(1 - grid)

    # Raise to the power of gamma
    d = np.power(d, gamma)

    # Normalize by Z = \sum_ij d_ij
    d /= np.sum(d)

    return d

def create_grid(L):
    """Create grid of size L by L (with 1's on the boundaries and 0's o/w)"""
    # Set up grid (L by L pixels)
    grid = np.zeros((256,256))

    # Set boundaries of grid to 1
    grid[:,0] = 1.
    grid[0,:] = 1.
    grid[L - 1,:] = 1.
    grid[:,L - 1] = 1.

    return grid

def main():
    # Grid dimension
    L = 256

    # Number of fractures to generate
    n = 50

    # Create grid
    grid = create_grid(L)

    # Calculation matrix of initialization probabilities
    gamma = 0.1
    d = initiation_probabilities(grid, gamma)

if __name__=="__main__":
    main()
