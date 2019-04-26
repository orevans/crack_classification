import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.misc import imsave
import sys, random

class Network():
    def __init__(self, gamma, omega, L=224, n=50):
        self.gamma = gamma
        self.omega = omega
        self.L = L
        self.n = n
        self.grid = self.create_grid()

    def initiation_probabilities(self):
        """Return probability matrix based on distances to nearest fracture"""
        # Compute distance matrix
        d = distance_transform_edt(1 - self.grid)

        # Raise to the power of gamma
        d[d != 0] = np.power(d[d != 0], self.gamma)

        # Normalize by Z = \sum_ij d_ij
        d /= np.sum(d)

        return d

    def create_grid(self):
        """Create grid of size L by L (with 1's on the boundaries and 0's o/w)"""
        # Set up grid (L by L pixels)
        grid = np.zeros((self.L,self.L))

        # Set boundaries of grid to 1
        grid[:,0] = 1.
        grid[0,:] = 1.
        grid[self.L - 1,:] = 1.
        grid[:,self.L - 1] = 1.

        return grid

    def propagation_direction(self):
        """Generate a random direction for fracture to propagate in."""
        # Step size
        step = np.random.randint(5)

        # Pick whether step size is in the vertical or horizontal direction
        step_direction = np.random.randint(2)

        if step_direction:
            return [random.choice([-1., 1.])*1., step]
        else:
            return[step, random.choice([-1., 1.])*1.]

    def propagate_fractures(self, i, j):
        """Propagate fractures according to values of omega:
        if omega = 0, propagate it opposite directions until both ends terminate
        if omega = 1, propagate until on of the ends terminates"""
        # Propagate fracture in random direction
        v = self.propagation_direction()
        prop = np.random.choice((0,1), p=(self.omega,1. - self.omega))

        i_, j_ = i, j
        if prop:
            hit_boundary = False
            while True:
                # Need to break out of double while loop
                if hit_boundary:
                    break

                i_counter, j_counter = abs(int(v[0])), abs(int(v[1]))
                while (i_counter > 0) or (j_counter > 0):
                    # Propagate first end until it meets a fracture/boundary
                    i_ += np.sign(int(v[0]))*(i_counter > 0)
                    j_ += np.sign(int(v[1]))*(j_counter > 0)
                    if self.grid[i_,j_] == 1.:
                        hit_boundary = True
                        break
                    else:
                        self.grid[i_,j_] = 1.
                        
                    if i_counter > 0:
                        i_counter -= 1
                    if j_counter > 0:
                        j_counter -= 1

            i_, j_ = i, j
            hit_boundary = False
            while True:
                # Need to break out of double while loop
                if hit_boundary:
                    break

                i_counter, j_counter = abs(int(v[0])), abs(int(v[1]))
                while (i_counter > 0) or (j_counter > 0):
                    # Propagate first end until it meets a fracture/boundary
                    i_ -= np.sign(int(v[0]))*(i_counter > 0)
                    j_ -= np.sign(int(v[1]))*(j_counter > 0)

                    if self.grid[i_,j_] == 1.:
                        hit_boundary = True
                        break
                    else:
                        self.grid[i_,j_] = 1.

                    if i_counter > 0:
                        i_counter -= 1
                    if j_counter > 0:
                        j_counter -= 1

        else:
            i_0, j_0 = i, j
            i_1, j_1 = i, j
            hit_boundary = False
            while True:
                # Need to break out of double while loop
                if hit_boundary:
                    break

                i_counter, j_counter = abs(int(v[0])), abs(int(v[1]))
                while (i_counter > 0) or (j_counter > 0):
                    # Propagate first end until it meets a fracture/boundary
                    i_0 += np.sign(int(v[0]))*(i_counter > 0)
                    j_0 += np.sign(int(v[1]))*(j_counter > 0)
                    i_1 -= np.sign(int(v[0]))*(i_counter > 0)
                    j_1 -= np.sign(int(v[1]))*(j_counter > 0)

                    if self.grid[i_0,j_0] == 1.:
                        hit_boundary = True
                        break
                    elif self.grid[i_1,j_1] == 1.:
                        hit_boundary = True
                        break
                    else:
                        self.grid[i_0,j_0] = 1.
                        self.grid[i_1,j_1] = 1.

                    if i_counter > 0:
                        i_counter -= 1
                    if j_counter > 0:
                        j_counter -= 1

    def generate_network(self):
        """Generate fracture grid and save as a .png image"""
        for k in range(self.n):
            # Calculate matrix of initialization probabilities
            d = self.initiation_probabilities().flatten()

            # Pick location to initiate next fracture
            i, j = np.unravel_index(np.random.choice(np.arange(self.L**2),p=d),
                                                    (self.L,self.L))

            # Update grid
            self.grid[i,j] = 1

            # Propagate fractures
            self.propagate_fractures(i,j)

        # Save image
        imsave('images/gamma_{}_omega_{}.png'.format(self.gamma,self.omega),
             1. - self.grid)

def progress_bar(iteration, total, prefix = '', suffix = '',
                decimals = 1, length = 100, fill = '█'):
    """Print progress bar"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

def main():
    # Number of sample
    s = int(sys.argv[1])

    # Pick gamma from a standard normal
    mu_gamma, sigma_gamma = 0, 1.
    #gamma = np.random.normal(mu_gamma, sigma_gamma, s)

    gamma = np.random.uniform(-2., 2., s)

    # Pick omega uniformly
    omega = np.random.uniform(0., 1., s)

    for k in range(s):
        cracks = Network(gamma[k],omega[k])
        cracks.generate_network()
        progress_bar(k,s)

if __name__=="__main__":
    main()
