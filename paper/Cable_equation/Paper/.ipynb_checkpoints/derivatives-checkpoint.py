# %% Imports
from scipy.ndimage import convolve1d
from scipy.interpolate import UnivariateSpline
import numpy as np
from deepymod.data import Dataset
from deepymod.data.burgers import BurgersDelta
from sklearn.linear_model import LassoCV

# %% Functions
def finite_diff(y, x, order, axis=0, bc_mode='reflect'):
    ''' Calculates finite difference of order n over axis. 
    Uses 2nd order accurate central difference.'''
    step_size = np.diff(x)[0] # assumes step size is constant
    if order == 1:
        stencil = np.array([1/2, 0, -1/2])
    elif order == 2:
        stencil = np.array([1, -2, 1])
    elif order == 3:
        stencil = np.array([1/2, -1, 0, 1, -1/2])
    else:
        raise NotImplementedError

    deriv = convolve1d(y, stencil, axis=axis, mode=bc_mode) / step_size**order
    return deriv


def spline_diff(y, x, order, **spline_kwargs):
    """Fits spline to data and returns derivatives of given order. order=0 corresponds to data.
    Good defaults for spline, k=4, s=1e-2/0.0 if not smooth"""
    spline = UnivariateSpline(x, y, **spline_kwargs)
    return spline(x, nu=order)


def library(y, x, t, poly_order=2, deriv_order=3, deriv_kind='spline', **deriv_kwargs):
    ''' Returns time deriv and library of given data. x and t are vectors, first axis of y should be time.'''
    if deriv_kind == 'spline':
        # Calculating polynomials
        u = np.stack([spline_diff(y[frame, :], x, order=0, **deriv_kwargs) for frame in np.arange(t.size)], axis=0).reshape(-1, 1) # if we do a spline on noisy data, we also get a 'denoised' data
        u = np.concatenate([u**order for order in np.arange(poly_order+1)], axis=1) # getting polynomials including offset

        # Calculating derivatives
        du = [np.ones((u.shape[0], 1))]
        for order in np.arange(1, deriv_order+1):
            du.append(np.stack([spline_diff(y[frame, :], x, order=order, **deriv_kwargs) for frame in np.arange(t.size)], axis=0).reshape(-1, 1)) 
        du = np.concatenate(du, axis=1)

        # Calculating theta
        theta = (u[:, :, None] @ du[:, None, :]).reshape(-1, u.shape[1] * du.shape[1])
      
    elif deriv_kind == 'fd':
        # Calculating polynomials
        u = np.concatenate([(y**order).reshape(-1, 1) for order in np.arange(poly_order+1)], axis=1)

        # Getting derivatives
        du = np.concatenate([(finite_diff(y, x, order=order, axis=1, **deriv_kwargs)).reshape(-1, 1) for order in np.arange(1, deriv_order+1)], axis=1)
        du = np.concatenate((np.ones((du.shape[0], 1)), du), axis=1)

        # Calculating theta
        theta = (u[:, :, None] @ du[:, None, :]).reshape(-1, u.shape[1] * du.shape[1])

    else:
        raise NotImplementedError
    # Calculating time diff by finite diff
    dt = finite_diff(u[:, 1].reshape(t.size, x.size), t, order=1, axis=0).reshape(-1, 1)
    return dt, theta

