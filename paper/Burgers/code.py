import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LassoCV 
from deepymod.data import Dataset
from deepymod.data.burgers import BurgersDelta


def fit_spline(x, u, k=5, s=0.0):
    spline = np.stack([UnivariateSpline(x, u[frame, :], k=k, s=s) for frame in np.arange(u.shape[0])], axis=0)
    return spline


def derivative(splines, x, order):
    return np.stack([spline(x, order) for spline in splines], axis=0)


def fit(u, u_t, u_x, u_xx, u_xxx):
    # Fit library to data
    threshold = 0.2
    reg = LassoCV(fit_intercept=False, max_iter=1e5)   

    y = u_t.reshape(-1, 1)
    v = np.concatenate([np.ones_like(u.reshape(-1, 1)), u.reshape(-1, 1), u.reshape(-1, 1)**2], axis=1)[:, :, None]
    dv = np.concatenate([np.ones_like(u.reshape(-1, 1)), u_x.reshape(-1, 1), u_xx.reshape(-1, 1), u_xxx.reshape(-1, 1)], axis=1)[:, None, :]
    theta = (v @ dv).reshape(-1, 12)
    
    theta = theta / np.linalg.norm(theta, axis=0, keepdims=True)
    y = y / np.linalg.norm(y, axis=0, keepdims=True)
    coeffs = reg.fit(theta, y.squeeze()).coef_
    coeffs[np.abs(coeffs) < threshold] = 0.0
    return coeffs[:, None]


def correct_eq(found_coeffs):
    # Correct coeffs for burgers
    correct_coeffs = np.zeros((12, 1))
    correct_coeffs[[2, 5]] = 1.0

    n_active_terms_incorrect = np.sum(found_coeffs[correct_coeffs != 0.0] == 0)
    n_inactive_terms_incorrect = np.sum(found_coeffs[correct_coeffs == 0.0] != 0)
    if n_active_terms_incorrect + n_inactive_terms_incorrect > 0:
        correct = False
    else:
        correct = True
    return correct


def make_spline_grid_data(n_x=np.arange(10, 51, 5), n_t=100, A=1, v=0.25, noise=0.0, **spline_kwargs):
    # Making dataframe and dataset
    df = pd.DataFrame()
    dataset = Dataset(BurgersDelta, A=A, v=v)

    # Adding grid sizes
    df['n_x'] = n_x
    df['n_t'] = n_t

    # Adding grids
    df['x'] = [np.linspace(-3, 4, row.n_x) for idx, row in df.iterrows()]
    df['t'] = [np.linspace(0.1, 1.1, row.n_t) for idx, row in df.iterrows()]
    df['t_grid'] = [np.meshgrid(row.t, row.x, indexing='ij')[0] for idx, row in df.iterrows()]
    df['x_grid'] = [np.meshgrid(row.t, row.x, indexing='ij')[1] for idx, row in df.iterrows()]

    # Generating solution and fitting spline
    df['u'] = [dataset.generate_solution(row.x_grid, row.t_grid) for idx, row in df.iterrows()]
    df['u'] = [row.u + noise * np.std(row.u) * np.random.randn(*row.u.shape) for idx, row in df.iterrows()] 
    df['spline'] = [fit_spline(row.x, row.u, **spline_kwargs) for idx, row in df.iterrows()]

    # Calculating derivatives via splines
    df['u_spline'] = [derivative(row.spline, row.x, 0) for idx, row in df.iterrows()] 
    df['u_x_spline'] = [derivative(row.spline, row.x, 1) for idx, row in df.iterrows()]
    df['u_xx_spline'] = [derivative(row.spline, row.x, 2) for idx, row in df.iterrows()]
    df['u_xxx_spline'] = [derivative(row.spline, row.x, 3) for idx, row in df.iterrows()]
    df['u_t_spline'] = [np.gradient(row.u_spline, row.t, axis=0) for idx, row in df.iterrows()]

    # Calculating true derivatives
    df['u_t'] = [dataset.time_deriv(row.x_grid, row.t_grid).reshape(row.x_grid.shape) for idx, row in df.iterrows()]
    df['u_x'] = [dataset.library(row.x_grid.reshape(-1, 1), row.t_grid.reshape(-1, 1), poly_order=2, deriv_order=3)[:, 1].reshape(row.x_grid.shape) for idx, row in df.iterrows()]
    df['u_xx'] = [dataset.library(row.x_grid.reshape(-1, 1), row.t_grid.reshape(-1, 1), poly_order=2, deriv_order=3)[:, 2].reshape(row.x_grid.shape) for idx, row in df.iterrows()]
    df['u_xxx'] = [dataset.library(row.x_grid.reshape(-1, 1), row.t_grid.reshape(-1, 1), poly_order=2, deriv_order=3)[:, 3].reshape(row.x_grid.shape) for idx, row in df.iterrows()]

    # Calculating normalizing properties
    df['l'] = [np.sqrt(4 * v * row.t)[:, None] for idx, row in df.iterrows()]
    df['dz'] = [(np.ones_like(row.t)[:, None] * np.diff(row.x)[0] / row.l) for idx, row in df.iterrows()]
    df['u0'] = [np.sqrt(v / (np.pi * row.t))[:, None] for idx, row in df.iterrows()]

    # Calculating errors
    df['u_t_error'] = [np.mean(np.abs(row.u_t - row.u_t_spline), axis=1) for idx, row in df.iterrows()]
    df['u_x_error'] = [np.mean(np.abs(row.u_x - row.u_x_spline) * (row.l**1 / row.u0), axis=1) for idx, row in df.iterrows()]
    df['u_xx_error'] = [np.mean(np.abs(row.u_xx - row.u_xx_spline) * (row.l**2 / row.u0), axis=1) for idx, row in df.iterrows()]
    df['u_xxx_error'] = [np.mean(np.abs(row.u_xxx - row.u_xxx_spline) * (row.l**3 / row.u0), axis=1) for idx, row in df.iterrows()]

    # Making some composite errors
    df['full_error'] = [(np.mean(np.abs((row.u_t - row.u_t_spline) / np.linalg.norm(row.u_t, axis=1, keepdims=True)) , axis=1) 
                        + np.mean(np.abs((row.u_x - row.u_x_spline) / np.linalg.norm(row.u_x, axis=1, keepdims=True)) , axis=1)
                        + np.mean(np.abs((row.u_xx - row.u_xx_spline) / np.linalg.norm(row.u_xx, axis=1, keepdims=True)) , axis=1)
                        + np.mean(np.abs((row.u_xxx - row.u_xxx_spline) / np.linalg.norm(row.u_xxx, axis=1, keepdims=True)) , axis=1)) 
                        for idx, row in df.iterrows()]
    df['deriv_error'] = [(np.mean(np.abs((row.u_x - row.u_x_spline) / np.linalg.norm(row.u_x, axis=1, keepdims=True)) , axis=1)
                        + np.mean(np.abs((row.u_xx - row.u_xx_spline) / np.linalg.norm(row.u_xx, axis=1, keepdims=True)) , axis=1)
                        + np.mean(np.abs((row.u_xxx - row.u_xxx_spline) / np.linalg.norm(row.u_xxx, axis=1, keepdims=True)) , axis=1)) 
                        for idx, row in df.iterrows()]

    # Make sure to throw away the edges
    df['coeffs'] = [fit(row.u[1:-1], row.u_t_spline[1:-1], row.u_x_spline[1:-1], row.u_xx_spline[1:-1], row.u_xxx_spline[1:-1]) for idx, row in df.iterrows()]
    df['coeffs_baseline'] = [fit(row.u[1:-1], row.u_t[1:-1], row.u_x[1:-1], row.u_xx[1:-1], row.u_xxx[1:-1]) for idx, row in df.iterrows()]

    df['correct'] = [correct_eq(row.coeffs) for idx, row in df.iterrows()]
    df['correct_baseline'] = [correct_eq(row.coeffs_baseline) for idx, row in df.iterrows()]
    
    # Fit per frame 
    '''
    df['frame_coeffs'] = [np.concatenate([fit(row.u[frame, :], 
                               row.u_t_spline[frame, :],
                               row.u_x_spline[frame, :], 
                               row.u_xx_spline[frame, :], 
                               row.u_xxx_spline[frame, :]) 
                               for frame in np.arange(row.t.size)], axis=1).T
                               for idx, row in df.iterrows()]
    df['frame_correct'] =[[correct_eq(row.frame_coeffs[frame, :][:, None]) for frame in np.arange(row.t.size)] for idx, row in df.iterrows()]
    '''
    return df