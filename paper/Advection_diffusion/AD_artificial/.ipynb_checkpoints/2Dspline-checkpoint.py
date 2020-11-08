def spline_diff2D(z, x, y, order, **spline_kwargs):
    ''' Fits 2D spline to 2D data. x and y are 1D arrays of coordinate grid, z is 2D array with dara.
    Good defaults for spline would be kx=4, ky=4, s=1e-2 with noise, 0.0 if no noise.'''
    spline = RectBivariateSpline(x, y, z, **spline_kwargs)
    return spline(x, y, dx=order, dy=order)
# Example usage
u_approx_spline = spline_diff2D(u_true, t, x, order=0, kx=4, ky=4, s=0.0) # approximation of data
u_approx_spline = spline_diff2D(u_true, t, x, order=3, kx=4, ky=4, s=0.0) # 3rd order deriv