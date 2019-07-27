import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


PLOT_XY = True
MARKER = '.'


# Generate a cubic Hermite spline from a key points.
# Key points: [[x0,y0],[x1,y1],[x2,y2],...].
class TCubicHermiteSpline:
    FINITE_DIFF = 0  # Tangent method: finite difference method
    CARDINAL = 1  # Tangent method: Cardinal spline (c is used)
    ZERO = 0  # End tangent: zero
    GRAD = 1  # End tangent: gradient (m is used)
   
    class TKeyPoint:
        X = 0.0  # Input
        Y = 0.0  # Output
        M = 0.0  # Gradient

        def __str__(self):
            return '[' + str(self.X) + ', ' + str(self.Y) + ', ' + str(self.M) + ']'

    def __init__(self):
        self.idx_prev = 0

    def find_idx(self, x, idx_prev=0):
        idx = idx_prev
        if idx >= len(self.KeyPts):
            idx = len(self.KeyPts) - 1
        while idx + 1 < len(self.KeyPts) and x > self.KeyPts[idx + 1].X:
            idx += 1
        while idx >= 0 and x < self.KeyPts[idx].X:
            idx -= 1
        return idx

    # Return interpolated value at t
    def evaluate(self, x):
        idx = self.find_idx(x, self.idx_prev)
        if abs(x - self.KeyPts[-1].X) < 1.0e-6:
            idx = len(self.KeyPts) - 2
        if idx < 0 or idx >= len(self.KeyPts) - 1:
            print('WARNING: Given t= %f is out of the key points (index: %i)' % (x, idx))
            if idx < 0:
                idx = 0
                x = self.KeyPts[0].X
            else:
                idx = len(self.KeyPts) - 2
                x = self.KeyPts[-1].X

        h00 = lambda t: t * t * (2.0 * t - 3.0) + 1.0
        h10 = lambda t: t * (t * (t - 2.0) + 1.0)
        h01 = lambda t: t * t * (-2.0 * t + 3.0)
        h11 = lambda t: t * t * (t - 1.0)

        self.idx_prev = idx
        p0 = self.KeyPts[idx]
        p1 = self.KeyPts[idx+1]
        xr = (x - p0.X) / (p1.X - p0.X)
        return h00(xr) * p0.Y + h10(xr) * (p1.X - p0.X) * p0.M + h01(xr) * p1.Y + h11(xr) * (p1.X - p0.X) * p1.M

    def Initialize(self, data, tan_method=CARDINAL, end_tan=GRAD, c=0.0, m=1.0):
        self.KeyPts = [self.TKeyPoint() for i in range(len(data))]
        for idx in range(len(data)):
            self.KeyPts[idx].X = data[idx][0]
            self.KeyPts[idx].Y = data[idx][1]

        grad = lambda idx1, idx2: (self.KeyPts[idx2].Y - self.KeyPts[idx1].Y) / (
                self.KeyPts[idx2].X - self.KeyPts[idx1].X)

        for idx in range(1, len(self.KeyPts) - 1):
            self.KeyPts[idx].M = 0.0
        if tan_method == self.FINITE_DIFF:
            for idx in range(1, len(self.KeyPts) - 1):
                self.KeyPts[idx].M = 0.5 * grad(idx, idx + 1) + 0.5 * grad(idx - 1, idx)
        elif tan_method == self.CARDINAL:
            for idx in range(1, len(self.KeyPts) - 1):
                self.KeyPts[idx].M = (1.0 - c) * grad(idx - 1, idx + 1)
        else:
            raise NotImplementedError

        if end_tan == self.ZERO:
            self.KeyPts[0].M = 0.0
            self.KeyPts[-1].M = 0.0
        elif end_tan == self.GRAD:
            self.KeyPts[0].M = m * grad(0, 1)
            self.KeyPts[-1].M = m * grad(-2, -1)
        else:
            raise NotImplementedError


#  Hermite Cubic spline 2D
def get_interp_cubic_hermite_spline(distances, x_array, y_array, distances_new, tan_method, c=None):
    # Interpolation de la ligne avec CHS?
    spline_x = TCubicHermiteSpline()  #  1D along x
    spline_y = TCubicHermiteSpline()  #  1D along y

    spline_x.Initialize(np.vstack((distances, x_array)).T, tan_method=tan_method, c=c)
    spline_y.Initialize(np.vstack((distances, y_array)).T, tan_method=tan_method, c=c)

    xx = []
    yy = []
    
    for dist in distances_new:
        x = spline_x.evaluate(dist)
        y = spline_y.evaluate(dist)
        xx.append(x)
        yy.append(y)
    return xx, yy


def get_interp_multiple1d(distances, x_array, y_array, distances_new, method='linear'):
    interpolator = interp1d(distances, np.vstack((x_array, y_array)).T, kind=method, axis=0)
    new = interpolator(distances_new)
    return new[:, 0], new[:, 1]

# plot x and y
def plot_spline(spline, dots_per_second=30):
    total_duration = spline.grid[-1] - spline.grid[0]
    times = spline.grid[0] + np.arange(int(total_duration * dots_per_second) + 1) / dots_per_second
    plt.plot(*spline(times).T, marker=MARKER, label='Spline')
    plt.axis('equal')


def plot_interpolation(i):
    """
    @param i <int>: 0 or -1
    """
    # list: file .i2s, .xy, .txt, ...., etc
    x_array = np.zeros(len(list))
    y_array = np.zeros(len(list))
    distances = np.zeros(len(list))

    # curvlinear distance calculate
    x_old, y_old = None, None
    for j, profile in enumerate(list):
        x = profile.x[i]
        y = profile.y[i]
        x_array[j] = x
        y_array[j] = y
        if j != 0:
            distances[j] = distances[j - 1] + math.sqrt((x - x_old) ** 2 + (y - y_old) ** 2)
        x_old, y_old = x, y

    distances_new = np.arange(distances.min(), distances.max(), 0.1)
    distances_new = np.unique(np.concatenate((distances_new, distances)))

    if PLOT_XY:
        # initialize a figure and add initializes points
        plt.plot(x_array, y_array, linestyle=':', color='lightgrey', marker='o', markeredgecolor='black',
                 label="Initial points")
        plt.axes().set_aspect('equal')
        plt.xlabel('X (m)')
        plt.xlabel('Y (m)')
    else:
        plt.plot(distances, np.zeros(len(distances)), linestyle=':', color='lightgrey', marker='x',
                 markeredgecolor='black',
                 label="Initial profiles")
        plt.xlabel('Distance (m)')
        plt.xlabel('Ecart (m)')

    # reference lines
    x_ref, y_ref = get_interp_multiple1d(distances, x_array, y_array, distances_new, 'linear')

    def plot_interp(x, y, label):
        if PLOT_XY:
            plt.plot(x, y, label=label, marker=MARKER)
        else:
            ecarts = np.sqrt((x_ref - x)**2 + (y_ref - y)**2)  # in meters
            rmse = math.sqrt(np.power(ecarts, 2).mean())  # mean() : moyenne
            print(rmse)
            plt.plot(distances_new, ecarts, label=label)

    plot_interp(x_ref, y_ref, label="Linear interpolation")

    plot_interp(*get_interp_cubic_hermite_spline(distances, x_array, y_array, distances_new, TCubicHermiteSpline.CARDINAL, 0.0),
                "Cardinal spline (c=0.0)")

    plot_interp(*get_interp_cubic_hermite_spline(distances, x_array, y_array, distances_new, TCubicHermiteSpline.CARDINAL, 0.5),
                "Cardinal spline (c=0.5)")

    plot_interp(*get_interp_cubic_hermite_spline(distances, x_array, y_array, distances_new, TCubicHermiteSpline.CARDINAL, 1.0),
                "Cardinal spline (c=1.0)")  # = Linear interpolation

    plot_interp(*get_interp_cubic_hermite_spline(distances, x_array, y_array, distances_new, TCubicHermiteSpline.FINITE_DIFF),
                "Finite-difference")

  

    X2, Y2 = get_interp_cubic_hermite_spline(distances, x_array, y_array, distances_new, TCubicHermiteSpline.FINITE_DIFF)
    with open('file.i2s', 'w') as fileout:
        for valx, valy in zip(X2, Y2):
            fileout.write('%f %f \n' % (valx, valy))

    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    plot_interpolation(0)
    plot_interpolation(-1)
