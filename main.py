"""
Usage: for running an example, change
    points = np.array([[0.32, 0.3, 0.2],
                       [-0.3, 0.31, -0.01],
                       [-0.29, -0.3, -0.1],
                       [0.3, -0.28, 0.05],
                       [0.01, -0.02, -0.1]])

    connections = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4]]
    Ls = [1, 1, 1, 1, 0.7, 0.7, 0.7, 0.7]
    in the main function.

    To check the implementation details, check the function `build_catenary_network`
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed
import time
from scipy.optimize import root_scalar
import matplotlib
from scipy.spatial import distance
# matplotlib.use('tkagg')


class CatenaryNetwork:
    def __init__(self, points, connections, lengths, num_samples=5, initial_guess=None, instant_update=True, visualized=False):
        self.points = points
        self.connections = connections
        self.lengths = lengths
        if initial_guess is None:
            self.guess = [[0.5, (points[connection[0]][0] + points[connection[0]][0]) / 2,
                           (points[connection[0]][2] + points[connection[0]][2]) / 2] for i, connection in
                          enumerate(self.connections)]
        else:
            self.guess = initial_guess

        self.instant_update = instant_update
        self.num_samples = num_samples
        self._updated = False
        self.samples = np.zeros((self.num_samples*len(connections), 3))
        self.catenary_network_params = None
        self.visualized = visualized
        if self.visualized:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.view_init(elev=30, azim=30)
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")
            self.ax.set_title("3D Catenary Curve Fitting with Sampled Points and Transparent Curves")

        if self.instant_update:
            self.catenary_network_params = self._build_catenary_network(self.points, parallelize=False)
            for i, catenary_param in enumerate(self.catenary_network_params):
                self.samples[i*self.num_samples:(i+1)*self.num_samples, :] = catenary_param[6]

    def update(self, points):
        self._updated = False
        self.points = points
        if self.instant_update:
            self.catenary_network_params = self._build_catenary_network(self.points, parallelize=False)
            for i, catenary_param in enumerate(self.catenary_network_params):
                self.guess[i] = catenary_param[:3]
                self.samples[i*self.num_samples:(i+1)*self.num_samples, :] = catenary_param[6]
            self._updated = True

    def get_samples(self):
        if not self._updated:
            self.catenary_network_params = self._build_catenary_network(self.points, parallelize=False)
            for i, catenary_param in enumerate(self.catenary_network_params):
                self.guess[i] = catenary_param[:3]
                self.samples[i * self.num_samples:(i + 1) * self.num_samples, :] = catenary_param[6]

        self._updated = True
        return np.array(self.samples)

    def visualize(self, ax=None, color_curve="blue", color_point="black"):
        if not self.visualized:
            self.visualized = True
            if ax is None:
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(111, projection='3d')
                self.ax.view_init(elev=30, azim=30)
                self.ax.set_xlabel("X")
                self.ax.set_ylabel("Y")
                self.ax.set_zlabel("Z")
                self.ax.set_title("3D Catenary Curve Fitting with Sampled Points and Transparent Curves")
            else:
                self.ax = ax

        for i, connection in enumerate(self.connections):
            # Retrieve curve parameters and sampled points
            p1, p2 = points[connection[0], :], points[connection[1], :]
            c, x0, z0, length, rotation, translation, sampled_points = self.catenary_network_params[i]

            # Plot the full catenary curve with transparency (alpha=0.5)
            x_full = np.linspace(0, length, 100)  # High-resolution sampling
            z_full = c * np.cosh((x_full - x0) / c) + z0
            y_full = np.zeros_like(x_full)
            full_curve_local = np.vstack((x_full, y_full, z_full)).T
            full_curve_global = rotation.inv().apply(full_curve_local) + translation

            self.ax.plot(
                full_curve_global[:, 0],
                full_curve_global[:, 1],
                full_curve_global[:, 2],
                color=color_curve,
                alpha=0.5,
                label=f"Curve {i}"
            )

            # Plot the sampled points on the curve
            self.ax.scatter(
                sampled_points[:, 0],
                sampled_points[:, 1],
                sampled_points[:, 2],
                color=color_point,
                label=f"Sampled Points {i}"
            )


    # Functions
    def _transform_to_local_frame(self, p1, p2):
        """
        Compute the rotation matrix and translation to align p1 and p2 along the x-axis in a local frame across p1 and p2
        and contains z-axis
        :param p1: point 1 (3x1 ndarray)
        :param p2: point 2 (3x1 ndarray)
        :return: rotation, translation, p1_local, p2_local, distance between p1 and p2
        """
        # Direction vector from p1 to p2
        direction = np.array(p2) - np.array(p1)
        direction[2] = 0.0
        length = np.linalg.norm(direction)
        direction /= length  # Normalize direction

        # Rotation to align direction with x-axis
        x_axis = np.array([1, 0, 0])
        rotation_axis = np.cross(direction, x_axis)
        rotation_angle = np.arccos(np.dot(direction, x_axis))
        if np.linalg.norm(rotation_axis) > 1e-6:  # Avoid divide-by-zero
            rotation_axis /= np.linalg.norm(rotation_axis)
        else:
            rotation_axis = np.array([0, 0, 0])  # No rotation needed

        rotation = R.from_rotvec(rotation_angle * rotation_axis)

        # Transform points to local frame
        p1_local = np.array([0, 0, 0])  # Set p1 as the origin
        p2_local = rotation.apply(np.array(p2) - np.array(p1))

        return rotation, p1, p1_local, p2_local, length


    def _catenary_function(self, x, c, x0, z0):
        """
        Compute the 2D catenary curve in the local frame.
        :param x: x axis of the catenary curve in the local frame
        :param c, x0, z0: catenary curve parameters
        :return: z axis of the catenary curve in the local frame
        """
        return c * np.cosh((x - x0) / c) + z0


    def _compute_exact_arc_length(self, c, x0, x1, x2):
        """
        Compute the exact arc length of the catenary curve.
        """
        return c * (np.sinh((x2 - x0) / c) - np.sinh((x1 - x0) / c))


    def _gradient_arc_length(self, c, x0, x1, x2):
        """
        Compute the gradients of the exact arc length with respect to c and x0.
        """
        sinh_x2 = np.sinh((x2 - x0) / c)
        sinh_x1 = np.sinh((x1 - x0) / c)
        cosh_x2 = np.cosh((x2 - x0) / c)
        cosh_x1 = np.cosh((x1 - x0) / c)

        # Partial derivative w.r.t c
        dL_dc = sinh_x2 - sinh_x1 - (1 / c) * ((x2 - x0) * cosh_x2 - (x1 - x0) * cosh_x1)

        # Partial derivative w.r.t x0
        dL_dx0 = -cosh_x2 + cosh_x1

        return dL_dc, dL_dx0


    def _objective_with_gradient(self, params, x1, z1, x2, z2, L):
        """
        Objective function with gradients for optimization using exact arc length.
        """
        c, x0, z0 = params

        # Predicted z-values
        z1_pred = c * np.cosh((x1 - x0) / c) + z0
        z2_pred = c * np.cosh((x2 - x0) / c) + z0

        # Exact curve length
        L_cat = self._compute_exact_arc_length(c, x0, x1, x2)

        # Residuals
        error_z1 = z1 - z1_pred
        error_z2 = z2 - z2_pred
        error_length = L - L_cat

        # Objective function
        objective_value = error_z1**2 + error_z2**2 + error_length**2

        # Gradients
        dz1_dc = np.cosh((x1 - x0) / c) - (x1 - x0) / c * np.sinh((x1 - x0) / c)
        dz2_dc = np.cosh((x2 - x0) / c) - (x2 - x0) / c * np.sinh((x2 - x0) / c)

        dz1_dx0 = -np.sinh((x1 - x0) / c)
        dz2_dx0 = -np.sinh((x2 - x0) / c)

        dz1_dz0 = 1
        dz2_dz0 = 1

        # Gradients for the arc length
        dL_dc, dL_dx0 = self._gradient_arc_length(c, x0, x1, x2)

        # Total gradient components
        grad_c = -2 * error_z1 * dz1_dc - 2 * error_z2 * dz2_dc - 2 * error_length * dL_dc
        grad_x0 = -2 * error_z1 * dz1_dx0 - 2 * error_z2 * dz2_dx0 - 2 * error_length * dL_dx0
        grad_z0 = -2 * error_z1 * dz1_dz0 - 2 * error_z2 * dz2_dz0

        return objective_value, np.array([grad_c, grad_x0, grad_z0])


    def _fit_3d_catenary(self, p1, p2, L, initial_guess):
        """
        Fit a 3D catenary curve given two endpoints and cable length.
        """
        # Transform to local frame
        rotation, translation, p1_local, p2_local, length = self._transform_to_local_frame(p1, p2)
        x1, z1 = p1_local[0], p1_local[2]
        x2, z2 = p2_local[0], p2_local[2]

        # Perform optimization in 2D
        bounds = [(1e-2, None), (None, None), (None, None)]  # Ensure c > 0
        result = minimize(
            self._objective_with_gradient,
            initial_guess,
            args=(x1, z1, x2, z2, L),
            bounds=bounds,
            method='L-BFGS-B',
            jac=True,
            options={"maxiter": 1000, "disp": False},
        )

        c, x0, z0 = result.x

        return c, x0, z0, length, rotation, translation


    def _invert_arc_length(self, s_target, c, x0, x1, x2):
        """
        Find x-coordinate corresponding to a target arc length s_target using root_scalar.
        """
        result = root_scalar(
            lambda x: c * (np.sinh((x - x0) / c) - np.sinh((x1 - x0) / c)) - s_target,
            bracket=[x1, x2],  # Ensure the solution lies within x1 and x2
            method='brentq'
        )
        return result.root


    def _sample_points_on_catenary(self, c, x0, z0, x1, x2, num_points=5):
        """
        Sample points that are equally spaced along the arc length of the catenary curve.
        """
        # Compute total arc length
        total_arc_length = self._compute_exact_arc_length(c, x0, x1, x2)

        # Compute arc length fractions for equally spaced points
        arc_lengths = np.linspace(0, total_arc_length, num_points)

        # Invert arc length function for each target arc length
        x_samples = [self._invert_arc_length(s, c, x0, x1, x2) for s in arc_lengths]

        # Vectorized z-coordinates
        x_samples = np.array(x_samples)
        z_samples = c * np.cosh((x_samples - x0) / c) + z0

        return np.array([x_samples, np.zeros_like(x_samples), z_samples]).T


    def _sampling_3d_catenary_points(self, c, x0, z0, length, rotation, translation, x1, x2, num_points=5):
        """
        Sample points on the 3D catenary curve that are equally spaced along its arc length.
        """
        # Sample points in the local frame
        points_local = self._sample_points_on_catenary(c, x0, z0, x1, x2, num_points)

        # Transform back to the global frame
        points_global = rotation.inv().apply(points_local) + translation
        return points_global


    def _build_catenary_network(self, points, parallelize=False):
        """
        Build a catenary network using fitting and sample equally spaced points on each curve.
        With small numbers of points and num_samples, it is NOT worth it enabling parallelize = True.
        :param points: quadrotor positions in the world frame
        :param connections: which quadrotors are connected by catenary curves
        :param parallelize: option to enable parallel computation
        :param Ls: curve lengths of the catenary curves
        :param num_samples: number of equally spaced samples on each catenary curve
        :return: list([c, x0, z0, length, rotation, translation, sampled_points])
            for each element in the list
            c, x0, z0: catenary parameters in its local 2D frame
            length: distance between the two endpoints
            rotation, translation: transformation from the local 2D frame to the world frame
            sampled_points: equally spaced sample points on the catenary curve in the world frame.
        """
        estimate_L = (self.lengths is None)

        def process_connection(i, connection, guess):
            p1, p2 = points[connection[0], :], points[connection[1], :]

            # Estimate or use provided cable length
            l = np.linalg.norm(p2 - p1) * 1.5 if estimate_L else self.lengths[i]

            c, x0, z0, length, rotation, translation = self._fit_3d_catenary(p1, p2, l, self.guess[i])

            # Sample equally spaced points along the curve
            sampled_points = self._sampling_3d_catenary_points(c, x0, z0, length, rotation, translation, 0, length, self.num_samples)

            return c, x0, z0, length, rotation, translation, sampled_points

        # Parallelize the fitting process
        if parallelize:
            catenary_network_params = Parallel(n_jobs=-1)(
                delayed(process_connection)(i, connection, self.guess) for i, connection in enumerate(self.connections)
            )
        else:
            catenary_network_params = [process_connection(i, connection, self.guess) for i, connection in enumerate(self.connections)]

        return catenary_network_params


def average_hausdorff_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """
    Computes the average Hausdorff distance between two sets of 3D points.

    Parameters:
        points_a (np.ndarray): An ndarray of shape (N, 3) representing the first set of 3D points.
        points_b (np.ndarray): An ndarray of shape (M, 3) representing the second set of 3D points.

    Returns:
        float: The average Hausdorff distance between the two sets of points.
    """
    if points_a.shape[1] != 3 or points_b.shape[1] != 3:
        raise ValueError("Both input arrays must have 3 columns representing 3D points.")

    # Compute pairwise distances
    d_matrix = distance.cdist(points_a, points_b)

    # Compute directed distances
    d_ab = np.mean(np.min(d_matrix, axis=1))  # Average of minimum distances from A to B
    d_ba = np.mean(np.min(d_matrix, axis=0))  # Average of minimum distances from B to A

    # Average Hausdorff distance
    return (d_ab + d_ba) / 2


def generate_mesh_intersections(x0, y0, z0, size=9, spacing=1.0):
    """
    Generate a 3D mesh of intersection points centered at (x0, y0, z0).

    Parameters:
        x0 (float): x-coordinate of the mesh center.
        y0 (float): y-coordinate of the mesh center.
        z0 (float): z-coordinate of the mesh center.
        size (int): Number of points along each dimension (default 9x9 grid).
        spacing (float): Distance between adjacent points (default 1.0).

    Returns:
        np.ndarray: Array of shape (size*size, 3) representing mesh intersection points.
    """
    if size % 2 == 0:
        raise ValueError("Size must be an odd number to center the grid at (x0, y0, z0).")

    half_size = size // 2
    x_coords = np.linspace(x0 - half_size * spacing, x0 + half_size * spacing, size)
    y_coords = np.linspace(y0 - half_size * spacing, y0 + half_size * spacing, size)

    # Create the grid
    xv, yv = np.meshgrid(x_coords, y_coords)
    zv = np.full_like(xv, z0)  # Add z-coordinate
    intersections = np.column_stack((xv.ravel(), yv.ravel(), zv.ravel()))
    return intersections


if __name__ == "__main__":
    """ Desired mesh, defined by the quadrotor positions """
    # desired_points = np.array([[0.49, 0.51, 0.03],  # 0
    #                            [-0.5, 0.5, 0.02],  # 1
    #                            [-0.5, -0.5, 0.0],  # 2
    #                            [0.5, -0.5, -0.01],  # 3
    #                            [0.0, 0.0, 0.01]])  # 4
    #
    # # which quadrotors are connected, in terms of their indices
    # desired_connections = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4]]
    #
    # # the lengths of the connections
    # desired_Ls = [1.0, 1.0, 1.0, 1.0, 0.707, 0.71, 0.71, 0.71]
    #
    # # get some initial guesses
    # initial_guess = None
    #
    # desired_catenary_network = CatenaryNetwork(desired_points, desired_connections, desired_Ls)
    # desired_sample_positions = desired_catenary_network.get_samples()
    desired_mesh = generate_mesh_intersections(1, 1, 0.5, size=9, spacing=0.125)

    """ Current quadrotor positions """
    points = np.array([[0.32, 0.3, 0.2],        # 0
                       [-0.3, 0.31, -0.01],     # 1
                       [-0.29, -0.3, -0.1],     # 2
                       [0.3, -0.28, 0.05],      # 3
                       [0.01, -0.02, -0.1]])    # 4

    # which quadrotors are connected, in terms of their indices
    connections = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4]]

    # the lengths of the connections
    Ls = [1, 1, 1, 1, 0.7, 0.7, 0.7, 0.7]

    # get some initial guesses
    initial_guess = None  # [0.5, (p1[0] + p2[0]) / 2, (p1[2] + p2[2]) / 2]

    catenary_network = CatenaryNetwork(points, connections, Ls)

    for i in range(1000):
        # Fit the 3D catenary network and sample points
        catenary_network.visualize()
        catenary_network.ax.scatter(desired_mesh[:, 0],
                                    desired_mesh[:, 1],
                                    desired_mesh[:, 2],
                                    color="red")
        # desired_catenary_network.visualize(catenary_network.ax, color_curve="red", color_point="black")
        plt.pause(0.0001)
        time_start = time.time()
        catenary_network.update(points)
        sample_points = catenary_network.get_samples()
        print("error", average_hausdorff_distance(desired_mesh, sample_points))
        print("Time elapsed for fitting the curves and calculating the error:", time.time() - time_start)
        points += np.random.normal(loc=0.0, scale=0.005, size=points.shape)
        catenary_network.ax.clear()