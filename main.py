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
from scipy.optimize import minimize, least_squares
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
plt.rc('font', family='serif')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from scipy.spatial import Delaunay


class CatenaryNetwork:
    def __init__(self, points, connections, lengths, surfaces, num_samples=5, initial_guess=None, instant_update=True, visualized=False):
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
        self.surfaces = surfaces
        self.surface_params = [[1.0, 0., 0., 0.,] for _ in self.surfaces]
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


    def get_surfaces(self):
        # Define the 3D catenary surface equation
        sample_points = self.get_samples()

        # for estimating the surface
        for i, surface in enumerate(self.surfaces):
            surface_points = np.zeros((len(surface)*num_samples, 3))
            for j, curve_id in enumerate(surface):
                surface_points[j*num_samples:(j+1)*num_samples, :] = sample_points[curve_id*num_samples:(curve_id + 1)*num_samples, :]

            initial_guess = self.surface_params[i]
            x = surface_points[:, 0]
            y = surface_points[:, 1]
            z = surface_points[:, 2]
            result = least_squares(self._residuals, initial_guess, args=(x, y, z))
            self.surface_params[i] = result.x

        return self.surface_params

    def _catenary_surface(self, params, x, y):
        a, x0, y0, z0 = params
        return z0 + a * np.cosh(np.sqrt((x - x0) ** 2 + (y - y0) ** 2) / a)

    # Define the residual function for least squares
    def _residuals(self, params, x, y, z):
        return z - self._catenary_surface(params, x, y)


    def visualize(self, ax=None, color_curve="blue", color_point="black"):
        if not self.visualized:
            self.visualized = True
            if ax is None:
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(111, projection='3d')
                self.ax.view_init(elev=30, azim=60)
                self.ax.set_xlabel("X")
                self.ax.set_ylabel("Y")
                self.ax.set_zlabel("Z")
                self.ax.set_title("3D Catenary Curve Fitting with Sampled Points and Transparent Curves")
            else:
                self.ax = ax

        for i, connection in enumerate(self.connections):
            # Retrieve curve parameters and sampled points
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
                alpha=0.75,
                label=f"Curve {i}",
                linewidth=2
            )

            # Plot the sampled points on the curve
            self.ax.scatter(
                sampled_points[1:self.num_samples-1, 0],
                sampled_points[1:self.num_samples-1, 1],
                sampled_points[1:self.num_samples-1, 2],
                color=color_point,
                label=f"Sampled Points {i}"
            )

            self.ax.scatter(
                sampled_points[0, 0],
                sampled_points[0, 1],
                sampled_points[0, 2],
                color="red",
                label=f"Endpoints {i}"
            )

            self.ax.scatter(
                sampled_points[-1, 0],
                sampled_points[-1, 1],
                sampled_points[-1, 2],
                color="red",
                label=f"Endpoints {i}"
            )

        # Create a mesh grid for the fitted surface
        for i, surface_param in enumerate(self.surface_params):
            points = set()
            connection_ids = self.surfaces[i]
            for connection_id in connection_ids:
                point_ids = self.connections[connection_id]
                for point_id in point_ids:
                    points.add(point_id)

            p1 = self.points[points.pop()][:2]
            p2 = self.points[points.pop()][:2]
            p3 = self.points[points.pop()][:2]
            x_mesh, y_mesh = self._generate_mesh_in_triangle(p1, p2, p3, 10)

            # Compute the fitted z-values
            z_mesh = self._catenary_surface(surface_param, x_mesh, y_mesh)

            # Plot the fitted catenary surface
            self.ax.plot_trisurf(x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten(), cmap='magma', alpha=1.0, edgecolor='none')


    def _generate_mesh_in_triangle(self, p1, p2, p3, resolution=10):
        """
        Generate a mesh grid enclosed by a triangle defined by three 2D points.

        Parameters:
            p1, p2, p3 (tuple): Vertices of the triangle, each represented as (x, y).
            resolution (int): Number of points along each side of the triangle.

        Returns:
            tuple: Two 2D arrays (x_mesh, y_mesh) representing the mesh grid points.
        """
        # Create a bounding grid
        x_min = min(p1[0], p2[0], p3[0])
        x_max = max(p1[0], p2[0], p3[0])
        y_min = min(p1[1], p2[1], p3[1])
        y_max = max(p1[1], p2[1], p3[1])

        x_grid = np.linspace(x_min, x_max, resolution)
        y_grid = np.linspace(y_min, y_max, resolution)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

        # Flatten the mesh grids
        x_flat = x_mesh.flatten()
        y_flat = y_mesh.flatten()

        # Create a matrix of points
        points = np.vstack((x_flat, y_flat)).T

        # Barycentric test to check if points are inside the triangle
        def barycentric_test(point, p1, p2, p3):
            # Convert triangle vertices and point to numpy arrays
            v0 = np.array(p3) - np.array(p1)
            v1 = np.array(p2) - np.array(p1)
            v2 = np.array(point) - np.array(p1)

            # Compute dot products
            dot00 = np.dot(v0, v0)
            dot01 = np.dot(v0, v1)
            dot02 = np.dot(v0, v2)
            dot11 = np.dot(v1, v1)
            dot12 = np.dot(v1, v2)

            # Compute barycentric coordinates
            denom = dot00 * dot11 - dot01 * dot01
            if denom == 0:
                return False  # Degenerate triangle

            u = (dot11 * dot02 - dot01 * dot12) / denom
            v = (dot00 * dot12 - dot01 * dot02) / denom

            # Check if point is inside the triangle
            return (u > 1e-3) and (v > 1e-3) and (u + v < 1)

        inside = np.array([barycentric_test(point, p1, p2, p3) for point in points])

        # Filter points inside the triangle
        x_inside = x_flat[inside]
        y_inside = y_flat[inside]

        return x_inside.reshape(-1, 1), y_inside.reshape(-1, 1)


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
        Perform batch operations to compute catenary parameters and sampled points efficiently.
        """
        num_connections = len(self.connections)
        num_samples = self.num_samples
        samples_per_connection = np.zeros((num_connections, num_samples, 3))

        # Step 1: Transform to local frames for all connections
        rotations = []
        translations = []
        local_frames = []
        lengths = []
        for connection in self.connections:
            p1, p2 = points[connection[0]], points[connection[1]]
            rotation, translation, p1_local, p2_local, length = self._transform_to_local_frame(p1, p2)
            rotations.append(rotation)
            translations.append(translation)
            local_frames.append((p1_local, p2_local))
            lengths.append(length)

        # Step 2: Optimize catenary parameters for all connections
        catenary_params = []
        for i, (p1_local, p2_local) in enumerate(local_frames):
            x1, z1 = p1_local[0], p1_local[2]
            x2, z2 = p2_local[0], p2_local[2]
            length = self.lengths[i] if self.lengths is not None else np.linalg.norm(
                points[self.connections[i][1]] - points[self.connections[i][0]])

            # Perform optimization
            result = minimize(
                self._objective_with_gradient,
                self.guess[i],
                args=(x1, z1, x2, z2, length),
                bounds=[(1e-2, None), (None, None), (None, None)],  # Ensure c > 0
                method='L-BFGS-B',
                jac=True,
                options={"maxiter": 1000, "disp": False},
            )
            catenary_params.append(result.x)  # Append optimized [c, x0, z0]

        # Step 3: Sample points on catenary curves for all connections
        for i, (c_params, rotation, translation, (p1_local, p2_local)) in enumerate(
                zip(catenary_params, rotations, translations, local_frames)):
            c, x0, z0 = c_params
            x1, x2 = p1_local[0], p2_local[0]

            # Compute sampled points in local frame
            total_arc_length = self._compute_exact_arc_length(c, x0, x1, x2)
            arc_lengths = np.linspace(0, total_arc_length, num_samples)
            x_samples = [self._invert_arc_length(s, c, x0, x1, x2) for s in arc_lengths]
            z_samples = c * np.cosh((np.array(x_samples) - x0) / c) + z0
            local_points = np.array([x_samples, np.zeros_like(x_samples), z_samples]).T

            # Transform sampled points back to global frame
            global_points = rotation.inv().apply(local_points) + translation
            samples_per_connection[i] = global_points

        # Combine results into a structured output
        catenary_network_params = [
            (
            c_params[0], c_params[1], c_params[2], lengths[i], rotations[i], translations[i], samples_per_connection[i])
            for i, c_params in enumerate(catenary_params)
        ]

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


def generate_planar_graph(points):
    """
    Generate a planar graph from 2D point positions using Delaunay triangulation.

    Parameters:
        points: A list of 2D points, where each point is represented as (x, y).

    Returns:
        tuple: A tuple containing:
            - list of tuple: A list of edges, where each edge is represented by the indices of the points it connects.
            - list of list: A list of triangle surfaces, where each surface is represented as [e1, e2, e3],
              with ei being the index of the edge in the edge list.
    """
    # Convert the list of points to a NumPy array
    points = np.array(points)

    # Perform Delaunay triangulation
    triangulation = Delaunay(points)

    # Extract edges from the simplices (triangles)
    simplices = triangulation.simplices
    edges = np.vstack([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]])

    # Sort each edge to ensure consistent representation
    edges = np.sort(edges, axis=1)

    # Remove duplicate edges
    edges = np.unique(edges, axis=0)

    # Create a mapping from edges to their indices
    edge_to_index = {tuple(edge): i for i, edge in enumerate(edges)}

    # Extract triangle surfaces as edge indices
    triangle_surfaces = []
    for simplex in simplices:
        triangle_edges = [
            edge_to_index[tuple(sorted([simplex[0], simplex[1]]))],
            edge_to_index[tuple(sorted([simplex[1], simplex[2]]))],
            edge_to_index[tuple(sorted([simplex[2], simplex[0]]))]
        ]
        triangle_surfaces.append(triangle_edges)

    return edges, triangle_surfaces


if __name__ == "__main__":
    desired_mesh = generate_mesh_intersections(0, 0, 0.0, size=9, spacing=0.125)

    """ Current quadrotor positions """
    points = np.array([[0.32, 0.3, -0.3],        # 0
                       [-0.3, 0.31, 0.2],      # 1
                       [-0.29, -0.3, -0.2],    # 2
                       [0.3, -0.28, 0.3]])      # 3
                       # [0.01, -0.02, -0.05],    # 4
                       # []]

    connections, surfaces = generate_planar_graph(points[:, :2])

    # this is an estimation of the curve lengths. It will improve the performance if we know them
    Ls = [1.2*np.linalg.norm(points[point_ids[0], :] - points[point_ids[1], :]) for point_ids in connections]
    # Ls = [1, 1, 1, 1.5, 1.]
    # print("Edges:", edges, "Surfs", surfs)

    # exit(0)
    # which quadrotors are connected, in terms of their indices
    # connections = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4]]

    # which curves enclose a surface
    # surfaces = [[0, 4, 5], [1, 5, 6], [2, 6, 7], [3, 7, 4]]

    # the lengths of the connections
    # Ls = [1.2, 1.2, 1.2, 1.2, 0.75, 0.75, 0.75, 0.75]

    # get some initial guesses
    initial_guess = None  # [0.5, (p1[0] + p2[0]) / 2, (p1[2] + p2[2]) / 2]

    num_samples = 5
    catenary_network = CatenaryNetwork(points, connections, Ls, surfaces, num_samples=num_samples)

    for i in range(1000):
        # Fit the 3D catenary network and sample points
        catenary_network.visualize()
        plt.pause(0.0001)
        time_start = time.time()
        catenary_network.update(points)
        sample_points = catenary_network.get_samples()
        catenary_network.get_surfaces()
        print("error", average_hausdorff_distance(desired_mesh, sample_points))
        print("Time elapsed for fitting the curves and calculating the error:", time.time() - time_start)
        points += np.random.normal(loc=0.0, scale=0.002, size=points.shape)
        # if i == 5:
        #     plt.pause(100000)
        catenary_network.ax.clear()