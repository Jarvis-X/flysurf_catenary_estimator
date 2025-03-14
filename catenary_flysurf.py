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
import matplotlib.colors as mcolors

# Define a sequence of colors from green -> gold -> red
christmas_colors = [
    "#006400",  # DarkGreen
    "#FFFFFF",  # White
    "#8B0000"  # DarkRed
]
# Create a linear segmented colormap from these colors
christmas_cmap = mcolors.LinearSegmentedColormap.from_list("christmas", christmas_colors, N=256)


def batch_surf_sampling(intersections, surf_info):
    # Convert input data to numpy arrays
    points = np.array(intersections)  # Shape (N, 2)
    surf_corners_list = [surf[0] for surf in surf_info]
    surf_params_list = [surf[1] for surf in surf_info]
    surf_params_all = np.array(surf_params_list)  # Shape (M, 4)
    surfaces = np.array(surf_corners_list)[:, :, :2]  # Shape (M, 3, 2)

    M = surfaces.shape[0]
    N = points.shape[0]

    # Extract triangle vertices for all surfaces
    p1 = surfaces[:, 0, :]  # (M, 2)
    p2 = surfaces[:, 1, :]
    p3 = surfaces[:, 2, :]

    # Compute vectors for barycentric coordinates
    v0 = p3 - p1  # (M, 2)
    v1 = p2 - p1  # (M, 2)
    points_exp = points[:, np.newaxis, :]  # (N, 1, 2)
    p1_exp = p1[np.newaxis, :, :]  # (1, M, 2)
    v2 = points_exp - p1_exp  # (N, M, 2)

    # Calculate dot products
    dot00 = np.sum(v0 * v0, axis=1)  # (M,)
    dot01 = np.sum(v0 * v1, axis=1)  # (M,)
    dot02 = np.sum(v0[np.newaxis, :, :] * v2, axis=2)  # (N, M)
    dot11 = np.sum(v1 * v1, axis=1)  # (M,)
    dot12 = np.sum(v1[np.newaxis, :, :] * v2, axis=2)  # (N, M)

    denom = dot00 * dot11 - dot01 ** 2  # (M,)
    denom_nonzero = denom != 0  # (M,)

    # Barycentric coordinates
    valid_mask = denom_nonzero[np.newaxis, :]  # Shape (1, M) -> broadcasts to (N, M)

    # Compute u and v using broadcasting with np.divide's where parameter
    u = np.zeros((N, M))
    v = np.zeros((N, M))
    np.divide(
        (dot11 * dot02) - (dot01 * dot12),
        denom,
        where=valid_mask,
        out=u
    )
    np.divide(
        (dot00 * dot12) - (dot01 * dot02),
        denom,
        where=valid_mask,
        out=v
    )

    # Inside check now uses broadcasted valid_mask
    is_inside = (u >= 0) & (v >= 0) & ((u + v) <= 1) & valid_mask

    # Distance calculation to edges
    def edge_distance(A, B):
        AB = B - A
        AP = points_exp - A[np.newaxis, :, :]
        AB_dot = np.sum(AB * AB, axis=1)
        safe_AB_dot = np.where(AB_dot == 0, 1e-10, AB_dot)
        AP_dot_AB = np.sum(AP * AB[np.newaxis, :, :], axis=2)
        t = AP_dot_AB / safe_AB_dot[np.newaxis, :]
        t_clamped = np.clip(t, 0.0, 1.0)
        closest = A[np.newaxis, :, :] + t_clamped[..., np.newaxis] * AB[np.newaxis, :, :]
        return np.linalg.norm(points_exp - closest, axis=2)

    d1 = edge_distance(p1, p2)
    d2 = edge_distance(p2, p3)
    d3 = edge_distance(p3, p1)
    min_dist = np.minimum(np.minimum(d1, d2), d3)

    # Combine distances with inside check
    distances = np.where(is_inside, 0.0, min_dist)

    # Find closest surface for each point
    min_indices = np.argmin(distances, axis=1)
    selected_params = surf_params_all[min_indices]

    # Compute z-values using the closest parameters
    z_vals = [
        flysurf._catenary_surface(tuple(params), point[0], point[1])
        for params, point in zip(selected_params, points)
    ]

    return z_vals


def barycentric_test_with_distance(point, p1, p2, p3):
    # Convert all points to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    point = np.array(point)

    v0 = p3 - p1
    v1 = p2 - p1
    v2 = point - p1

    # Compute dot products
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    denom = dot00 * dot11 - dot01 ** 2
    is_inside = False

    if denom != 0:
        # Compute barycentric coordinates
        u = (dot11 * dot02 - dot01 * dot12) / denom
        v = (dot00 * dot12 - dot01 * dot02) / denom
        is_inside = (u >= 0) and (v >= 0) and (u + v <= 1)

    if is_inside:
        return 0.0  # Point is inside the triangle
    else:
        # Helper function to compute distance from a point to a line segment
        def distance_to_segment(P, A, B):
            AP = P - A
            AB = B - A
            t = np.dot(AP, AB) / np.dot(AB, AB)
            t_clamped = np.clip(t, 0.0, 1.0)
            closest_point = A + t_clamped * AB
            return np.linalg.norm(P - closest_point)

        # Compute distances to all three edges of the triangle
        d1 = distance_to_segment(point, p1, p2)
        d2 = distance_to_segment(point, p2, p3)
        d3 = distance_to_segment(point, p3, p1)

        return min(d1, d2, d3)  # Minimum distance to the triangle


class CatenarySurfaceOptimizer:
    def __init__(self, surface_points):
        self.surface_points = surface_points
        self.x = surface_points[:, 0]
        self.y = surface_points[:, 1]
        self.z = surface_points[:, 2]

    def _catenary_surface(self, params):
        """Catenary surface model: z0 + a*cosh(r/a) where r=√[(x-x0)² + (y-y0)²]"""
        a, x0, y0, z0 = params
        r = np.sqrt((self.x - x0) ** 2 + (self.y - y0) ** 2)
        return z0 + a * np.cosh(r / a)

    def objective(self, params):
        """Sum of squared residuals objective function"""
        scaling = 0.1
        model = scaling*self._catenary_surface(params)
        return np.sum((scaling*self.z - model) ** 2)

    def fit(self, initial_guess):
        """Optimization using minimize without gradient"""
        # Enforce a > 0 using bounds
        bounds = [
            (1e-3, None),  # a must be positive
            (None, None),  # x0
            (None, None),  # y0
            (None, None)  # z0
        ]

        result = minimize(
            fun=self.objective,
            x0=initial_guess,
            method='L-BFGS-B',  # Works well even with numerical gradients
            bounds=bounds,
            options={'disp': False, 'maxiter': 1000}
        )
        return result


class CatenaryFlySurf:
    def __init__(self, lc, lr, l_cell, num_sample_per_curve=10):
        self.log_catenary_time = []
        self.log_surface_time = []
        self.lc = lc
        self.lr = lr
        self.num_points = lc * lr
        self.cell_length = l_cell
        self.num_samples = num_sample_per_curve
        self.active_ridge = None
        self.active_surface = None

        # stores catenary-related data based on vertex indices
        self.catenary_curve_params = dict()  # [(Vi, Vj)]: [curve length, last stored c x y, other data]
        self.catenary_surface_params = dict()  #

        for i in range(self.num_points):
            for j in range(i + 1, self.num_points):
                i1, j1 = self._index2coord(i)
                i2, j2 = self._index2coord(j)
                self.catenary_curve_params[tuple(sorted((i, j)))] = [
                    np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2) * self.cell_length, [0.1, 0, 0], None]

        """ The following was a bit too much (4 Billion elements w/ lc = lr = 9) """
        # max_num_curves = len(self.catenary_curve_params)
        # for i in range(max_num_curves):
        #     for j in range(i+1, max_num_curves):
        #         for k in range(j+1, max_num_curves):
        #             self.catenary_surface_params[tuple(sorted((i, j, k)))] = None
        """ Let's do it dynamically for now """

    def update(self, points_coord, points_position):
        """
        update the catenary estimation of flysurf given new points and the associated coordinate on the mesh
        Parameters:

        :param points_coord: ndarray that assigns the mesh coordinates (maybe no need to specify each time, we'll see)
        :param points_position: ndarray that specifies the position of the lifting points
        """
        # check the discrete distance from the center of the mesh to the lifting point
        dist = np.sum(np.abs(points_coord - (points_coord.min(axis=0) + points_coord.max(axis=0)) / 2), axis=1)
        # normalize the height between 0 and 1
        if abs(points_position[:, 2].max() - points_position[:, 2].min()) < 1e-6:
            height_norm = np.zeros_like(dist)
        else:
            height_norm = (points_position[:, 2] - points_position[:, 2].min()) / (
                        points_position[:, 2].max() - points_position[:, 2].min())
        order_of_points = np.argsort(dist - height_norm)

        points = points_position[order_of_points]
        network_coords = points_coord[order_of_points]
        # print(network_coords)

        # edges = self._generate_planar_graph(network_coords, points)
        edges = self._generate_planar_graph_delaunay(network_coords, points)
        self.active_ridge = edges
        time_start_build_catenary_network = time.time()
        self._build_catenary_network(parallelize=False)
        self.log_catenary_time.append(time.time() - time_start_build_catenary_network)

        num_samples = self.num_samples

        # for estimating the surface
        time_start_estimating_surface = time.time()
        for i, surface in enumerate(self.active_surface):
            num_vertices = len(surface)
            num_edges = num_vertices
            surface_points = np.zeros((num_edges * num_samples, 3))
            index = 0
            for j in range(num_vertices):
                for k in range(j + 1, num_vertices):
                    edge = tuple(sorted((surface[j], surface[k])))
                    sample_points = self.catenary_curve_params[edge][2][-1]
                    surface_points[(index) * num_samples:(index + 1) * num_samples, :] = sample_points
                    index += 1

            initial_guess = self.catenary_surface_params[surface]
            if np.isnan(initial_guess[0]):
                initial_guess[0] = 0.5
            initial_guess = np.nan_to_num(initial_guess, nan=0.0)
            # x = surface_points[:, 0]
            # y = surface_points[:, 1]
            # z = surface_points[:, 2]
            optimizer = CatenarySurfaceOptimizer(surface_points)
            result = optimizer.fit(initial_guess=initial_guess)
            # result = least_squares(self._residuals, initial_guess, args=(x, y, z))
            self.catenary_surface_params[surface] = result.x

        self.log_surface_time.append(time.time() - time_start_estimating_surface)

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

        inside = np.array([self._barycentric_test(point, p1, p2, p3) for point in points])

        # Filter points inside the triangle
        x_inside = x_flat[inside]
        y_inside = y_flat[inside]

        return x_inside.reshape(-1, 1), y_inside.reshape(-1, 1)

    def _barycentric_test(self, point, p1, p2, p3):
        return barycentric_test_with_distance(point, p1, p2, p3) == 0

    def _catenary_surface(self, params, x, y):
        a, x0, y0, z0 = params
        return z0 + a * np.cosh(np.sqrt((x - x0) ** 2 + (y - y0) ** 2) / a)

    # Define the residual function for least squares
    def _residuals(self, params, x, y, z):
        return z - self._catenary_surface(params, x, y)

    def _generate_planar_graph(self, network_coords, points):
        """
        Generate a planar graph from 3D point positions by projecting to 2D and connecting neighbors.
        Parameters:
            network_coords (ndarray):
            points (ndarray): each row is a 3D point (x, y, z).

        Returns:
            edges: A list of edges, where each edge is represented by the indices of the points it connects.
        """
        # Project points to 2D by discarding the z-coordinate
        points_array = np.array(points)
        points_2d = network_coords  # points_array[:, :2] #

        # Number of points
        n_points = points_2d.shape[0]

        # Collect edges
        edges_set = set()
        edges = dict()

        def dfs_pairs(L, i=0, visited=None):
            if visited is None:
                visited = set()
            for j in range(i + 1, len(L)):
                if tuple(sorted((i, j))) not in visited:
                    visited.add(tuple(sorted((i, j))))
                    yield (L[i], L[j])
                    yield from dfs_pairs(L, j, visited)

        # ij_pairs = list(dfs_pairs(range(n_points)))
        #
        # for i, j in ij_pairs:
        for i in range(n_points):
            for j in range(i + 1, n_points):
                intersection = False
                for e1, e2 in edges_set:
                    if self._intersect(points_2d[i], points_2d[j], points_2d[e1], points_2d[e2]):
                        intersection = True
                        # if points[i][2] + points[j][2] > points[e1][2] + points[e2][2]:
                        #     higher = tuple(sorted((e1, e2)))
                        break
                if not intersection:
                    coord1 = int(self._coord2index(network_coords[i]))
                    coord2 = int(self._coord2index(network_coords[j]))
                    key = [coord1, coord2]
                    val = [points_array[i, :], points_array[j, :]]
                    key, val = zip(*sorted(zip(key, val)))
                    key = tuple(key)
                    edges[key] = val
                    local_indices = tuple(sorted((i, j)))
                    edges_set.add(local_indices)

        # Generate triangle surfaces
        active_surface = []
        for i in range(n_points):
            for j in range(i + 1, n_points):
                for k in range(j + 1, n_points):
                    # Check if edges exist
                    if (i, j) in edges_set and (j, k) in edges_set and (i, k) in edges_set:
                        coord1 = int(self._coord2index(network_coords[i]))
                        coord2 = int(self._coord2index(network_coords[j]))
                        coord3 = int(self._coord2index(network_coords[k]))
                        key = tuple(sorted((coord1, coord2, coord3)))

                        # 1) Build the actual 3D (or 2D) coordinates for the new triangle
                        triangle_new = [
                            network_coords[i],  # e.g. (x_i, y_i, z_i)
                            network_coords[j],
                            network_coords[k]
                        ]

                        # 2) Check geometry vs. existing triangles
                        exclude_new = False

                        for tri_key in active_surface:
                            # Reconstruct the old triangle from tri_key -> tri_old
                            tcoords = [self._index2coord(idx) for idx in tri_key]

                            # Decide if triangle_new is inside or encloses tri_old
                            # (or they share the same plane, etc.)
                            if self.is_inside(triangle_new, tcoords) or self.is_enclosing(triangle_new, tcoords):
                                exclude_new = True
                                break

                        # 3) If still valid after geometry check, add
                        if not exclude_new:
                            if key not in self.catenary_surface_params:
                                self.catenary_surface_params[key] = [0.5, 0, 0, 0]
                            active_surface.append(key)

        self.active_surface = active_surface
        return edges


    def _generate_planar_graph_delaunay(self, network_coords, points):
        """
        Generate a planar graph from 3D point positions by projecting to 2D and connecting neighbors.
        Taking as a priori that the first four points are always connected
        Parameters:
            network_coords (ndarray):
            points (ndarray): each row is a 3D point (x, y, z).

        Returns:
            edges: A list of edges, where each edge is represented by the indices of the points it connects.
        """
        # Generates the planar graph using Delaunay triangulation
        tri = Delaunay(network_coords)
        simplices = tri.simplices  # corners of the triangular subregions

        # Extract and sort edges from all simplices
        edges_index = np.vstack((
            simplices[:, [0, 1]],
            simplices[:, [1, 2]],
            simplices[:, [2, 0]]))
        edges_index.sort(axis=1)
        np.unique(edges_index, axis=0)

        # Collect edges
        edges = dict()

        points_array = np.array(points)

        # for i, j in ij_pairs:
        for edge_index in edges_index.tolist():
            coord1 = int(self._coord2index(network_coords[edge_index[0]]))
            coord2 = int(self._coord2index(network_coords[edge_index[1]]))
            key = [coord1, coord2]
            val = [points_array[edge_index[0], :], points_array[edge_index[1], :]]
            key, val = zip(*sorted(zip(key, val)))
            key = tuple(key)
            edges[key] = val

        # Generate triangle surfaces
        active_surface = []
        for surface in simplices.tolist():
            i, j, k = surface
            coord1 = int(self._coord2index(network_coords[i]))
            coord2 = int(self._coord2index(network_coords[j]))
            coord3 = int(self._coord2index(network_coords[k]))
            key = tuple(sorted((coord1, coord2, coord3)))

            if key not in self.catenary_surface_params:
                # print(key)
                self.catenary_surface_params[key] = [0.5, 0, 0, 0]
            active_surface.append(key)

        self.active_surface = active_surface
        return edges


    def is_inside(self, triA, triB):
        """
        Returns True if all vertices of triA are inside triB.
        triA and triB are lists of 2D or 3D points (or after projection).
        """
        # For each vertex in triA, check if it's inside triB
        for pt in triA:
            if not self._barycentric_test(pt, triB[0], triB[1], triB[2]):
                return False
        return True

    def point_in_triangle(self, pt, v1, v2, v3):
        """
        Returns True if pt is inside the triangle formed by v1, v2, v3 (2D coordinates).
        """
        # Transform points into arrays for convenience
        p = np.array(pt)
        a = np.array(v1)
        b = np.array(v2)
        c = np.array(v3)

        # Compute vectors
        v0 = c - a
        v1_ = b - a
        v2_ = p - a

        # Dot products
        dot00 = v0.dot(v0)
        dot01 = v0.dot(v1_)
        dot02 = v0.dot(v2_)
        dot11 = v1_.dot(v1_)
        dot12 = v1_.dot(v2_)

        # Barycentric coordinates
        invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom

        # Check if point is in triangle
        return (u > 0) and (v > 0) and (u + v < 1)

    def is_enclosing(self, triA, triB):
        """
        Returns True if triA encloses triB (i.e. all vertices of triB are inside triA).
        """
        return self.is_inside(triB, triA)

    def _build_catenary_network(self, parallelize=False):
        """
        Build a catenary network using fitting and sample equally spaced points on each curve.
        Perform batch operations to compute catenary parameters and sampled points efficiently.
        Parameters:
            parallelize: [no effective] enable parallelization?
        Returns: None
        """
        num_connections = len(self.active_ridge)
        num_samples = self.num_samples
        samples_per_connection = np.zeros((num_connections, num_samples, 3))

        # Step 1: Transform to local frames for all connections
        rotations = []
        translations = []
        local_frames = []
        lengths = []
        curve_lengths = []
        guesses = []
        for connection, points in self.active_ridge.items():
            p1, p2 = points
            rotation, translation, p1_local, p2_local, length = self._transform_to_local_frame(p1, p2)
            rotations.append(rotation)
            translations.append(translation)
            local_frames.append((p1_local, p2_local))
            lengths.append(np.linalg.norm(p1_local - p2_local))
            catenary_curve_params = self.catenary_curve_params[tuple(sorted((connection[0], connection[1])))]
            curve_lengths.append(catenary_curve_params[0])
            guesses.append(catenary_curve_params[1])

        # Step 2: Optimize catenary parameters for all connections
        catenary_params = []
        for i, (p1_local, p2_local) in enumerate(local_frames):
            x1, z1 = p1_local[0], p1_local[2]
            x2, z2 = p2_local[0], p2_local[2]
            length = curve_lengths[i]

            # Perform optimization
            result = minimize(
                self._objective_with_gradient,
                guesses[i],
                args=(x1, z1, x2, z2, length),
                bounds=[(1e-2, 10), (None, None), (None, None)],  # Ensure c > 0
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
            samples_per_connection[i, :, :] = global_points

        # Combine results into a structured output
        for i, val in enumerate(zip(self.active_ridge.keys(), catenary_params)):
            connection, c_params = val
            self.catenary_curve_params[tuple(sorted((connection[0], connection[1])))][1] = c_params
            self.catenary_curve_params[tuple(sorted((connection[0], connection[1])))][2] = \
                lengths[i], rotations[i], translations[i], samples_per_connection[i, :, :]

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

    def _objective_with_gradient(self, params, x1, z1, x2, z2, L,
                                 alpha=0.1, beta=0.1, gamma=0.1):
        """
        Objective function with gradients and scaling factors.
        alpha: scaling factor for z1 error
        beta: scaling factor for z2 error
        gamma: scaling factor for length error
        """
        c, x0, z0 = params

        # Predicted z-values
        z1_pred = c * np.cosh((x1 - x0) / c) + z0
        z2_pred = c * np.cosh((x2 - x0) / c) + z0

        # Exact curve length
        L_cat = self._compute_exact_arc_length(c, x0, x1, x2)

        # Residuals with scaling
        error_z1 = (z1 - z1_pred) * np.sqrt(alpha)  # sqrt for proper scaling in least squares
        error_z2 = (z2 - z2_pred) * np.sqrt(beta)
        error_length = (L - L_cat) * np.sqrt(gamma)

        # Scaled objective function
        objective_value = (alpha * (z1 - z1_pred) ** 2 +
                           beta * (z2 - z2_pred) ** 2 +
                           gamma * (L - L_cat) ** 2)

        # Gradients (using intermediate variables)
        dz1_dc = np.cosh((x1 - x0) / c) - (x1 - x0) / c * np.sinh((x1 - x0) / c)
        dz2_dc = np.cosh((x2 - x0) / c) - (x2 - x0) / c * np.sinh((x2 - x0) / c)

        dz1_dx0 = -np.sinh((x1 - x0) / c)
        dz2_dx0 = -np.sinh((x2 - x0) / c)

        # Arc length gradients (assuming implementation exists)
        dL_dc, dL_dx0 = self._gradient_arc_length(c, x0, x1, x2)

        # Scaled gradient components
        grad_c = (-2 * alpha * (z1 - z1_pred) * dz1_dc
                  - 2 * beta * (z2 - z2_pred) * dz2_dc
                  - 2 * gamma * (L - L_cat) * dL_dc)

        grad_x0 = (-2 * alpha * (z1 - z1_pred) * dz1_dx0
                   - 2 * beta * (z2 - z2_pred) * dz2_dx0
                   - 2 * gamma * (L - L_cat) * dL_dx0)

        grad_z0 = (-2 * alpha * (z1 - z1_pred) * 1.0  # dz1/dz0 = 1
                   - 2 * beta * (z2 - z2_pred) * 1.0)  # dz2/dz0 = 1

        return objective_value, np.array([grad_c, grad_x0, grad_z0])

    def _compute_exact_arc_length(self, c, x0, x1, x2):
        """
        Compute the exact arc length of the catenary curve.
        """
        return c * (np.sinh((x2 - x0) / c) - np.sinh((x1 - x0) / c))

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
        if abs(rotation_angle) < 1e-6:  # Avoid divide-by-zero
            rotation_angle = 1e-6 * ((rotation_angle > 0) - 0.5)

        if np.linalg.norm(rotation_axis) > 1e-6:  # Avoid divide-by-zero
            rotation_axis /= np.linalg.norm(rotation_axis)
        else:
            rotation_axis = np.array([0, 0, 1])  # No rotation needed

        rotation = R.from_rotvec(rotation_angle * rotation_axis)

        # Transform points to local frame
        p1_local = np.array([0, 0, 0])  # Set p1 as the origin
        p2_local = rotation.apply(np.array(p2) - np.array(p1))

        return rotation, p1, p1_local, p2_local, length

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

    def orientation(self, p, q, r):
        """
        Return:
          0 -> p, q, r are collinear
          1 -> Clockwise
          2 -> Counterclockwise
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # collinear
        elif val > 0:
            return 1  # clockwise
        else:
            return 2  # counterclockwise

    def on_segment(self, p, q, r):
        """
        Check if point q lies on segment pr
        """
        if (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1])):
            return True
        return False

    def intersect(self, A, B, C, D):
        """
        Returns True if segments AB and CD intersect, otherwise False.
        """
        # Step 1: Find the four orientations needed for the general and special cases
        o1 = self.orientation(A, B, C)
        o2 = self.orientation(A, B, D)
        o3 = self.orientation(C, D, A)
        o4 = self.orientation(C, D, B)

        # Step 2: General Case
        if o1 != o2 and o3 != o4:
            return True

        # Step 3: Special Cases
        # A, B, C are collinear and C lies on segment AB
        if o1 == 0 and self.on_segment(A, C, B):
            return True

        # A, B, D are collinear and D lies on segment AB
        if o2 == 0 and self.on_segment(A, D, B):
            return True

        # C, D, A are collinear and A lies on segment CD
        if o3 == 0 and self.on_segment(C, A, D):
            return True

        # C, D, B are collinear and B lies on segment CD
        if o4 == 0 and self.on_segment(C, B, D):
            return True

        # Otherwise
        return False

    def _intersect(self, A, B, C, D):
        """
        Returns True if segments AB and CD 'strictly' intersect.
        A single shared endpoint does NOT count as an intersection.
        """
        if self.intersect(A, B, C, D) is False:
            return False

        # 1) If they truly intersect in the interior, keep it as True.
        # 2) If they intersect at a single endpoint only, exclude.

        # Gather all endpoints that are exactly the same
        shared_points = []
        for p in [A, B]:
            for q in [C, D]:
                if np.allclose(p, q):
                    shared_points.append(p)

        # If there's exactly one shared point and no other interior crossing,
        # consider that 'no intersection'.
        if len(shared_points) == 1:
            return False

        return True

    def _coord2index(self, coord):
        i, j = coord
        return i * self.lc + j

    def _index2coord(self, index: int):
        i = index // self.lc
        j = index - i * self.lc
        return i, j


def visualize(fig, ax, flysurf, plot_dot=True, plot_curve=True, plot_surface=True, num_samples=10):
    for i, connection in enumerate(flysurf.active_ridge):
        # Retrieve curve parameters and sampled points
        V1, V2 = connection
        i1, j1 = flysurf._index2coord(V1)
        i2, j2 = flysurf._index2coord(V2)
        curve_length, catenary_param, other_data = flysurf.catenary_curve_params[tuple(sorted((V1, V2)))]
        c, x0, z0 = catenary_param
        dist, rotation, translation, samples_per_connection = other_data

        if plot_dot:
            # Plot the sampled points on the curve
            ax.scatter(
                samples_per_connection[1:flysurf.num_samples - 1, 0],
                samples_per_connection[1:flysurf.num_samples - 1, 1],
                samples_per_connection[1:flysurf.num_samples - 1, 2],
                color="black",
                label=f"Sampled Points {i}"
            )

            ax.scatter(
                samples_per_connection[0, 0],
                samples_per_connection[0, 1],
                samples_per_connection[0, 2],
                color="red",
                label=f"Endpoints {i}"
            )

            ax.scatter(
                samples_per_connection[-1, 0],
                samples_per_connection[-1, 1],
                samples_per_connection[-1, 2],
                color="red",
                label=f"Endpoints {i}"
            )
        if plot_curve:
            x_end = np.linalg.norm(samples_per_connection[-1][:2] - samples_per_connection[0][:2])

            # Plot the full catenary curve
            x_full = np.linspace(0, x_end, num_samples)  # sampling
            z_full = c * np.cosh((x_full - x0) / c) + z0
            y_full = np.zeros_like(x_full)
            full_curve_local = np.vstack((x_full, y_full, z_full)).T
            full_curve_global = rotation.inv().apply(full_curve_local) + translation
            ax.plot(
                full_curve_global[:, 0],
                full_curve_global[:, 1],
                full_curve_global[:, 2],
                "*",
                alpha=0.75,
                label=f"Curve {i}",
                linewidth=2
            )
    if plot_surface:
        for i, surface in enumerate(flysurf.active_surface):
            num_edges = len(surface)
            c, x, y, z = flysurf.catenary_surface_params[surface]
            points = []
            for j in range(num_edges):
                for k in range(j + 1, num_edges):
                    edge = tuple(sorted((surface[j], surface[k])))
                    for v in flysurf.active_ridge[edge]:
                        good = True
                        for p in points:
                            if np.allclose(v, p):
                                good = False
                                break
                        if good:
                            points.append(v)

            x_mesh, y_mesh = flysurf._generate_mesh_in_triangle(np.array(points[0])[:2],
                                                                np.array(points[1])[:2],
                                                                np.array(points[2])[:2], num_samples)

            # Compute the fitted z-values
            z_mesh = flysurf._catenary_surface((c, x, y, z), x_mesh, y_mesh)

            # Plot the fitted catenary surface
            try:
                # ax.plot_trisurf(x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten(), cmap=christmas_cmap, alpha=1.0, edgecolor='none')
                ax.plot(x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten(), "*")
            except:
                pass

    plt.pause(0.0001)


def compute_intersection(p1, p2, q1, q2):
    """
    Compute the intersection of two line segments defined by endpoints p1 -> p2 and q1 -> q2.
    Returns the intersection point if it exists and lies within both line segments, otherwise None.
    """
    A = np.array([p2 - p1, q1 - q2]).T
    b = q1 - p1

    try:
        t, s = np.linalg.solve(A, b)  # Solve for parameters t and s
        if 0 <= t <= 1 and 0 <= s <= 1:  # Ensure the intersection lies within both segments
            return (1 - t) * p1 + t * p2  # Compute intersection point
    except np.linalg.LinAlgError:
        pass  # Lines are parallel or coincident

    return None


def find_all_intersections_batch(S1, S2):
    """
    Optimized function to find all intersections between line segments in S1 and S2 using numpy batch operations.
    S1 and S2 are arrays of shape (N, 2, 2) and (M, 2, 2), where each segment is defined by two endpoints [p1, p2].
    Returns an array of intersection points.
    """
    S1 = np.array(S1)  # Ensure input is a numpy array
    S2 = np.array(S2)

    p1 = S1[:, 0, :]  # Start points of segments in S1
    p2 = S1[:, 1, :]  # End points of segments in S1
    q1 = S2[:, 0, :]  # Start points of segments in S2
    q2 = S2[:, 1, :]  # End points of segments in S2

    # Direction vectors for segments
    d1 = p2 - p1  # Direction vectors for S1
    d2 = q2 - q1  # Direction vectors for S2

    # Cross product for batch determinant computation
    cross_d1_d2 = np.cross(d1[:, None], d2)
    cross_q1_p1_d2 = np.cross(q1[None, :] - p1[:, None], d2)
    cross_p1_q1_d1 = np.cross(p1[:, None] - q1, d1)

    # Avoid division by zero (parallel lines) by masking invalid values
    valid_mask = cross_d1_d2 != 0

    # Solve for t and s only where valid
    t = np.where(valid_mask, cross_q1_p1_d2 / cross_d1_d2, np.nan)
    s = np.where(valid_mask, cross_p1_q1_d1 / cross_d1_d2, np.nan)

    # Compute intersection points where valid
    intersections = p1[:, None] + t[..., None] * d1[:, None]
    return intersections.reshape(-1, 2)


class FlysurfSampler:
    def __init__(self, flysurf, resolution, points=None, coordinates=None):
        self.flysurf = flysurf
        self.resolution = resolution
        if points is None:
            self.filtered_samples = np.zeros((resolution ** 2, 3))
        else:
            self.filtered_samples = self.sampling_v1(None, None, points, coordinates)
        self.vel = np.zeros((resolution ** 2, 3))

    def sampling_v1(self, fig, ax, points, coordinates, plot=False):
        """ NOTE:
            We have to ensure that the 4 corners of the mesh are
            ALWAYS giving us four active outermost ridges
            @params:
                fig, ax: matplotlib figure handles to facilitate the visualization
                plot: set to True to plot on the ax
                flysurf: the flysurf instance
                resolution: the number of samples to take on each boundary ridge
                points: the positions of the actuated mass points on the flysurf
        """
        # collection of all samples
        # FIRST: Four corners
        resolution = self.resolution
        flysurf = self.flysurf

        all_samples = np.zeros((resolution ** 2, 3))
        # upper-right
        # upper-left
        # lower-left
        # lower-right
        all_samples[[resolution ** 2 - 1,
                     resolution ** 2 - resolution,
                     0, resolution - 1], :] = points[0:4, :]

        if plot:
            ax.plot(points[0:4, 0], points[0:4, 1], points[0:4, 2], "*")

        time_start_sampling_v1 = time.time_ns()
        four_outermost_edges = [(0, resolution - 1),  # bottom
                                (resolution - 1, resolution ** 2 - 1),  # right
                                ((resolution - 1) * resolution, resolution ** 2 - 1),  # top
                                (0, (resolution - 1) * resolution)]  # left

        line_seg_points = []

        for i, connection in enumerate(four_outermost_edges):
            # Retrieve curve parameters and sampled points
            curve_length, catenary_param, other_data = flysurf.catenary_curve_params[connection]
            if other_data:
                # if the corners are directed connected
                dist, rotation, translation, samples_per_connection = other_data

                # Plot the full catenary curve
                full_curve_global = samples_per_connection[1:len(samples_per_connection) - 1]
            else:
                # if the corners are separated by mid points on the outermost sides
                if i == 0:
                    possible_side_points = list(range(0, resolution))
                elif i == 1:
                    possible_side_points = list(range(resolution - 1, resolution ** 2, resolution))
                elif i == 2:
                    possible_side_points = list(range((resolution - 1) * resolution, resolution ** 2))
                elif i == 3:
                    possible_side_points = list(range(0, (resolution - 1) * resolution + 1, resolution))

                index1 = 0
                full_curve_global = np.zeros_like(all_samples[1:resolution - 1, :])

                while index1 < len(possible_side_points) - 1:
                    index2 = index1 + 1
                    while index2 < len(possible_side_points):
                        point1 = possible_side_points[index1]
                        point2 = possible_side_points[index2]
                        connection_key = (point1, point2)
                        curve_length, catenary_param, other_data = flysurf.catenary_curve_params[connection_key]
                        if other_data:
                            # Hah! We found a connection!
                            dist, rotation, translation, samples_per_connection = other_data
                            if index1 == 0:
                                num_samples = index2 - index1 - 1
                                if not num_samples == 0:
                                    if len(range(resolution // num_samples + 1, resolution,
                                                 resolution // num_samples)) < num_samples:
                                        full_curve_global[index1:index1 + num_samples, :] = samples_per_connection[
                                                                                            resolution // num_samples - 1:resolution:resolution // num_samples,
                                                                                            :]
                                    else:
                                        full_curve_global[index1:index1 + num_samples, :] = samples_per_connection[
                                                                                            resolution // num_samples + 1:resolution:resolution // num_samples,
                                                                                            :]
                            else:
                                num_samples = index2 - index1
                                full_curve_global[index1 - 1:index2, :] = samples_per_connection[
                                                                          1:resolution:resolution // num_samples, :]
                            index1 = index2 - 1
                            break
                        else:
                            index2 += 1
                    index1 += 1

            # SECOND: four sides
            if i == 0:
                all_samples[1:resolution - 1, :] = full_curve_global
            elif i == 1:
                all_samples[range(resolution * 2 - 1, resolution ** 2 - 1, resolution), :] = full_curve_global
            elif i == 2:
                all_samples[(resolution - 1) * resolution + 1: resolution ** 2 - 1, :] = full_curve_global
            elif i == 3:
                all_samples[range(resolution, (resolution - 1) * resolution, resolution), :] = full_curve_global

            line_seg_points.append(full_curve_global[:, 0:2])
            if plot:
                ax.plot(
                    full_curve_global[:, 0],
                    full_curve_global[:, 1],
                    full_curve_global[:, 2],
                    "*"
                )
        time_start_sampling_v1 = time.time_ns()

        line_pairs = [[line_seg_points[1], line_seg_points[3]], [line_seg_points[0], line_seg_points[2]]]
        S = [np.zeros((resolution - 2, 2, 2)), np.zeros((resolution - 2, 2, 2))]
        for i, points_pair in enumerate(line_pairs):
            points1, points2 = points_pair
            if not compute_intersection(points1[0], points2[0], points1[-1], points2[-1]) is None:
                # let's see how we align the points on the two segments
                points1 = points1[::-1]
            for j in range(resolution - 2):
                S[i][j, 0, :] = points1[j, :]
                S[i][j, 1, :] = points2[j, :]

        print("Check point 1 (ms):", (time.time_ns() - time_start_sampling_v1) * 1e-6)
        time_start_sampling_v1 = time.time_ns()
        # print(S[0], S[1])
        intersections = find_all_intersections_batch(S[0], S[1])
        print("Check point 2 (ms):", (time.time_ns() - time_start_sampling_v1) * 1e-6)
        time_start_sampling_v1 = time.time_ns()

        surf_info = []
        for i, surface in enumerate(flysurf.active_surface):
            num_edges = len(surface)
            c, x, y, z = flysurf.catenary_surface_params[surface]
            surf_corners = []
            for j in range(num_edges):
                for k in range(j + 1, num_edges):
                    edge = tuple(sorted((surface[j], surface[k])))
                    for v in flysurf.active_ridge[edge]:
                        good = True
                        for p in surf_corners:
                            if np.allclose(v, p):
                                good = False
                                break
                        if good:
                            surf_corners.append(v)

            surf_info.append((surf_corners, [c, x, y, z]))

        print("Check point 3 (ms):", (time.time_ns() - time_start_sampling_v1) * 1e-6)
        time_start_sampling_v1 = time.time_ns()

        z_vals = batch_surf_sampling(intersections, surf_info)

        print("Check point 4 (ms):", (time.time_ns() - time_start_sampling_v1)*1e-6)

        # THIRD: inner surfaces
        intersections = np.column_stack((intersections, z_vals))

        all_samples[[i * resolution + j for i in range(1, resolution - 1) for j in range(1, resolution - 1)],
                    :] = intersections

        # Force the samples to cover the inner actuators
        # inner_points = points[4:, :]
        # inner_coordinates = coordinates[4:, :]
        control_indices = []
        for coordinate in coordinates:
            i, j = coordinate
            control_indices.append(i * resolution + j)

        num_actuators = len(control_indices)
        sigma = np.zeros((num_actuators,)) + 0.3
        alpha = np.zeros((num_actuators,)) + 0.3
        if num_actuators > 4:
            sigma[4] -= 0.1
            alpha[4] += 0.7

        all_samples = drag_points_vectorized(all_samples, control_indices, points,
                                             sigma=sigma)

        if plot:
            ax.plot(intersections[:, 0], intersections[:, 1], intersections[:, 2], "*")
            plt.pause(0.0001)
        return all_samples

    def smooth_particle_cloud(self, measurements, L, dt, alpha=0.5, filter_on=True):
        """
        Smooth a cloud of particle trajectories with Lipschitz continuity constraints.

        Args:
            measurements: Array of shape (n_particles, n_steps, 3) containing noisy 3D positions.
            L: Lipschitz constant (maximum velocity magnitude).
            dt: Time step between consecutive measurements.
            alpha: Smoothing factor (0 < alpha < 1). Smaller values = more smoothing.
            filter_on: Enable filtering on particles when True.

        Returns:
            smoothed: Array of shape (n_particles, n_steps, 3) containing smoothed positions.
        """
        if filter_on:
            n_particles, _ = measurements.shape
            max_step = L * dt  # Maximum allowed displacement per step

            # Compute raw delta between current measurement and previous smoothed position
            raw_delta = measurements - self.filtered_samples
            step = alpha * raw_delta

            # Compute norms of steps for all particles
            step_norms = np.linalg.norm(step, axis=1, keepdims=True)  # Shape: (n_particles, 1)
            step_norms += 1e-3 * (step_norms < 1e-3)  # to overcome divided-by-zero problem

            # Clamp steps exceeding max_step
            over_limit = step_norms > max_step
            self.vel = np.where(over_limit, (step / step_norms) * max_step, step)

            # Update smoothed positions
            self.filtered_samples += self.vel
        else:
            self.vel = (measurements - self.filtered_samples)
            self.filtered_samples = measurements

        return self.filtered_samples


def drag_points_vectorized(A, control_indices, B, sigma, alpha=None):
    """
    Drag points in A based on control points and their target positions, with a per-control-point sigma.

    Parameters:
    A : ndarray of shape (n, 3)
        The original 3D points.
    control_indices : array-like of int
        Indices of control points in A that are dragged.
    B : ndarray of shape (k, 3)
        The target positions for the control points.
    sigma : ndarray of shape (k,)
        Gaussian spread for each control point; smaller values make the influence more local.
    alpha : ndarray of shape (k,), optional
        Weight of effect for each control point. Larger values amplify the influence.
        If None, defaults to all ones.

    Returns:
    A_new : ndarray of shape (n, 3)
        The updated 3D points after applying the drag.
    """
    # Ensure sigma is a numpy array and has proper dimensions.
    sigma = np.asarray(sigma)
    if sigma.ndim != 1 or sigma.shape[0] != B.shape[0]:
        raise ValueError("sigma must be a 1D array with the same length as the number of control points in B")
    
    # Validate alpha
    if alpha is None:
        alpha = np.ones(B.shape[0])
    else:
        alpha = np.asarray(alpha)
        if alpha.ndim != 1 or alpha.shape[0] != B.shape[0]:
            raise ValueError("alpha must be a 1D array with the same length as B")

    # Extract control points from A and compute their displacements
    A_control = A[control_indices]  # shape (k, 3)
    d = B - A_control  # shape (k, 3)

    # Compute differences between every point in A and each control point:
    # diff has shape (n, k, 3)
    diff = A[:, None, :] - A_control[None, :, :]

    # Compute squared Euclidean distances, shape (n, k)
    dist_sq = np.sum(diff ** 2, axis=2)

    # Use broadcasting to compute per-control-point Gaussian weights.
    # (sigma**2) is of shape (k,), so we add a new axis to broadcast over n.
    weights = alpha[None, :] * np.exp(-dist_sq / (sigma ** 2)[None, :])

    # Compute the weighted displacement for each point in A.
    # weights[..., None] has shape (n, k, 1), d[None, :, :] has shape (1, k, 3)
    weighted_disp = np.sum(weights[..., None] * d[None, :, :], axis=1)

    # Normalize by the sum of weights for each point to get the average displacement.
    weight_sum = np.sum(weights, axis=1)[:, None]

    # Update A with the computed displacements.
    A_new = A + weighted_disp / weight_sum

    return A_new


def oscillation(i):
    return np.sin(i * np.pi / 100) * 0.05


def Euler_distance_points(rows, cols, states):
    states_reshaped = states.reshape((rows, cols, 3))  # Shape (9, 9, 3)

    # Initialize list to store distances
    distances = []

    # Compute distances for adjacent points (right and down)
    for i in range(rows):
        for j in range(cols):
            # Current point
            current_point = states_reshaped[i, j]

            # Right neighbor (if not on the last column)
            if j < cols - 1:
                right_point = states_reshaped[i, j + 1]
                distance = np.linalg.norm(current_point - right_point)
                distances.append(distance)

            # Downward neighbor (if not on the last row)
            if i < rows - 1:
                down_point = states_reshaped[i + 1, j]
                distance = np.linalg.norm(current_point - down_point)
                distances.append(distance)

    # Calculate the average distance
    average_distance = np.mean(distances)

    return average_distance


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


if __name__ == "__main__":
    vel_hist = []
    vel_raw_hist = []
    dt = 0.1
    max_speed = 0.5

    """ NOTE: 
        Please make sure the first four points are the four outermost
        corners of the mesh, AND
        They are ordered as upper-right
                            upper-left
                            lower-left
                            lower-right
    """
    mesh_size = 15  # number of samples on the outermost sides
    points_coord = np.array([[mesh_size - 1, mesh_size - 1],
                             [mesh_size - 1, 0],
                             [0, 0],
                             [0, mesh_size - 1],
                             [(mesh_size - 1) // 2, (mesh_size - 1) // 2],#])
                             [mesh_size - 1, (mesh_size - 1) // 2],
                             [0, (mesh_size - 1) // 2],
                             [(mesh_size - 1) // 2, mesh_size - 1],
                             [(mesh_size - 1) // 2, 0]])

    points = np.array([[0.9, 0.4, 0.45],
                       [0.1, 0.4, 0.45],
                       [0.1, -0.4, 0.45],
                       [0.9, -0.4, 0.45],
                       [0.5, 0., 0.45],#])
                       [0.5, 0.4, 0.45],
                       [0.5, -0.4, 0.45],
                       [0.9, 0.0, 0.45],
                       [0.1, 0.0, 0.45]])

    # points = np.array([[ 0.41,   0.39,    0.15],       # 0
    #                    [-0.38,   0.42,   -0.05],     # 1
    #                    [-0.44,  -0.37,    0.08],      # 2
    #                    [ 0.40,  -0.28,   -0.04],    # 3
    #                 #    [-0.19,   0.01,     0.2],
    #                 #    [ 0.21,  -0.02,     0.1],
    #                    [0.04,    0.07,    -0.1]])
    flysurf = CatenaryFlySurf(mesh_size, mesh_size, 1.0 / (mesh_size - 1), num_sample_per_curve=mesh_size)
    flysurf.update(points_coord, points)
    sampler = FlysurfSampler(flysurf, mesh_size, points, points_coord)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90, azim=-90)

    try:
        for i in range(2000):
            ax.clear()
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_xlim(-0.0, 1.0)
            ax.set_ylim(-0.5, 0.5)
            ax.set_zlim(-0.5, 0.5)
            # random_array = np.random.normal(loc=0, scale=0.001, size=points.shape)
            # points += random_array
            points[0, 2] += 0.11 * oscillation(5.0 * i)
            points[1, 2] += 0.13 * oscillation(7.0 * i + 1)
            points[2, 2] -= 0.14 * oscillation(8.0 * i + 1.74)
            points[3, 2] += 0.13 * oscillation(6.0 * i + 4.1)
            points[4, 2] -= 0.12 * oscillation(2.5 * i + 3)
            points[5, 2] -= 0.10 * oscillation(3.1 * i + 1.2)
            points[6, 2] += 0.11 * oscillation(5.1 * i + 2.0)
            points[7, 2] -= 0.13 * oscillation(0.9 * i - 0.5)
            points[8, 2] += 0.13 * oscillation(4.1 * i - 1.1)
            points[0, :2] += 0.01 * np.array([np.cos(0.1 * i), np.sin(0.1 * i)])
            points[1, :2] += 0.01 * np.array([np.cos(0.1 * i), np.sin(0.1 * i)])
            points[2, :2] += 0.01 * np.array([np.cos(0.1 * i), np.sin(0.1 * i)])
            points[3, :2] += 0.01 * np.array([np.cos(0.1 * i), np.sin(0.1 * i)])
            points[4, :2] += 0.01 * np.array([np.cos(0.1 * i), np.sin(0.1 * i)])
            points[5, :2] += 0.01 * np.array([np.cos(0.1 * i), np.sin(0.1 * i)])
            points[6, :2] += 0.01 * np.array([np.cos(0.1 * i), np.sin(0.1 * i)])
            points[7, :2] += 0.01 * np.array([np.cos(0.1 * i), np.sin(0.1 * i)])
            points[8, :2] += 0.01 * np.array([np.cos(0.1 * i), np.sin(0.1 * i)])
            # points += np.random.normal(loc=0, scale=0.002, size=points.shape)

            # ax.view_init(elev=45+15*np.cos(i/17), azim=60+0.45*i)
            # time_start = time.time()
            sampler.flysurf.update(points_coord, points)
            # print("elapsed time till update:", time.time() - time_start)

            # visualize(fig, ax, flysurf, plot_dot=False, plot_curve=True, plot_surface=False, num_samples=25)

            all_samples = sampler.sampling_v1(fig, ax, points, coordinates=points_coord)
            # print("elapsed time till sampling:", time.time() - time_start)
            vel_raw_hist.append(np.linalg.norm((all_samples - sampler.filtered_samples)[:, 1]))

            filtered_points = sampler.smooth_particle_cloud(all_samples, max_speed, dt)
            # print("total elapsed time:", time.time() - time_start)
            vel_hist.append(np.linalg.norm(sampler.vel[:, 1]))

            # Before filter
            # ax.plot(all_samples[:, 0], all_samples[:, 1], all_samples[:, 2], "*")
            #
            unfiltered_samples_rows = all_samples.reshape((mesh_size, mesh_size, 3))

            # for i in range(unfiltered_samples_rows.shape[0]):
            #     ax.plot(unfiltered_samples_rows[i, :, 0], unfiltered_samples_rows[i, :, 1], unfiltered_samples_rows[i, :, 2], "*-")
            # plt.pause(0.0001)

            # After filter
            # ax.plot(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2], "*")
            # plt.pause(0.0001)
            filtered_samples_rows = filtered_points.reshape((mesh_size, mesh_size, 3))
            for i in range(filtered_samples_rows.shape[0]):
                ax.plot(filtered_samples_rows[i, :, 0], filtered_samples_rows[i, :, 1], filtered_samples_rows[i, :, 2], "-*")
            plt.pause(0.0001)

            # print(average_hausdorff_distance(all_samples, filtered_points))
            # input()
    finally:
        print("mean and std time in finding the catenary:", np.mean(flysurf.log_catenary_time),
              np.std(flysurf.log_catenary_time))
        print("mean and std time in finding the surfaces:", np.mean(flysurf.log_surface_time),
              np.std(flysurf.log_surface_time))
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(flysurf.log_catenary_time, "r.")
        ax1.plot(flysurf.log_surface_time, "b.")
        plt.pause(0.0001)
        input()
