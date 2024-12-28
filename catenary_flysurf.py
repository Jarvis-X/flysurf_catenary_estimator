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
    "#8B0000"   # DarkRed
]
# Create a linear segmented colormap from these colors
christmas_cmap = mcolors.LinearSegmentedColormap.from_list("christmas", christmas_colors, N=256)


class CatenaryFlySurf:
    def __init__(self, lc, lr, l_cell, num_sample_per_curve=5):
        self.lc = lc
        self.lr = lr
        self.num_points = lc*lr
        self.cell_length = l_cell
        self.num_samples = num_sample_per_curve
        self.active_ridge = None
        self.active_surface = None

        # stores catenary-related data based on vertex indices
        self.catenary_curve_params = dict()  # [(Vi, Vj)]: [curve length, last stored c x y, other data]
        self.catenary_surface_params = dict()  #

        for i in range(self.num_points):
            for j in range(i+1, self.num_points):
                i1, j1 = self._index2coord(i)
                i2, j2 = self._index2coord(j)
                self.catenary_curve_params[tuple(sorted((i, j)))] = [
                    np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2) * self.cell_length, [0.5, 0, 0], None]

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
        dist = np.sum(np.abs(points_coord - (points_coord.min(axis=0) + points_coord.max(axis=0))/2), axis=1)
        # normalize the height between 0 and 1
        height_norm = (points_position[:, 2] - points_position[:, 2].min()) / (points_position[:, 2].max() - points_position[:, 2].min())
        order_of_points = np.argsort(dist-height_norm)
        # order_of_points = np.argsort(- points_position[:, -1])

        points = points_position[order_of_points]
        network_coords = points_coord[order_of_points]
        # print(network_coords)

        edges = self._generate_planar_graph(network_coords, points)
        self.active_ridge = edges
        self._build_catenary_network(parallelize=False)

        num_samples = self.num_samples

        # for estimating the surface
        for i, surface in enumerate(self.active_surface):
            num_vertices = len(surface)
            num_edges = num_vertices
            surface_points = np.zeros((num_edges * num_samples, 3))
            index = 0
            for j in range(num_vertices):
                for k in range(j+1, num_vertices):
                    edge = tuple(sorted((surface[j], surface[k])))
                    sample_points = self.catenary_curve_params[edge][2][-1]
                    surface_points[(index) * num_samples:(index + 1) * num_samples, :] = sample_points
                    index += 1

            initial_guess = self.catenary_surface_params[surface]
            x = surface_points[:, 0]
            y = surface_points[:, 1]
            z = surface_points[:, 2]
            result = least_squares(self._residuals, initial_guess, args=(x, y, z))
            self.catenary_surface_params[surface] = result.x

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
        return (u > 0) and (v > 0) and (u + v < 1)

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
        points_2d = network_coords # points_array[:, :2]

        # Number of points
        n_points = points_2d.shape[0]

        # Collect edges
        edges_set = set()
        edges = dict()

        def dfs_pairs(L, i=0, visited=None):
            if visited is None:
                visited = set()
            for j in range(i + 1, len(L)):
                if (i, j) not in visited:
                    visited.add((i, j))
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
                    # print([network_coords[i], network_coords[j]])
                    coord1 = int(self._coord2index(network_coords[i]))
                    coord2 = int(self._coord2index(network_coords[j]))
                    edges[tuple(sorted((coord1, coord2)))] = points_array[i, :], points_array[j, :]
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

    def is_inside(self, triA, triB):
        """
        Returns True if all vertices of triA are inside triB.
        triA and triB are lists of 2D or 3D points (or after projection).
        """
        # For each vertex in triA, check if it's inside triB
        # If 3D, you might need to project them onto a common plane or handle them in 3D if they're guaranteed to be co-planar
        for pt in triA:
            if self._barycentric_test(pt, triB[0], triB[1], triB[2]):
                return True
        return False

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
        if np.linalg.norm(rotation_axis) > 1e-6:  # Avoid divide-by-zero
            rotation_axis /= np.linalg.norm(rotation_axis)
        else:
            rotation_axis = np.array([0, 0, 0])  # No rotation needed

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
        return i*self.lc + j

    def _index2coord(self, index: int):
        i = index // self.lc
        j = index - i*self.lc
        return i, j


def visualize(fig, ax, flysurf, plot_dot=True, plot_curve=True, plot_surface=True):
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
            x_full = np.linspace(0, x_end, 10)  # sampling
            z_full = c * np.cosh((x_full - x0) / c) + z0
            y_full = np.zeros_like(x_full)
            full_curve_local = np.vstack((x_full, y_full, z_full)).T
            full_curve_global = rotation.inv().apply(full_curve_local) + translation
            ax.plot(
                full_curve_global[:, 0],
                full_curve_global[:, 1],
                full_curve_global[:, 2],
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
                                                                np.array(points[2])[:2], 10)

            # Compute the fitted z-values
            z_mesh = flysurf._catenary_surface((c, x, y, z), x_mesh, y_mesh)

            # Plot the fitted catenary surface
            try:
                # ax.plot_trisurf(x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten(), cmap=christmas_cmap, alpha=1.0, edgecolor='none')
                ax.plot(x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten(), "*")
            except:
                pass

    plt.pause(0.0001)

def oscillation(i):
    return np.sin(i*np.pi/100)*0.01


if __name__ == "__main__":
    points_coord = np.array([[8, 8],
                             [0, 8],
                             [0, 0],
                             [8, 0],
                             [2, 4],
                             [6, 4],
                             [5, 5]])
    points = np.array([[ 0.41,   0.39,    0.15],       # 0
                       [-0.38,   0.42,   -0.05],     # 1
                       [-0.44,  -0.37,    0.08],      # 2
                       [ 0.40,  -0.28,   -0.04],    # 3
                       [-0.19,   0.01,     0.2],
                       [ 0.21,  -0.02,     0.1],
                       [0.04,    0.07,    -0.1]])

    flysurf = CatenaryFlySurf(9, 9, 0.2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=60)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    for i in range(1000):
        ax.clear()
        time_start = time.time()
        points[1, 2] += oscillation(i)
        points[3, 2] += oscillation(i)
        ax.view_init(elev=45+15*np.cos(i/20), azim=60+0.5*i)
        flysurf.update(points_coord, points)
        print(time.time() - time_start)
        visualize(fig, ax, flysurf, plot_dot=False, plot_curve=True, plot_surface=True)

