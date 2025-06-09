import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.integrate import quad
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from joblib import Parallel, delayed
import time
from scipy.optimize import root_scalar
import matplotlib
from scipy.spatial import distance
from scipy.spatial.distance import cdist

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

""" Utility functions
"""
def estimate_yaw_transformation(points1, points2):
    if points1.shape[0] == 0:
        return 0.0, np.zeros(3)
    
    # Step 1: Compute centroids
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)
    
    # Step 2: Center the points
    centered1 = points1 - centroid1
    centered2 = points2 - centroid2
    
    # Step 3: Extract XY components
    xy1 = centered1[:, :2]
    xy2 = centered2[:, :2]
    
    # Step 4: Compute yaw angle using cross and dot products
    cross = np.sum(xy1[:, 0] * xy2[:, 1] - xy1[:, 1] * xy2[:, 0])
    dot = np.sum(xy1[:, 0] * xy2[:, 0] + xy1[:, 1] * xy2[:, 1])
    yaw = np.arctan2(cross, dot)
    
    # Step 5: Construct 3D rotation matrix
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1]
    ])
    
    # Step 6: Compute translation
    rotated_centroid1 = R @ centroid1
    t = centroid2 - rotated_centroid1
    
    return yaw, t, R

def transform_points(points, yaw, t):
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1]
    ])
    return (R @ points.T).T + t

def catenary_surface(params, u, v):
    a, x_S, y_S, z_S = params
    z = z_S + a * np.cosh(u / a)
    x = x_S + u * np.cos(v)
    y = y_S + u * np.sin(v)  
    return np.column_stack((x, y, z))

def cost_function(params, u, v, data):
    prediction = catenary_surface(params, u, v)
    residuals = prediction - data
    return np.sum(residuals**2)

def interpolate_and_convert_to_polar(p1, p2, n):
    """
    Interpolate between two Cartesian points and convert to polar coordinates.
    
    Args:
        p1: First point as (x1, y1) or (x1, y1, z1)
        p2: Second point as (x2, y2) or (x2, y2, z2)
        n: Number of points to generate (including endpoints)
        
    Returns:
        Tuple of (r_values, theta_values) where:
        - r_values: Array of radii
        - theta_values: Array of angles in radians [-π, π]
        The output arrays have length n.
    """
    # Convert inputs to numpy arrays
    p1 = np.array(p1[:2])[::-1]  # Only use x,y coordinates
    p2 = np.array(p2[:2])[::-1]
    
    # Generate n evenly spaced points between p1 and p2
    t = np.linspace(0, 1, n)
    points = np.outer(1 - t, p1) + np.outer(t, p2)
    
    # Convert to polar coordinates
    x = points[:, 0]
    y = points[:, 1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    return r, theta

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

def find_triangles(points, triangles, tol=1e-12):
    """
    Find containing triangles with fallback to closest edge.
    
    Parameters:
        points: (n_points, 2) array of query points
        triangles: (n_triangles, 3, 2) array of triangle vertices
        tol: Numerical tolerance for point-in-triangle test
        
    Returns:
        (n_points,) array of triangle indices
    """
    n_points = points.shape[0]
    n_triangles = triangles.shape[0]
    
    # Precompute all triangle edges [AB, BC, CA] for each triangle
    edges = np.stack([
        triangles[:, 1] - triangles[:, 0],  # AB
        triangles[:, 2] - triangles[:, 1],  # BC
        triangles[:, 0] - triangles[:, 2]   # CA
    ], axis=1)  # shape (n_triangles, 3, 2)
    
    # Original barycentric coordinate calculation
    A = triangles[:, 0]
    B = triangles[:, 1]
    C = triangles[:, 2]
    
    denominator = ((B[:, 1] - C[:, 1]) * (A[:, 0] - C[:, 0]) + 
                 (C[:, 0] - B[:, 0]) * (A[:, 1] - C[:, 1]))
    
    P = points[:, np.newaxis, :]
    u = ((B[:, 1] - C[:, 1]) * (P[..., 0] - C[:, 0]) + 
        (C[:, 0] - B[:, 0]) * (P[..., 1] - C[:, 1])) / denominator
    v = ((C[:, 1] - A[:, 1]) * (P[..., 0] - C[:, 0]) + 
        (A[:, 0] - C[:, 0]) * (P[..., 1] - C[:, 1])) / denominator
    w = 1 - u - v
    
    contains = (u >= -tol) & (v >= -tol) & (w >= -tol)
    tri_indices = np.argmax(contains, axis=1)
    tri_indices[~np.any(contains, axis=1)] = -1
    
    # Enhanced fallback: find closest edge
    if np.any(tri_indices == -1):
        missing_mask = (tri_indices == -1)
        missing_points = points[missing_mask]
        
        # Compute distance from each missing point to all edges
        min_dists = np.full(len(missing_points), np.inf)
        best_tri = np.full(len(missing_points), -1, dtype=int)
        
        for tri_idx in range(n_triangles):
            # Get all edges for this triangle
            v0, v1, v2 = triangles[tri_idx]
            edges = [
                (v0, v1),  # AB
                (v1, v2),  # BC
                (v2, v0)   # CA
            ]
            
            # Compute distance to each edge
            for edge_idx, (start, end) in enumerate(edges):
                # Vector from start to end
                edge_vec = end - start
                # Vector from start to point
                point_vec = missing_points - start
                
                # Projection parameter (0 to 1 means on segment)
                t = np.dot(point_vec, edge_vec) / np.dot(edge_vec, edge_vec)
                t = np.clip(t, 0, 1)
                
                # Find closest point on edge
                proj = start + t[:, np.newaxis] * edge_vec
                
                # Distance to edge
                dists = np.linalg.norm(missing_points - proj, axis=1)
                
                # Update best match
                better = dists < min_dists
                min_dists[better] = dists[better]
                best_tri[better] = tri_idx
        
        tri_indices[missing_mask] = best_tri
        
    return tri_indices


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


def batch_surf_sampling(intersections, surf_info, flysurf):
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

""" 
Flysurf related classes
"""

class CatenarySurfaceOptimizer:
    def __init__(self, surface_points, num_samples, num_edges):
        self.surface_points = surface_points
        self.x = surface_points[:, 0]
        self.y = surface_points[:, 1]
        self.z = surface_points[:, 2]
        self.n = num_samples
        self.e = num_edges

    def _catenary_surface(self, params):
        """Catenary surface model: z0 + a*cosh(r/a) where r=√[(x-x0)² + (y-y0)²]"""
        a, x0, y0, z0 = params
        r = np.sqrt((self.x - x0) ** 2 + (self.y - y0) ** 2)
        return z0 + a * np.cosh(r / a)

    def objective(self, params):
        """Sum of squared residuals objective function"""
        scaling = 0.1*np.eye(self.n*self.e)
        first_indices = np.arange(0, self.n*self.e, self.n)
        last_indices = np.arange(self.n-1, self.n*self.e, self.n)

        scaling[first_indices, first_indices] *= self.n*10.0
        scaling[last_indices, last_indices] *= self.n*10.0

        model = self._catenary_surface(params)
        return np.sum(scaling*(self.z - model) ** 2)

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
        self.catenary_surface_params = dict()  # [(V1, V2, V3)]: [a, x0, y0, z0, p1, p2, p3]

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

        # for estimating the surface
        time_start_estimating_surface = time.time()
        self._catenary_surface_fitting_v1()
        # input("Press Enter to continue...")  # For debugging purposes
        self.log_surface_time.append(time.time() - time_start_estimating_surface)

    def _catenary_surface_fitting_v1(self):
        num_samples = self.num_samples
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

            initial_guess = self.catenary_surface_params[surface][:4]
            if np.isnan(initial_guess[0]):
                initial_guess[0] = 0.5
            initial_guess = np.nan_to_num(initial_guess, nan=0.0)
            optimizer = CatenarySurfaceOptimizer(surface_points, num_samples, num_edges)
            result = optimizer.fit(initial_guess=initial_guess)
            self.catenary_surface_params[surface][:4] = result.x


    def _catenary_surface_fitting_v2(self, points_coord, points_position):
        yaw, t, R = estimate_yaw_transformation(points_position, np.hstack([points_coord, np.zeros((points_coord.shape[0], 1))]))
        num_samples = self.num_samples 
        for i, surface in enumerate(self.active_surface):
            num_vertices = len(surface)
            num_edges = num_vertices
            surface_points = np.zeros((num_edges * num_samples, 3))
            points_coord_u = np.zeros((num_edges * num_samples))
            points_coord_v = np.zeros((num_edges * num_samples))
            index = 0 

            # we need to find the sample points on the edges of the surface, associating their Cartisian and polar coordinates
            for j in range(num_vertices):
                for k in range(j + 1, num_vertices):
                    # obtain the edges of a surface 
                    edge = tuple(sorted((surface[j], surface[k])))
                    # find the edge endpoints' coordinates in Euclidean Space
                    rl_start = np.array(self._index2coord(edge[0]))*self.cell_length
                    rl_end = np.array(self._index2coord(edge[1]))*self.cell_length
                    # take samples on the edge in polar coordinates
                    u, v = interpolate_and_convert_to_polar(rl_start, rl_end, num_samples)
                    # print(u, v, rl_start, rl_end)
                    # input()
                    points_coord_u[index * num_samples:(index + 1) * num_samples] = u
                    points_coord_v[index * num_samples:(index + 1) * num_samples] = v

                    surface_points[index * num_samples:(index + 1) * num_samples, :] = self.catenary_curve_params[edge][2][-1]
                    index += 1

            # Initial guess
            initial_guess = self.catenary_surface_params[surface][:4]
            if np.isnan(initial_guess[0]):
                initial_guess[0] = 0.5
            initial_guess = np.nan_to_num(initial_guess, nan=0.0)

            # Optimization using minimize
            surface_points = transform_points(surface_points, yaw, t)
            result = minimize(
                fun=cost_function,
                x0=initial_guess,
                args=(points_coord_u, points_coord_v, surface_points),
                method='L-BFGS-B',  # Can also try 'TNC' or 'SLSQP' for bounded problems
                bounds = [
                    (1e-3, None),  # a (must be positive)
                    (None, None),  # x_S
                    (None, None),  # y_S
                    (None, None),  # z_S
                ],
                options={'maxiter': 1000, 'disp': False}  # Set disp=True for debugging
            )
            # print(result.x)
            self.catenary_surface_params[surface][:4] = result.x

    def _catenary_surface(self, params, x, y):
        a, x0, y0, z0 = params
        return z0 + a * np.cosh(np.sqrt((x - x0) ** 2 + (y - y0) ** 2) / a)

    # Define the residual function for least squares
    def _residuals(self, params, x, y, z):
        return z - self._catenary_surface(params, x, y)


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
        # edges_index = np.unique(edges_index, axis=0)

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
            key = [coord1, coord2, coord3]
            val = [points_array[i, :], points_array[j, :], points_array[k, :]]
            key, val = zip(*sorted(zip(key, val)))

            if key not in self.catenary_surface_params:
                self.catenary_surface_params[key] = [0.5, 0, 0, 0, val]
            else:
                self.catenary_surface_params[key][-1] = val
            active_surface.append(key)

        self.active_surface = active_surface
        return edges

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

            c_init, x0_init, y0_init = guesses[i]
            # Perform optimization
            result = minimize(
                self._objective_with_gradient,
                guesses[i],
                args=(x1, z1, x2, z2, length, c_init, x0_init, y0_init),
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

    def _objective_with_gradient(self, params, x1, z1, x2, z2, L, c_init, x0_init, z0_init,
                                 alpha=0.1, beta=0.1, gamma=0.1, zeta=0.1):
        """
        Objective function with gradients and scaling factors.
        alpha: scaling factor for z1 error
        beta: scaling factor for z2 error
        gamma: scaling factor for length error
        zeta: scaling factor for change in parameters
        """
        c, x0, z0 = params

        # Predicted z-values
        z1_pred = c * np.cosh((x1 - x0) / c) + z0
        z2_pred = c * np.cosh((x2 - x0) / c) + z0

        # Exact curve length
        L_cat = self._compute_exact_arc_length(c, x0, x1, x2)

        # Residuals with scaling
        change = (c - c_init) ** 2 + (x0 - x0_init) ** 2 + (z0 - z0_init) ** 2

        # Scaled objective function
        objective_value = (alpha * (z1 - z1_pred) ** 2 +
                           beta * (z2 - z2_pred) ** 2 +
                           gamma * (L - L_cat) ** 2 + 
                           zeta * change)

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
                  - 2 * gamma * (L - L_cat) * dL_dc 
                  + 2 * zeta * (c - c_init))

        grad_x0 = (-2 * alpha * (z1 - z1_pred) * dz1_dx0
                   - 2 * beta * (z2 - z2_pred) * dz2_dx0
                   - 2 * gamma * (L - L_cat) * dL_dx0
                   + 2 * zeta * (x0 - x0_init))

        grad_z0 = (-2 * alpha * (z1 - z1_pred)
                   - 2 * beta * (z2 - z2_pred)
                   + 2 * zeta * (z0 - z0_init))  

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

class FlysurfSampler:
    def __init__(self, flysurf, resolution, points=None, coordinates=None):
        self.flysurf = flysurf
        self.resolution = resolution
        if points is None:
            self.filtered_samples = np.zeros((resolution ** 2, 3))
        else:
            self.filtered_samples = self.sampling_v3(None, None, points, coordinates)
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
                            dist, rotation, translation, samples_per_connection = other_data
                            # Calculate the number of samples needed between index1 and index2
                            num_samples = index2 - index1

                            # Ensure we have at least 1 sample
                            if num_samples > 0:
                                # Generate evenly spaced indices including first and last elements
                                step = len(samples_per_connection) / num_samples
                                indices = [round(i * step) for i in range(num_samples)]
                                
                                if index1 == 0:
                                    if num_samples > 1:
                                        full_curve_global[index1: index2-1, :] = samples_per_connection[indices[1:], :]
                                else:
                                    # Perform the sampling
                                    full_curve_global[index1-1: index2-1, :] = samples_per_connection[indices, :]

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
        # time_start_sampling_v1 = time.time_ns()

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

        # print("Check point 1 (ms):", (time.time_ns() - time_start_sampling_v1) * 1e-6)
        # time_start_sampling_v1 = time.time_ns()
        # print(S[0], S[1])
        intersections = find_all_intersections_batch(S[0], S[1])
        # print("Check point 2 (ms):", (time.time_ns() - time_start_sampling_v1) * 1e-6)
        # time_start_sampling_v1 = time.time_ns()

        surf_info = []
        for i, surface in enumerate(flysurf.active_surface):
            num_edges = len(surface)
            c, x, y, z = flysurf.catenary_surface_params[surface][:4]
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

        # print("Check point 3 (ms):", (time.time_ns() - time_start_sampling_v1) * 1e-6)
        # time_start_sampling_v1 = time.time_ns()

        z_vals = batch_surf_sampling(intersections, surf_info, flysurf)

        # print("Check point 4 (ms):", (time.time_ns() - time_start_sampling_v1)*1e-6)

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
        alpha = np.zeros((num_actuators,)) + 0.1
        if num_actuators > 4:
            sigma[4] -= 0.1
            alpha[4] += 0.5

        all_samples = drag_points_vectorized(all_samples, control_indices, points,
                                             sigma=sigma, alpha=alpha)

        if plot:
            ax.plot(intersections[:, 0], intersections[:, 1], intersections[:, 2], "*")
            plt.pause(0.0001)
        return all_samples

    def sampling_v2(self, fig, ax, points, coordinates, plot=False):
        """ NOTE:
            This sampling method uses an explicit mapping from 
            the mesh grid parametrization to the deformable surface points 
            @params:
                fig, ax: matplotlib figure handles to facilitate the visualization
                plot: set to True to plot on the ax
                flysurf: the flysurf instance
                resolution: the number of samples to take on each boundary ridge
                points: the positions of the actuated mass points on the flysurf
        """
        # collection of all samples
        resolution = self.resolution
        flysurf = self.flysurf
        cell_length = flysurf.cell_length
        yaw, t, R = estimate_yaw_transformation(np.hstack([coordinates, np.zeros((coordinates.shape[0], 1))]), points)

        all_samples = []
        num_samples = resolution ** 2
        triangles = []
        for i, surface in enumerate(flysurf.active_surface):
            triangles.append([flysurf._index2coord(surface[j]) for j in range(len(surface))])

        triangles = np.array(triangles)

        # take samples in the mesh grid, i.e., 
        sample_points_R2 = np.array([flysurf._index2coord(i)[::-1] for i in range(num_samples)])
        sample_regions = find_triangles(sample_points_R2, triangles)

        for i in range(num_samples):
            uvy, uvx = np.array(flysurf._index2coord(i))*cell_length
            triangle = triangles[sample_regions[i]]
            key = tuple(sorted([flysurf._coord2index(triangle_vertex) for triangle_vertex in triangle]))
            u, v = np.sqrt(uvx**2 + uvy**2), np.arctan2(uvy, uvx)
            # print(i, (u, v))
            params = flysurf.catenary_surface_params[key][:4]
            # print(key, params)
            all_samples.append(transform_points(catenary_surface(params, u, v).flatten(), yaw, t))

        # assert False
        return np.array(all_samples)
    
    def sampling_v3_curv(self, fig, ax, points, coordinates, plot=False):
        resolution = self.resolution
        flysurf = self.flysurf
        num_vertices = 3
        num_samples = resolution ** 2
        
        # Precompute triangle data
        active_surfaces = np.array(flysurf.active_surface)
        triangle_coordinates = np.array([
            [flysurf._index2coord(surface[j]) for j in range(num_vertices)]
            for surface in active_surfaces
        ])
        triangle_positions = np.array([flysurf.catenary_surface_params[tuple(surface)][-1] 
                                    for surface in active_surfaces])
        
        # Generate sample grid and find containing triangles
        sample_points_R2 = np.array([flysurf._index2coord(i) for i in range(num_samples)])
        sample_regions = find_triangles(sample_points_R2, triangle_coordinates)
        
        # Vectorized parameter lookup
        triangle_keys = np.array([
            tuple(sorted([flysurf._coord2index(vertex) for vertex in tri]))
            for tri in triangle_coordinates
        ])
        all_params = np.array([flysurf.catenary_surface_params[tuple(key)][:4] for key in triangle_keys])
        
        # Get parameters for each sample point
        sample_params = all_params[sample_regions]
        
        # Vectorized barycentric interpolation
        A = triangle_coordinates[sample_regions].transpose(0, 2, 1)  # Shape: (num_samples, 2, 3)
        A = np.concatenate([A, np.ones((num_samples, 1, 3))], axis=1)  # Add row of ones
        
        b = np.column_stack([sample_points_R2, np.ones(num_samples)])  # Shape: (num_samples, 3)
        lambdas = np.linalg.solve(A, b[:, :, np.newaxis]).squeeze()  # Shape: (num_samples, 3)
        
        # Interpolate (x,y) coordinates
        xyz = triangle_positions[sample_regions]  # Shape: (num_samples, 3, 3)
        xy = np.einsum('ij,ijk->ik', lambdas, xyz)  # Shape: (num_samples, 3)
        
        # Compute z from catenoid equation
        a, x0, y0, z0 = sample_params.T
        r = np.sqrt((xy[:, 0] - x0)**2 + (xy[:, 1] - y0)**2)
        z = z0 + a * np.cosh(r / a)
        
        # Combine results
        all_samples = np.column_stack([xy[:, :2], z])

        return all_samples
    
    def sampling_v3_curv_not_good(self, fig, ax, points, coordinates, plot=False):
        """Catenoid curvature-aware sampling with edge fallback"""
        resolution = self.resolution
        flysurf = self.flysurf
        num_samples = resolution ** 2
        
        # Precompute all triangle data
        active_surfaces = np.array(flysurf.active_surface)
        triangle_coords = np.array([
            [flysurf._index2coord(surface[j]) for j in range(3)]
            for surface in active_surfaces
        ])
        triangle_positions = np.array([
            flysurf.catenary_surface_params[tuple(surface)][-1] 
            for surface in active_surfaces
        ])
        
        # Generate sample grid and find containing triangles
        sample_ij = np.array([flysurf._index2coord(i) for i in range(num_samples)])
        tri_indices = find_triangles(sample_ij, triangle_coords)
        
        # Get parameters for each triangle
        triangle_params = np.array([
            flysurf.catenary_surface_params[tuple(surface)][:4]
            for surface in active_surfaces
        ])
        
        # Initialize output array
        all_samples = np.zeros((num_samples, 3))
        
        # Process each unique triangle found
        for tri_idx in np.unique(tri_indices):
            mask = (tri_indices == tri_idx)
            ij_points = sample_ij[mask]
            tri_ij = triangle_coords[tri_idx]
            tri_xyz = triangle_positions[tri_idx]
            a, x0, y0, z0 = flysurf.catenary_surface_params[tuple(active_surfaces[tri_idx])][:4]
            
            # Compute barycentric coordinates in grid space (i,j) 
            #FIXME: it is problematic when the sample is away from the centroid and the vertices
            # because (i, j) grid space is not isometric to the embedded Euclidean space of the catenoid
            A = np.column_stack([tri_ij, np.ones(3)])
            lambdas = np.linalg.solve(A.T, np.column_stack([ij_points, np.ones(len(ij_points))]).T).T
            
            # Get vertices in Cartesian space
            P0, P1, P2 = tri_xyz[:, :2]
            
            # Catenoid-aware mapping
            # 1. Compute conformal coordinates (s, theta) for triangle vertices (curve lengths, azimuthal angle) from the lowest point
            rs = np.array([np.sqrt((P[0]-x0)**2 + (P[1]-y0)**2) for P in [P0, P1, P2]])
            ss = a * np.sinh(rs / a)
            # ss = np.array([np.sqrt((P[2] - z0)**2 - a**2) for P in [P0, P1, P2] ] )
            ts = np.array([np.arctan2(P[1]-y0, P[0]-x0) for P in [P0, P1, P2]])
 
            # 2. Interpolate in eclidean space intrinsic to the catenoid (x, y)
            us = ss * np.cos(ts)
            vs = ss * np.sin(ts)
            uavg = lambdas.dot(us)
            vavg = lambdas.dot(vs)

            # Then convert back to conformal coordinates
            savg = np.sqrt(uavg**2 + vavg**2)
            tavg = np.arctan2(vavg, uavg)
            
            # 3. Map back to physical space
            ravg = a * np.arcsinh(savg / a)
            x_samples = x0 + ravg * np.cos(tavg)
            y_samples = y0 + ravg * np.sin(tavg)
            z_samples = z0 + np.sqrt(a**2 + savg**2) # z0 + a * np.cosh(ravg / a)
            
            # Store results
            all_samples[mask, 0] = x_samples
            all_samples[mask, 1] = y_samples
            all_samples[mask, 2] = z_samples
        
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
    mesh_size = 25  # number of samples on the outermost sides
    points_coord = np.array([[            mesh_size - 1 ,            mesh_size - 1  ],
                             [            mesh_size - 1 ,                        0  ],
                             [                        0 ,                        0  ],
                             [                        0 ,            mesh_size - 1  ], # corners
                             [     (mesh_size - 1) // 2 ,     (mesh_size - 1) // 2  ], # center
                             [            mesh_size - 1 ,     (mesh_size - 1) // 2  ], 
                             [                        0 ,     (mesh_size - 1) // 2  ],
                             [     (mesh_size - 1) // 2 ,            mesh_size - 1  ],
                             [     (mesh_size - 1) // 2 ,                        0  ], # midpoints
                            #  [            mesh_size - 1 ,     (mesh_size - 1) // 4  ], 
                            #  [            mesh_size - 1 ,   (mesh_size - 1)*3 // 4  ], # top mid-midpoints
                            #  [   (mesh_size - 1)*3 // 4 ,                        0  ],
                            #  [   (mesh_size - 1)*3 // 4 ,     (mesh_size - 1) // 4  ],
                            #  [   (mesh_size - 1)*3 // 4 ,     (mesh_size - 1) // 2  ],
                            #  [   (mesh_size - 1)*3 // 4 ,   (mesh_size - 1)*3 // 4  ],
                            #  [   (mesh_size - 1)*3 // 4 ,          (mesh_size - 1)  ], # top-mid mid-midpoints
                            #  [     (mesh_size - 1) // 2 ,     (mesh_size - 1) // 4  ],
                            #  [     (mesh_size - 1) // 2 ,   (mesh_size - 1)*3 // 4  ], # mid-mid mid-midpoints
                            #  [     (mesh_size - 1) // 4 ,                        0  ],
                            #  [     (mesh_size - 1) // 4 ,     (mesh_size - 1) // 4  ],
                            #  [     (mesh_size - 1) // 4 ,     (mesh_size - 1) // 2  ],
                            #  [     (mesh_size - 1) // 4 ,   (mesh_size - 1)*3 // 4  ],
                            #  [     (mesh_size - 1) // 4 ,          (mesh_size - 1)  ], # bottom-mid mid-midpoints
                            #  [                        0 ,     (mesh_size - 1) // 4  ],
                            #  [                        0 ,   (mesh_size - 1)*3 // 4  ], # bottom mid-midpoints
    ])

    points = np.array([[0.9, 0.4, 0.45],
                       [0.1, 0.4, 0.45],
                       [0.1, -0.4, 0.45],
                       [0.9, -0.4, 0.45],   # corners
                       [0.5, 0., 0.45],     # center
                       [0.5, 0.4, 0.45],
                       [0.5, -0.4, 0.45],
                       [0.9, 0.0, 0.45],
                       [0.1, 0.0, 0.45],    # midpoints
                    #    [0.3, 0.4, 0.45],
                    #    [0.7, 0.4, 0.45],    # top mid-midpoints
                    #    [0.1, 0.2, 0.45],
                    #    [0.3, 0.2, 0.45],
                    #    [0.5, 0.2, 0.45],
                    #    [0.7, 0.2, 0.45],   
                    #    [0.9, 0.2, 0.45],    # top-mid mid-midpoints
                    #    [0.3, 0.0, 0.45],   
                    #    [0.7, 0.0, 0.45],    # mid-mid mid-midpoints
                    #    [0.1, -0.2, 0.45],
                    #    [0.3, -0.2, 0.45],
                    #    [0.5, -0.2, 0.45],
                    #    [0.7, -0.2, 0.45],   
                    #    [0.9, -0.2, 0.45],   # bottom-mid mid-midpoints
                    #    [0.3, -0.4, 0.45],   
                    #    [0.7, -0.4, 0.45],    # mid-mid mid-midpoints
    ])

    flysurf = CatenaryFlySurf(mesh_size, mesh_size, 2.0 / (mesh_size - 1), num_sample_per_curve=mesh_size)
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
            ax.set_xlim(-0.25, 1.25)
            ax.set_ylim(-0.5, 0.75)
            ax.set_zlim(-1.0, 1.0)
            # random_array = np.random.normal(loc=0, scale=0.001, size=points.shape)
            # points += random_array
            points[0, 2] += 0.11 * oscillation(5.0 * i)
            points[1, 2] += 0.12 * oscillation(7.0 * i + 1)
            points[2, 2] -= 0.11 * oscillation(8.0 * i + 1.74)
            points[3, 2] += 0.09 * oscillation(6.0 * i + 4.1)
            points[4, 2] -= 0.12 * oscillation(2.5 * i + 3)
            points[5, 2] -= 0.10 * oscillation(3.1 * i + 1.2)
            points[6, 2] += 0.11 * oscillation(5.1 * i + 2.0)
            points[7, 2] -= 0.09 * oscillation(0.9 * i - 0.5)
            points[8, 2] += 0.11 * oscillation(4.1 * i - 1.1)
            # points[9, 2] += 0.11 * oscillation(5.0 * i)
            # points[10, 2] += 0.12 * oscillation(7.0 * i + 1)
            # points[11, 2] -= 0.11 * oscillation(8.0 * i + 1.74)
            # points[12, 2] += 0.09 * oscillation(6.0 * i + 4.1)
            # points[13, 2] -= 0.12 * oscillation(2.5 * i + 3)
            # points[14, 2] -= 0.10 * oscillation(3.1 * i + 1.2)
            # points[15, 2] += 0.11 * oscillation(5.1 * i + 2.0)
            # points[16, 2] -= 0.09 * oscillation(0.9 * i - 0.5)
            # points[17, 2] += 0.11 * oscillation(4.1 * i - 1.1)
            # points[18, 2] += 0.11 * oscillation(5.0 * i)
            # points[19, 2] += 0.12 * oscillation(7.0 * i + 1)
            # points[10, 2] -= 0.11 * oscillation(8.0 * i + 1.74)
            # points[21, 2] += 0.09 * oscillation(6.0 * i + 4.1)
            # points[22, 2] -= 0.12 * oscillation(2.5 * i + 3)
            # points[23, 2] -= 0.10 * oscillation(3.1 * i + 1.2)
            # points[24, 2] += 0.11 * oscillation(5.1 * i + 2.0)
            points[:, :2] += np.random.normal(loc=0, scale=0.005, size=points[:, :2].shape)

            # ax.view_init(elev=45+15*np.cos(i/17), azim=60+0.45*i)
            time_start = time.time()
            sampler.flysurf.update(points_coord, points)
            print("elapsed time till update:", time.time() - time_start)
            # visualize(fig, ax, flysurf, plot_dot=False, plot_curve=True, plot_surface=False, num_samples=25)

            all_samples = sampler.sampling_v3_curv(fig, ax, points, coordinates=points_coord)
            print("elapsed time till sampling:", time.time() - time_start)
            vel_raw_hist.append(np.linalg.norm((all_samples - sampler.filtered_samples)[:, 1]))

            filtered_points = sampler.smooth_particle_cloud(all_samples, max_speed, dt)
            print("total elapsed time:", time.time() - time_start)
            vel_hist.append(np.linalg.norm(sampler.vel[:, 1]))

            # Before filter
            # ax.plot(all_samples[:, 0], all_samples[:, 1], all_samples[:, 2], "*")
            #
            # unfiltered_samples_rows = all_samples.reshape((mesh_size, mesh_size, 3))

            # for i in range(unfiltered_samples_rows.shape[0]):
            #     # ax.plot(unfiltered_samples_rows[i, 0:2, 0], unfiltered_samples_rows[i, 0:2, 1], unfiltered_samples_rows[i, 0:2, 2], "o-")
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
        ax1.plot(flysurf.log_catenary_time, "r.", label="catenary time")
        ax1.plot(flysurf.log_surface_time, "b.", label="surface time")
        ax1.legend()
        plt.pause(0.0001)
        input()