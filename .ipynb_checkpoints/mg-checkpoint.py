# This is a collection of methods of determining the 3D shape of a simulated galaxy.
#
#
# The functions find_galaxy_shape2d and find_galaxy_shape3d are designed to load/process galaxies, 
# measure them, and store the data into csv files. They are not required to run the iterative tensor. 
# 
#
# The main functions of this script are the iterative reduced inertia tensor, 
# which measures the 2D/3D shape of a galaxy. 
#
#
# Input the centered galaxy data as = (['Coordinates'], ['Masses'])
# Required packages: 
# galaxy_tools (see Courtney Klein https://github.com/courtk32/mockobservation-tools/blob/master/mockobservation_tools/galaxy_tools.py )


# set up
import galaxy_tools
import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
import copy
import os
import time
from astropy.cosmology import Planck13

def ellipsoidal_radius_2d(coord, a, b):
    """
    Calculates the ellipsoidal radius for a given point and semi-axes lengths in 2D.

    Parameters
    ----------
    coord : array_like
        Coordinates of the point (x, y).
    a, b : float
        Semi-axis lengths along the x-axis and y-axis respectively.

    Returns
    -------
    radius : float
        The ellipsoidal radius.
    """
    return np.sqrt((coord[0]**2 / (a**2)) + (coord[1]**2 / (b**2)))
    
def ellipsoidal_radius_3d(coord, a, b, c):
    """
    Calculates the ellipsoidal radius for a given point and semi-axes lengths.

    Parameters
    ----------
    coord : array_like
        Coordinates of the point (x, y, z).
    a, b, c : float
        Semi-axis lengths along the x-axis, y-axis, and z-axis respectively.

    Returns
    -------
    radius : float
        The ellipsoidal radius.
    """
    return np.sqrt((coord[0]**2 / (a**2)) + (coord[1]**2 / (b**2)) + (coord[2]**2 / (c**2)))
def ellipsoidal_radius_3d_vectorized(coordinates, a, b, c):
    """
    Calculates the ellipsoidal radius for a given set of points and semi-axes lengths using vectorized operations.

    Parameters
    ----------
    coordinates : array_like
        Array of coordinates of the points (N, 3).
    a, b, c : float
        Semi-axis lengths along the x-axis, y-axis, and z-axis respectively.

    Returns
    -------
    radii : ndarray
        Array of ellipsoidal radii.
    """
    coordinates = np.array(coordinates)
    x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    radii = np.sqrt((x**2 / a**2) + (y**2 / b**2) + (z**2 / c**2))
    return radii
    
def principal_axes_2d(matrix):
    """
    Calculate the lengths and directions of the principal axes of an ellipsoid from its covariance  matrix in 2D.

    Parameters
    ----------
    matrix : array_like
        Covariance matrix of the ellipsoid.

    Returns
    -------
    A, B : float
        Largest and smallest eigenvalues (in order).
    Av, Bv : array_like
        Eigenvectors of the principal axes A and B (in order).
    """
    vals, vecs = np.linalg.eig(matrix)
    lengths = np.sqrt(np.abs(vals))
    indices = np.argsort(lengths)[::-1]
    lengths = lengths[indices]
    vecs = vecs[:, indices]
    A, B = lengths
    Av, Bv = vecs.T

    return A, B, Av, Bv


def principal_axes_3d(matrix):
    """
    Calculate the lengths and directions of the principal axes of an ellipsoid from its covariance matrix.

    Parameters
    ----------
    matrix : array_like
        Covariance matrix of the ellipsoid.

    Returns
    -------
    A, B, C : float
        Largest, intermediate, and smallest eigenvalues (in order).
    Av, Bv, Cv : array_like
        Eigenvectors of the principal axes A, B, and C (in order).
    """
    vals, vecs = np.linalg.eig(matrix)
    lengths = np.sqrt(np.abs(vals))
    indices = np.argsort(lengths)[::-1]
    lengths = lengths[indices]
    vecs = vecs[:, indices]

    A, B, C = lengths
    Av, Bv, Cv = vecs.T

    return A, B, C, Av, Bv, Cv

def vals(lengths):
    """
    Calculate the axis ratios, ellipticity, and triaxiality from the principal axes lengths.

    Parameters
    ----------
    lengths : array_like
        Lengths of the principal axes.

    Returns
    -------
    axis_ratios : tuple
        Tuple containing the two axis ratios (a/b, b/c).
    ellipticity : float
        Ellipticity of the shape (1 - c/a).
    triaxiality : float
        Triaxiality parameter ((a^2 - b^2) / (a^2 - c^2)).
    """
    a, b, c = lengths[:3]  # Take the first three lengths in case there are more
    axis_ratios = (b / a, c / a)
    ellipticity = 1 - c / a
    triaxiality = (a**2 - b**2) / (a**2 - c**2)
    return axis_ratios, ellipticity, triaxiality
    
def vals_2d(lengths):
    """
    Calculate the axis ratio and ellipticity from the principal axes lengths in 2D.

    Parameters
    ----------
    lengths : array_like
        Lengths of the principal axes.

    Returns
    -------
    axis_ratio : float
        Ratio of the semi-axes lengths (b/a).
    ellipticity : float
        Ellipticity of the shape (1 - b/a).
    """
    a, b = lengths[:2]  # Take the first two lengths in case there are more
    axis_ratio = b / a
    ellipticity = 1 - b / a
    return axis_ratio, ellipticity



def reduced_tensor_2d(star_center, a, b):
    """
    Calculates the reduced inertia tensor of a galaxy in 2D, where particles are downweighted by a factor of
    1 / r_p^2, where r_p is the ellipsoidal radius of each particle corresponding to the ellipsoid with axes a and b.

    Parameters
    ----------
    star_center : dict
        Input a centered galaxy with keys ['Coordinates'] and ['Masses'].

    a, b : float
        Initial ellipsoid semi-axes lengths.

    Returns
    -------
    Mr : np.ndarray
        Reduced inertia tensor matrix.
    A, B : float
        Principal axes lengths.
    axis_ratio : float
        Ratio of the semi-axes lengths (b/a).
    ellipticity : float
        Ellipticity of the shape (1 - b/a).
    """
    coordinates = star_center['Coordinates']
    masses = star_center['Masses']
    ellipsoidal_radii = np.array([ellipsoidal_radius_2d(coord, a, b) for coord in coordinates])
    inverse_radii_squared = 1.0 / ellipsoidal_radii**2
    weighted_coords = np.array([m * inv_r2 * np.outer(coord, coord) for m, inv_r2, coord in zip(masses, inverse_radii_squared, coordinates)])
    Mr = np.sum(weighted_coords, axis=0) / np.sum(masses * inverse_radii_squared)
    A, B, Av, Bv = principal_axes_2d(Mr)
    
    axis_ratio, ellipticity = vals_2d([A, B])
    return Mr, A, B, axis_ratio, ellipticity
'''
def reduced_tensor_3d(star_center, a, b, c):
    """
    Calculates the reduced inertia tensor of a galaxy, where particles are downweighted by a factor of
    1 / r_p^2, where r_p is the ellipsoidal radius of each particle corresponding to the ellipsoid with axes a, b, c. 

    Parameters
    ----------
    star_center :  Input a centered galaxy with (['Coordinates'], ['Masses'])

    a, b, c :  Initial ellipsoid to use
    
    Returns
    -------
    Mr :  Reduced inertia tensor matrix

    """
    coordinates = star_center['Coordinates']
    masses = star_center['Masses']
    ellipsoidal_radii = np.array([ellipsoidal_radius_3d(coord, a, b, c) for coord in coordinates]) # !!!!
    inverse_radii_squared = np.abs(1.0 / ellipsoidal_radii**2) # change back to 1.0 / ellipsoidal_radii**2
    weighted_coords = np.array([m * inv_r2 * np.outer(coord, coord) for m, inv_r2, coord in zip(masses, inverse_radii_squared, coordinates)])
    Mr = np.sum(weighted_coords, axis=0) / np.sum(masses * inverse_radii_squared)
    A, B, C, Av, Bv, Cv = principal_axes_3d(Mr)
    
    axis_ratios, ellipticity, triaxiality = vals([A, B, C])

    return Mr, A, B, C, axis_ratios[0], axis_ratios[1], ellipticity, triaxiality
'''

def reduced_tensor_3d(star_center, a, b, c):
    """
    Calculates the reduced inertia tensor of a galaxy, where particles are downweighted by a factor of
    1 / r_p^2, where r_p is the ellipsoidal radius of each particle corresponding to the ellipsoid with axes a, b, c. 

    Parameters
    ----------
    star_center : dict
        Dictionary containing 'Coordinates' and 'Masses' of the galaxy.

    a, b, c : float
        Semi-axis lengths along the x-axis, y-axis, and z-axis respectively.
    
    Returns
    -------
    Mr : ndarray
        Reduced inertia tensor matrix.
    A, B, C : float
        Principal axes lengths.
    axis_ratio_1, axis_ratio_2 : float
        Axis ratios.
    ellipticity : float
        Ellipticity of the galaxy.
    triaxiality : float
        Triaxiality of the galaxy.
    """
    coordinates = star_center['Coordinates']
    masses = star_center['Masses']
    
    # Calculate ellipsoidal radii using vectorized function
    ellipsoidal_radii = ellipsoidal_radius_3d_vectorized(coordinates, a, b, c)
    inverse_radii_squared = 1.0 / ellipsoidal_radii**2
    
    # Calculate weighted coordinates tensor
    weighted_coords = masses[:, np.newaxis, np.newaxis] * inverse_radii_squared[:, np.newaxis, np.newaxis] * np.einsum('ij,ik->ijk', coordinates, coordinates)
    Mr = np.sum(weighted_coords, axis=0) / np.sum(masses * inverse_radii_squared)
    
    A, B, C, Av, Bv, Cv = principal_axes_3d(Mr)
    axis_ratios, ellipticity, triaxiality = vals([A, B, C])

    return Mr, A, B, C, axis_ratios[0], axis_ratios[1], ellipticity, triaxiality

def uniform_random_rotation(x):
    """Apply a random rotation in 3D, with a distribution uniform over the
    sphere.

    Arguments:
        x: vector or set of vectors with dimension (n, 3), where n is the
            number of vectors

    Returns:
        Array of shape (n, 3) containing the randomly rotated vectors of x,
        about the mean coordinate of x, and the rotation matrix used.
    """

    def generate_random_z_axis_rotation():
        """Generate random rotation matrix about the z axis."""
        R = np.eye(3)
        x1 = np.random.rand()
        R[0, 0] = R[1, 1] = np.cos(2 * np.pi * x1)
        R[0, 1] = -np.sin(2 * np.pi * x1)
        R[1, 0] = np.sin(2 * np.pi * x1)
        return R

    # There are two random variables in [0, 1) here (naming is same as paper)
    x2 = 2 * np.pi * np.random.rand()
    x3 = np.random.rand()

    # Rotation of all points around x axis using matrix
    R = generate_random_z_axis_rotation()
    v = np.array([
        np.cos(x2) * np.sqrt(x3),
        np.sin(x2) * np.sqrt(x3),
        np.sqrt(1 - x3)
    ])
    H = np.eye(3) - (2 * np.outer(v, v))
    M = -(H @ R)
    x = x.reshape((-1, 3))
    mean_coord = np.mean(x, axis=0)
    rotated_coords = ((x - mean_coord) @ M) + mean_coord @ M

    return rotated_coords, M

def csv_2d(folder_path, file_name, data):
    """
    Save simulation data to a CSV file in a specified folder.

    Parameters
    ----------
    folder_path : str
        The path to the folder where the file will be saved.
    file_name : str
        The name of the file to save the data to.
    data : list of dict
        The data to be saved, where each dictionary contains the following keys:
        'simulation', 'a', 'b', 'c', 'b/a', 'c/a', 'c/b', 'ellipticity', 'triaxiality', 'iterations', 'initial_particles', 'remaining_particles', 'max_radius'.

    Returns
    -------
    None
    """
    # Ensure the folder path exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Combine folder path and file name to create the full file path
    full_file_path = os.path.join(folder_path, file_name)
    
    # Write data to CSV file
    with open(full_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Simulation', 'a', 'b', 'b/a', 'Iterations', 'Initial Particles', 'Remaining Particles', 'Max Radius', 'r_50', 'mass'])
        for entry in data:
            writer.writerow([entry['simulation'], entry['a'], entry['b'], entry['b/a'], entry['iterations'], entry['initial_particles'], entry['remaining_particles'], entry['max_radius'], entry['r_50'], entry['mass']])
            
def csv_3d(folder_path, file_name, data):
    """
    Save simulation data to a CSV file in a specified folder.

    Parameters
    ----------
    folder_path : str
        The path to the folder where the file will be saved.
    file_name : str
        The name of the file to save the data to.
    data : list of dict
        The data to be saved, where each dictionary contains the following keys:
        'simulation', 'a', 'b', 'c', 'b/a', 'c/a', 'c/b', 'ellipticity', 'triaxiality', 'iterations', 'initial_particles', 'remaining_particles', 'max_radius'.

    Returns
    -------
    None
    """
    # Ensure the folder path exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Combine folder path and file name to create the full file path
    full_file_path = os.path.join(folder_path, file_name)
    
    # Write data to CSV file
    with open(full_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Simulation', 'a', 'b', 'c', 'b/a', 'c/a', 'c/b', 'ellipticity', 'triaxiality', 'Iterations', 'Initial Particles', 'Remaining Particles', 'Max Radius', 'r_50', 'mass', 'Age Range', 'Snapshot', 'Redshift', 'Center of Mass', 'Rotation Matrix'])
        for entry in data:
            writer.writerow([entry['simulation'], entry['a'], entry['b'], entry['c'], entry['b/a'], entry['c/a'], entry['c/b'], entry['ellipticity'], entry['triaxiality'], entry['iterations'], entry['initial_particles'], entry['remaining_particles'], entry['max_radius'], entry['r_50'], entry['mass'], entry['age_range'], entry['snap_num'], entry['redshift'], entry['cm'], entry['rotation_matrix']])

# def rotation_angle(n):

def center_mass(
    star):
    
    '''
    Calculates the center of mass given coordinates and masses
    Works with any M dimensional coordinate system with N particles
    Does NOT like mass shape as (N,1)
    
    Parameters
    ----------
    coords: array_like, shape (N,M), coordinates, any unit
    mass:   array_like, shape (N,), mass, any unit
    
    Returns
    -------
    cm: array_like, shape (M,)
        center of mass in same units as coords
    
    Example
    -------
    If you have 3 seperate coords arrays use this set up:
    xcm, ycm, zcm = center_mass(np.transpose([xcoord,ycoord,zcoord]), mass)
    '''
    coords = np.array(star['Coordinates'])
    masses = np.array(star['Masses'])
    cm = masses.dot(coords) / np.sum(masses)
    coords = coords - cm
    star['Coordinates'] = coords.tolist()
    return star, cm


def iter_RIT_2d(star_input, a=None, b=None, c=None, max_iterations=20, tolerance=1e-4, initial_max_radius=400.0, particle_fraction=1.0):
    star = star_input
    star, cm = center_mass(star)
    total_mass = np.sum(star['Masses'])

    coords = star['Coordinates']
    if len(coords) < 1000:
        print('galaxy too small: only has ' + str(len(coords)) + ' star particles')
        return None
    if len(coords) > 5000000:
        print('big galaxy: has ' + str(len(coords)) + ' star particles, will take a long time')
    
    masses = star['Masses']
    coords = np.array(coords)
    masses = np.array(masses)
    # Mask out particles with position 0 0 0
    mask = np.any(coords != 0, axis=1)
    coords = coords[mask]
    masses = masses[mask]
    
    # Apply particle fraction to randomly delete particles
    particle_mask = np.random.rand(len(coords)) < particle_fraction
    coords = coords[particle_mask]
    masses = masses[particle_mask]

    # Update the star with the filtered coordinates and masses
    star['Coordinates'] = coords
    star['Masses'] = masses
    
    # rotate galaxy by a random angle
    rotated_coords, rotation_matrix = uniform_random_rotation(coords)
    star['Coordinates'] = rotated_coords
    xy = rotated_coords[:, [0, 1]]
    yz = rotated_coords[:, [1, 2]]
    zx = rotated_coords[:, [2, 0]]
    # HAHAHAAH AMONG US I LOVE AMONGUS
    # Create separate dictionaries for each projection
    star_xy = {'Coordinates': xy, 'Masses': masses}
    star_yz = {'Coordinates': yz, 'Masses': masses}
    star_zx = {'Coordinates': zx, 'Masses': masses}
    projections = {
        'xy': star_xy,
        'yz': star_yz,
        'zx': star_zx
    }
    results = {}
    for name, projection in projections.items():
        coords = projection['Coordinates']
        masses = projection['Masses']
        prev_a, prev_b = None, None
        I, A, B, _, _ = reduced_tensor_2d(projection, a=initial_max_radius, b=initial_max_radius)
        a, b = A, B
        A, B, Av, Bv = principal_axes_2d(I)
        # Align the galaxy with the principal axes
        V0 = np.stack([Av, Bv], axis=-1)  
        V1 = np.eye(2)  
        inv_matrix = np.linalg.solve(V0, V1)
        # rotation_matrix = np.linalg.inv(inv_matrix)

        # Rotate and align the galaxy into the ellipsoid
        coords_rotated = [list(inv_matrix.dot(coord)) for coord in coords]
        star_proj = {'Coordinates': coords_rotated, 'Masses': masses}

        prev_a, prev_b = a, b
        r = np.array([ellipsoidal_radius_2d(coord, a, b) for coord in coords])
        max_radius = np.max(r)
        initial_volume = a * b 
        
        coords = star_proj['Coordinates']
        masses = star_proj['Masses']
        
        initial_particles = len(coords)
        
        for iteration in range(max_iterations):
            # Calculate the reduced inertia tensor and its eigenvalues/eigenvectors
            I, _, _, _, _ = reduced_tensor_2d(star_proj, a, b)
            A, B, Av, Bv = principal_axes_2d(I)
            # Align the galaxy with the principal axes
            
            V0 = np.stack([Av, Bv], axis=-1)  
            V1 = np.eye(2)  # Standard basis vectors in 3D
            inv_matrix = np.linalg.solve(V0, V1)
            rotation_matrix = np.linalg.inv(inv_matrix)
            # Recalculate the reduced tensor
            star_rotated = {'Coordinates': coords_rotated, 'Masses': masses}
            I, A, B, C, _ = reduced_tensor_2d(star_rotated, a, b)
            a_new, b_new = A, B
            # Normalize the updates to maintain constant volume
            volume_factor = (a_new * b_new ) / initial_volume
            a_new /= volume_factor ** (1/2)
            b_new /= volume_factor ** (1/2)
    
            # Check for convergence based on axis ratio
            if np.abs(b_new/a_new - prev_b/prev_a) < tolerance:
                star_proj = star_rotated
                break
    
            a, b = a_new, b_new
            prev_a, prev_b = a, b 
    
            # Exclude particles outside the ellipse
            ellipsoidal_radii_new = np.array([ellipsoidal_radius_2d(coord, a, b) for coord in coords_rotated])
            mask = ellipsoidal_radii_new <= max_radius
            coords_rotated = [coord for coord, m in zip(coords_rotated, mask) if m]
            masses = masses[mask]
            star_proj = {'Coordinates': coords_rotated, 'Masses': masses}
            # Update max radius
            max_radius = np.max(ellipsoidal_radii_new)
            
        coords = np.array(star_proj['Coordinates'])
        masses = np.array(star_proj['Masses'])
        r = np.array([ellipsoidal_radius_2d(coord, a, b) for coord in coords])
        sorted_indices = np.argsort(r)
        sorted_masses = masses[sorted_indices]
        cumulative_mass = np.cumsum(sorted_masses)
        np.sum(star_proj['Masses'])
        mass_limit = 0.9 * total_mass
        limit_index = np.searchsorted(cumulative_mass, mass_limit)
        if limit_index >= len(r):
            limit_index = len(r) - 1
        scaling_radius = r[sorted_indices[limit_index]]
    
        #calculate r50
        mass_limit_50 = 0.5 * total_mass
        limit_index_50 = np.searchsorted(cumulative_mass, mass_limit_50)
        if limit_index_50 >= len(r):
            limit_index_50 = len(r) - 1
        scaling_radius_50 = r[sorted_indices[limit_index_50]]
        mask50 = r <= scaling_radius_50
        coords50 = coords[mask50]
        r_50 = np.max(coords50[:, 0])
        
        # Update star to only include particles within scaling_radius (set at r90)
        mask = r <= scaling_radius
        star_proj['Coordinates'] = coords[mask]
        star_proj['Masses'] = masses[mask]
        remaining_particles = len(star_proj['Masses'])
        coords = star_proj['Coordinates']
        max_radius = np.max(coords[:, 0])
        mass = np.sum(star_proj['Masses'])
        results[name] = {
            'projection': name,
            'a': a,
            'b': b,
            'b/a': b / a,
            'iterations': iteration + 1,
            'initial_particles': initial_particles,
            'remaining_particles': remaining_particles,
            'scaling_radius': scaling_radius,
            'max_radius': max_radius,
            'r_50': r_50,
            'mass': mass,
        }
    return results

def partition_age(star_center_copy, age_partitions):
    """
    Partition star_center_copy into groups based on StellarAge and create new star_centers.

    Parameters
    ----------
    star_center_copy : dict
        Dictionary containing the star data, including 'StellarAge'.
    age_partitions : list of int
        List of age partitions to divide the stars.

    Returns
    -------
    partitioned_star : list of dict
        List of star_center dictionaries for each age partition.
    """
    age_partitions = [0] + age_partitions + [float('inf')]  # Ensure partitions cover all ages
    partitioned_star = []

    for i in range(len(age_partitions) - 1):
        lower_bound = age_partitions[i]
        upper_bound = age_partitions[i + 1]

        # Filter stars based on the age partition
        mask = (star_center_copy['StellarAge'] >= lower_bound) & (star_center_copy['StellarAge'] < upper_bound)
        new_star_center = {key: value[mask] for key, value in star_center_copy.items()}

        partitioned_star.append(new_star_center)

    return partitioned_star


def iter_RIT_3d(star_input, max_iterations=30, tolerance=1e-4, initial_max_radius=400.0, particle_fraction=1.0, particle_bound = 1000):
    """
    Perform an iterative calculation of the Reduced Inertia Tensor (RIT) for a galaxy, 
    holding the volume constant.

    Parameters
    ----------
    star_input : dict
        Input a centered galaxy with (['Coordinates'], ['Masses']).

    a, b, c : float, optional
        Initial principal axes of the ellipsoid. If not provided, they are calculated 
        from the first iteration.

    max_iterations : int, optional
        Maximum number of iterations to perform. Default is 20.

    tolerance : float, optional
        Convergence criterion. The iteration stops when the relative change in axis 
        ratios is less than this value. Default is 1e-4.

    initial_max_radius : float, optional
        Initial maximum radius to consider for particles. 
        Default is 30.0, but try setting it to a scale factor times the virial radius.

    particle_fraction : float, optional
        Fraction of particles to randomly keep. Default is 1.0 (keep all particles).

    Returns
    -------
    result : dict or None
        A dictionary containing the final values of the principal axes ('a', 'b', 'c'),
        the axis ratios b/a and c/a, the ellipticity and triaxiality of the ellipsoid, 
        the number of iterations performed ('iterations'), the number of remaining 
        particles after the iterations ('remaining_particles'), and the maximum 
        ellipsoidal radius used ('max_radius'). Returns None if the total mass of 
        the galaxy is less than 1000, indicating that the calculation cannot be 
        performed effectively.
    """

    cumulative_time = 0
    star = star_input
    min_age = np.min(star['StellarAge'])
    max_age = np.max(star['StellarAge'])
    age_range = [min_age, max_age]
    total_mass = np.sum(star['Masses'])
    coords = np.array(star['Coordinates'])
    masses = np.array(star['Masses'])
    initial_particles = len(coords)
    
    if initial_particles < particle_bound:
        print(f'Galaxy too small: only has {initial_particles} star particles')
        return None
    
    if initial_particles > 5000000:
        print(f'Big galaxy: has {initial_particles} star particles, will take a long time')
    
    # Mask out particles outside of initial max radius
    distances = np.linalg.norm(coords, axis=1)
    initial_mask = distances <= initial_max_radius
    coords = coords[initial_mask]
    masses = masses[initial_mask]
    # Apply particle fraction to randomly delete particles
    particle_mask = np.random.rand(len(coords)) < particle_fraction
    coords = coords[particle_mask]
    masses = masses[particle_mask]

    # Update the star with the filtered coordinates and masses
    star['Coordinates'] = coords
    star['Masses'] = masses
    # Center the galaxy
    star, cm = center_mass(star)
    
    # Mask out particles with position 0 0 0
    coords = np.array(star['Coordinates'])
    masses = np.array(star['Masses'])
    
    mask = np.any(coords != 0, axis=1)
    coords = coords[mask]
    masses = masses[mask]

    # Update the star with the filtered coordinates and masses
    star['Coordinates'] = coords
    star['Masses'] = masses
    a, b, c = None, None, None
    prev_a, prev_b, prev_c = None, None, None

    max_radius = initial_max_radius

    # First iteration, calculate the initial ellipsoid given the principal axes of the 
    # reduced inertia tensor of all particles within the max radius
    I, A, B, C, _, _, _, _ = reduced_tensor_3d(star, a=initial_max_radius, b=initial_max_radius, c=initial_max_radius)
    a, b, c = A, B, C
    A, B, C, Av, Bv, Cv = principal_axes_3d(I)
    # Align the galaxy with the principal axes
    V0 = np.stack([Av, Bv, Cv], axis=-1)  # Shape: (npts, 3, 3)
    V1 = np.eye(3)  # Standard basis vectors in 3D
    inv_matrix = np.linalg.solve(V0, V1)
    # rotation_matrix = np.linalg.inv(inv_matrix)
    final_matrix = inv_matrix
    # Rotate and align the galaxy into the ellipsoid
    coords_rotated = np.dot(star['Coordinates'], inv_matrix)
    ## coords_rotated = coords.dot(inv_matrix.T) 
    ## coords_rotated = [list(inv_matrix.dot(coord)) for coord in coords]
    star['Coordinates'] = coords_rotated
    star['Masses'] = masses

    
    prev_a, prev_b, prev_c = a, b, c
    ## r = np.array([ellipsoidal_radius_3d(coord, a, b, c) for coord in coords])
    r = ellipsoidal_radius_3d_vectorized(coords_rotated, a, b, c)
    max_radius = np.max(r)
    star_rotated = star
    # Calculate the initial volume
    initial_volume = a * b * c
    coords = star['Coordinates']
    masses = star['Masses']
    for iteration in range(max_iterations):
        iteration_start_time = time.time()

        # Calculate the reduced inertia tensor, eigenvalues/eigenvectors of the galaxy
        I, _, _, _, _, _, _, _ = reduced_tensor_3d(star, a, b, c)
        A, B, C, Av, Bv, Cv = principal_axes_3d(I)
        
        # Rotate and align the galaxy with the principal axes
        V0 = np.stack([Av, Bv, Cv], axis=-1)  # Shape: (npts, 3, 3)
        V1 = np.eye(3)  # Standard basis vectors in 3D
        inv_matrix = np.linalg.solve(V0, V1)
        final_matrix = final_matrix @ inv_matrix
        coords_rotated = np.dot(star['Coordinates'], inv_matrix)
        
        # Recalculate the reduced tensor on rotated system
        star_rotated['Coordinates'] = coords_rotated
        star_rotated['Masses'] = masses
        I, A, B, C, _, _, _, _ = reduced_tensor_3d(star_rotated, a, b, c)
        a_new = A
        b_new = B
        c_new = C

        # Normalize the updates to maintain constant volume
        volume_factor = (a_new * b_new * c_new) / initial_volume
        a_new /= volume_factor ** (1/3)
        b_new /= volume_factor ** (1/3)
        c_new /= volume_factor ** (1/3)

        # Check for convergence based on axis ratios
        if np.abs(b_new/a_new - prev_b/prev_a) < tolerance and np.abs(c_new/a_new - prev_c/prev_a) < tolerance:
            star = star_rotated
            break

        a, b, c = a_new, b_new, c_new
        prev_a, prev_b, prev_c = a, b, c

        # Exclude particles outside the ellipsoid
        ## ellipsoidal_radii_new = np.array([ellipsoidal_radius_3d(coord, a, b, c) for coord in coords_rotated])
        ellipsoidal_radii_new = ellipsoidal_radius_3d_vectorized(coords_rotated, a, b, c)

        mask = ellipsoidal_radii_new <= max_radius
        
        # Update ellipsoid and galaxy
        ## coords_rotated = [coord for coord, m in zip(coords_rotated, mask) if m]
        coords_rotated = coords_rotated[mask]
        masses = masses[mask]
        star = {'Coordinates': coords_rotated, 'Masses': masses}
        
        # Update max radius
        max_radius = np.max(ellipsoidal_radii_new)

        # Calculate the elapsed time for each iteration
        elapsed_time = time.time() - iteration_start_time
        cumulative_time += elapsed_time
        #print(f"Iteration {iteration + 1} took {elapsed_time:.2f} seconds (Cumulative: {cumulative_time:.2f} seconds)")
    
    # Scale ellipsoid to 90% of original mass
    coords = np.array(star['Coordinates'])
    masses = np.array(star['Masses'])
    total_mass = np.sum(star['Masses'])
    r = np.array([ellipsoidal_radius_3d(coord, a, b, c) for coord in coords])
    sorted_indices = np.argsort(r)
    sorted_masses = masses[sorted_indices]
    cumulative_mass = np.cumsum(sorted_masses)
    np.sum(star['Masses'])
    mass_limit = 0.9 * total_mass
    limit_index = np.searchsorted(cumulative_mass, mass_limit)
    if limit_index >= len(r):
        limit_index = len(r) - 1
    scaling_radius = r[sorted_indices[limit_index]]
    # Calculate r50
    mass_limit_50 = 0.5 * total_mass
    limit_index_50 = np.searchsorted(cumulative_mass, mass_limit_50)
    if limit_index_50 >= len(r):
        limit_index_50 = len(r) - 1
    scaling_radius_50 = r[sorted_indices[limit_index_50]]
    mask50 = r <= scaling_radius_50
    coords50 = coords[mask50]
    r_50 = np.max(coords50[:, 0])
    
    # Update star to only include particles within scaling_radius
    maskSR = r <= scaling_radius
    star['Coordinates'] = coords[maskSR]
    star['Masses'] = masses[maskSR]
    remaining_particles = len(star['Masses'])
    coords = star['Coordinates']
    max_radius = np.max(coords[:, 0])
    mass = np.sum(star['Masses'])
    return {
        'a': a,
        'b': b,
        'c': c,
        'b/a': b / a,
        'c/a': c / a,
        'c/b': c / b,
        'ellipticity': 1 - c / a,
        'triaxiality': (a**2 - b**2) / (a**2 - c**2),
        'iterations': iteration + 1,
        'initial_particles': initial_particles,
        'remaining_particles': remaining_particles,
        'max_radius': max_radius,
        'r_50': r_50,
        'mass': mass,
        'remaining_galaxy': star,
        'cm': cm,
        'age_range': age_range,
        'rotation_matrix' : final_matrix
    }


def find_galaxy_shape2d(sim_inputs, output_file, snap_num_input=1200, rvir_scales=[0.1], particle_fractions=[1.0], repeats=1, FIREBox=0):
    """
    Analyze the shape of galaxies from simulation data and save the results to a CSV file.

    Parameters
    ----------
    sim_inputs : list of str
        A list of simulation identifiers.
    snap_num_input : int, optional
        Snapshot number to load from the simulation (default is 1200).
    rvir_scales : list of float, optional
        Scale factors to apply to the virial radius of the halo (default is [0.1]).
    particle_fractions : list of float, optional
        Fractions of particles to analyze for shape determination (default is [1.0]).
    repeats : int, optional
        Number of repeats for each simulation identifier (default is 1).
    FIREBox : int, optional
        Flag indicating whether the simulation data is from FIREBox (default is 0).

    Returns
    -------
    results : list of dict
        A list of dictionaries, each containing the shape parameters of a galaxy. Each dictionary includes:
        - 'simulation': str, Simulation identifier.
        - 'projection': str, Indicates the projection plane ('xy', 'yz', 'zx').
        - 'a': float, Length of the principal axis corresponding to the largest eigenvalue.
        - 'b': float, Length of the principal axis corresponding to the intermediate eigenvalue.
        - 'b/a': float, Axis ratio b/a.
        - 'iterations': int, Number of iterations in the shape finding algorithm.
        - 'initial_particles': int, Number of initial particles before masking.
        - 'remaining_particles': int, Number of particles remaining after masking.
        - 'max_radius': float, Maximum radius considered in the analysis.
        - 'r_50': float, Half-mass radius of the galaxy.
        - 'mass': float, Mass of the galaxy after masking.
    """
    results = []
    k = repeats
    for sim in sim_inputs:
        repeats = 1
        start_time = time.time()
        # load halo and snapdicts
        snap_num = snap_num_input
        if FIREBox == 0:
            sim_path = '/DFS-L/DATA/cosmo/grenache/aalazar/FIRE/GVB/' + str(sim) + '/output/hdf5/snapdir_' + str(snap_num) + '/'           
            if snap_num == 600:
                halo_path = '/DFS-L/DATA/cosmo/grenache/aalazar/FIRE/GVB/' + str(sim) + '/halo/rockstar_dm/hdf5/'
                halo = galaxy_tools.load_halo(halo_path, snap_num, host=True, filetype='hdf5', hostnumber=1)
            elif snap_num == 184:
                halo_path = '/DFS-L/DATA/cosmo/grenache/aalazar/FIRE/GVB/' + str(sim) + '/halo/rockstar_dm/catalog/'
                halo = galaxy_tools.load_halo(halo_path, snap_num, host=True, filetype='ascii', hostnumber=1)
            else:
                print("Invalid snap_num")
                continue
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            # print(f"Loading halo took {elapsed_time:.2f} seconds")
            
            star_snapdict, gas_snapdict = galaxy_tools.load_sim(sim_path, snap_num)
            
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            # print(f"Loading simulation data and masking to halo took {elapsed_time:.2f} seconds")

            for i in rvir_scales:
                star_center, gas_center, halo2 = galaxy_tools.mask_sim_to_halo(
                    star_snapdict, gas_snapdict, halo, orient=False, lim=True, limvalue=halo['rvir'].values[0] * i)
                for j in particle_fractions:
                    while repeats <= k:
                        star_center_copy = copy.deepcopy(star_center)
                        result = iter_RIT_2d(star_center_copy, particle_fraction=j)
                        if result:
                            # Unpack the first dictionary from the result tuple
                            result_data = result
                            
                            # Extract data for each projection ('xy', 'yz', 'zx')
                            xy_data = {
                                'simulation': sim,
                                'a': result_data['xy']['a'],
                                'b': result_data['xy']['b'],
                                'b/a': result_data['xy']['b/a'],
                                'iterations': result_data['xy']['iterations'],
                                'initial_particles': result_data['xy']['initial_particles'],
                                'remaining_particles': result_data['xy']['remaining_particles'],
                                'max_radius': result_data['xy']['max_radius'],
                                'r_50': result_data['xy']['r_50'],
                                'mass': result_data['xy']['mass'],
                            }
                            
                            yz_data = {
                                'simulation': sim,
                                'a': result_data['yz']['a'],
                                'b': result_data['yz']['b'],
                                'b/a': result_data['yz']['b/a'],
                                'iterations': result_data['yz']['iterations'],
                                'initial_particles': result_data['yz']['initial_particles'],
                                'remaining_particles': result_data['yz']['remaining_particles'],
                                'max_radius': result_data['yz']['max_radius'],
                                'r_50': result_data['yz']['r_50'],
                                'mass': result_data['yz']['mass'],
                            }
                            
                            zx_data = {
                                'simulation': sim,
                                'a': result_data['zx']['a'],
                                'b': result_data['zx']['b'],
                                'b/a': result_data['zx']['b/a'],
                                'iterations': result_data['zx']['iterations'],
                                'initial_particles': result_data['zx']['initial_particles'],
                                'remaining_particles': result_data['zx']['remaining_particles'],
                                'max_radius': result_data['zx']['max_radius'],
                                'r_50': result_data['zx']['r_50'],
                                'mass': result_data['zx']['mass']
                            }
                            
                            results.append(xy_data)
                            results.append(yz_data)
                            results.append(zx_data)
                            
                            repeats += 1
        
        else:
            sim_path = '/DFS-L/DATA/cosmo/jgmoren1/FIREBox/FB15N1024/old_objects_1200/' + str(sim) + '.hdf5'
            star_snapdict, gas_snapdict = galaxy_tools.load_sim_FIREBox(sim_path)
            while repeats <= k:
                star_snapdict_copy = copy.deepcopy(star_snapdict)
                result = iter_RIT_2d(star_snapdict_copy, particle_fraction=[1.0])
                if result:
                    # Unpack the first dictionary from the result tuple
                    result_data = result
                    
                    # Extract data for each projection ('xy', 'yz', 'zx')
                    xy_data = {
                        'simulation': sim,
                        'a': result_data['xy']['a'],
                        'b': result_data['xy']['b'],
                        'b/a': result_data['xy']['b/a'],
                        'iterations': result_data['xy']['iterations'],
                        'initial_particles': result_data['xy']['initial_particles'],
                        'remaining_particles': result_data['xy']['remaining_particles'],
                        'max_radius': result_data['xy']['max_radius'],
                        'r_50': result_data['xy']['r_50'],
                        'mass': result_data['xy']['mass'],
                    }
                    
                    yz_data = {
                        'simulation': sim,
                        'a': result_data['yz']['a'],
                        'b': result_data['yz']['b'],
                        'b/a': result_data['yz']['b/a'],
                        'iterations': result_data['yz']['iterations'],
                        'initial_particles': result_data['yz']['initial_particles'],
                        'remaining_particles': result_data['yz']['remaining_particles'],
                        'max_radius': result_data['yz']['max_radius'],
                        'r_50': result_data['yz']['r_50'],
                        'mass': result_data['yz']['mass'],
                    }
                    
                    zx_data = {
                        'simulation': sim,
                        'a': result_data['zx']['a'],
                        'b': result_data['zx']['b'],
                        'b/a': result_data['zx']['b/a'],
                        'iterations': result_data['zx']['iterations'],
                        'initial_particles': result_data['zx']['initial_particles'],
                        'remaining_particles': result_data['zx']['remaining_particles'],
                        'max_radius': result_data['zx']['max_radius'],
                        'r_50': result_data['zx']['r_50'],
                        'mass': result_data['zx']['mass'],
                    }
                    
                    results.append(xy_data)
                    results.append(yz_data)
                    results.append(zx_data)
                    
                    repeats += 1
            print('measured ' + str(sim))
        repeats = k
        
    # save_to_csv(output_file, results)
    csv_2d('2D shapes', output_file, results) 
    return results

def partition_age(star_center_copy, age_partitions):
    """
    Partition star_center_copy into groups based on StellarAge and create new star_centers.

    Parameters
    ----------
    star_center_copy : dict
        Dictionary containing the star data, including 'StellarAge'.
    age_partitions : list of int
        List of age partitions to divide the stars.

    Returns
    -------
    partitioned_star : list of dict
        List of star_center dictionaries for each age partition.
    """
    age_partitions = [0] + age_partitions + [float('inf')]  # Ensure partitions cover all ages
    partitioned_star = []

    for i in range(len(age_partitions) - 1):
        lower_bound = age_partitions[i]
        upper_bound = age_partitions[i + 1]

        # Filter stars based on the age partition
        mask = (star_center_copy['StellarAge'] >= lower_bound) & (star_center_copy['StellarAge'] < upper_bound)

        new_star_center = {}
        for key, value in star_center_copy.items():
            if isinstance(value, (list, np.ndarray)):
                new_star_center[key] = np.array(value)[mask]
            else:
                new_star_center[key] = value  # Keep scalar values as is

        partitioned_star.append(new_star_center)

    return partitioned_star

    

def find_galaxy_shape3d(sim_inputs, output_file, snap_num_input=1200, host_num = 1, rvir_scales=[0.1], particle_fractions=[1.0], repeats=1, FIREBox=0, min_particles = 1000, age_partitions = None):
    """
    Analyze the shape of galaxies from simulation data and save the results to a CSV file.

    Parameters
    ----------
    sim_inputs : list of str
        A list of simulation identifiers.
    snap_num_input : int
        Snapshot number to load from the simulation.
    rvir_scale : float
        Scale factor to apply to the virial radius of the halo.
    output_file : str
        The name of the CSV file to save the results.

    Returns
    -------
    results : list of dict
        A list of dictionaries, each containing the shape parameters of a galaxy. Each dictionary includes:
        - 'simulation': str, Simulation identifier.
        - 'a, b, c': float, Lengths of the principal axes corresponding to the largest, 
            intermediate, and smallest eigenvalues.
        - 'b/a, c/a, c/b': float, Axis ratios of the principal axes.
        - 'ellipticity': float, Ellipticity of the galaxy shape.
        - 'triaxiality': float, Triaxiality of the galaxy shape.
        - 'iterations': int, Number of iterations in the shape finding algorithm.
        - 'remaining_particles': int, Number of particles remaining after masking.
        - 'max_radius': float, Maximum radius considered in the analysis.
        - 'remaining_galaxy' : dict, Dictionary containing the final galaxy coordinates and masses.
    """
    results = []
    k = repeats
    redshift = z(snap_num_input, '/DFS-L/DATA/cosmo/grenache/omyrtaj/fofie/snapshot_times.txt')
    for sim in sim_inputs:
        repeats = 1
        start_time = time.time()
        # load halo and snapdicts
        snap_num = snap_num_input
        if FIREBox == 0:
            sim_path = '/DFS-L/DATA/cosmo/grenache/aalazar/FIRE/GVB/' + str(sim) + '/output/hdf5/snapdir_' + str(snap_num) + '/'           
            if snap_num == 600:
                halo_path = '/DFS-L/DATA/cosmo/grenache/aalazar/FIRE/GVB/' + str(sim) + '/halo/rockstar_dm/hdf5/'
                halo = galaxy_tools.load_halo(halo_path, snap_num, host=True, filetype='hdf5', hostnumber=host_num)
            elif snap_num == 184:
                halo_path = '/DFS-L/DATA/cosmo/grenache/aalazar/FIRE/GVB/' + str(sim) + '/halo/rockstar_dm/catalog/'
                halo = galaxy_tools.load_halo(halo_path, snap_num, host=True, filetype='ascii', hostnumber=host_num)
            else:
                print("Invalid snap_num")
                continue
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            # print(f"Loading halo took {elapsed_time:.2f} seconds")
            
            star_snapdict, gas_snapdict = galaxy_tools.load_sim(sim_path, snap_num)
            
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            # print(f"Loading simulation data and masking to halo took {elapsed_time:.2f} seconds")

            for i in rvir_scales:
                star_center, gas_center, halo2 = galaxy_tools.mask_sim_to_halo(
                    star_snapdict, gas_snapdict, halo, orient=False, lim=True, limvalue=halo['rvir'].values[0] * i)
                for j in particle_fractions:
                    while repeats <= k:
                        star_center_copy = copy.deepcopy(star_center)
                        # print(f"The smallest value in 'StellarAge' is {min(star_center_copy['StellarAge'])} and the largest value is {max(star_center_copy['StellarAge'])}")
                        partitioned_star = partition_age(star_center_copy, age_partitions)
                        for partition in partitioned_star:
                            result = iter_RIT_3d(partition, particle_fraction=j, particle_bound = min_particles)
                            if result:
                                results.append({
                                    'simulation': sim,
                                    'a': result['a'],
                                    'b': result['b'],
                                    'c': result['c'],
                                    'b/a': result['b/a'],
                                    'c/a': result['c/a'],
                                    'c/b': result['c/b'],
                                    'ellipticity': result['ellipticity'],
                                    'triaxiality': result['triaxiality'],
                                    'iterations': result['iterations'],
                                    'initial_particles': result['initial_particles'],
                                    'remaining_particles': result['remaining_particles'],
                                    'max_radius': result['max_radius'],
                                    'r_50': result['r_50'],
                                    'mass': result['mass'],
                                    'cm': result['cm'],
                                    'age_range': result['age_range'],
                                    'snap_num': snap_num, 
                                    'redshift': redshift,
                                    'rotation_matrix': result['rotation_matrix'],
                                    'remaining_galaxy': result['remaining_galaxy']
                                })
                        repeats += 1
        
        else:
            sim_path = '/DFS-L/DATA/cosmo/jgmoren1/FIREBox/FB15N1024/old_objects_1200/' + str(sim) + '.hdf5'
            star_snapdict, gas_snapdict = galaxy_tools.load_sim_FIREBox(sim_path)  
            result = iter_RIT_3d(star_snapdict, particle_fraction=[1.0])
            if result:
                results.append({
                    'simulation': sim,
                    'a': result['a'],
                    'b': result['b'],
                    'c': result['c'],
                    'b/a': result['b/a'],
                    'c/a': result['c/a'],
                    'c/b': result['c/b'],
                    'ellipticity': result['ellipticity'],
                    'triaxiality': result['triaxiality'],
                    'iterations': result['iterations'],
                    'initial_particles': result['initial_particles'],
                    'remaining_particles': result['remaining_particles'],
                    'max_radius': result['max_radius'],
                    'r_50': result['r_50'],
                    'mass': result['mass'],
                    'cm': result['cm'],
                    'age_range': result['age_range'],
                    'snap_num': snap_num, 
                    'redshift': redshift,
                    'rotation_matrix' : result['rotation_matrix'],
                    'remaining_galaxy': result['remaining_galaxy']
                })
                repeats += 1
            print('measured ' + str(sim))

        repeats = k
            # rewrite this loop to sort rvir_scales from large to small and then just cut stuff
        # instead of loading it every time
    # save_to_csv(output_file, results)
    csv_3d('3D shapes', output_file, results) 
    return results

def outlier_shape3d(sim_inputs, output_file, cutoff_radius = 100, cm = [0, 0, 0], snap_num_input=1200, rvir_scales=[0.1], particle_fractions=[1.0], repeats=1, FIREBox=0):
    """
    Analyze the shape of galaxies from simulation data and save the results to a CSV file.

    Parameters
    ----------
    sim_inputs : list of str
        A list of simulation identifiers.
    snap_num_input : int
        Snapshot number to load from the simulation.
    rvir_scale : float
        Scale factor to apply to the virial radius of the halo.
    output_file : str
        The name of the CSV file to save the results.

    Returns
    -------
    results : list of dict
        A list of dictionaries, each containing the shape parameters of a galaxy. Each dictionary includes:
        - 'simulation': str, Simulation identifier.
        - 'a, b, c': float, Lengths of the principal axes corresponding to the largest, 
            intermediate, and smallest eigenvalues.
        - 'b/a, c/a, c/b': float, Axis ratios of the principal axes.
        - 'ellipticity': float, Ellipticity of the galaxy shape.
        - 'triaxiality': float, Triaxiality of the galaxy shape.
        - 'iterations': int, Number of iterations in the shape finding algorithm.
        - 'remaining_particles': int, Number of particles remaining after masking.
        - 'max_radius': float, Maximum radius considered in the analysis.
        - 'remaining_galaxy' : dict, Dictionary containing the final galaxy coordinates and masses.
    """
    results = []
    k = repeats
    if snap_num_input < 601:
        redshift = z(snap_num_input, '/DFS-L/DATA/cosmo/grenache/omyrtaj/fofie/snapshot_times.txt')
    else:
        redshift = 0.0
    for sim in sim_inputs:
        repeats = 1
        start_time = time.time()
        # load halo and snapdicts
        snap_num = snap_num_input
        if FIREBox == 0:
            sim_path = '/DFS-L/DATA/cosmo/grenache/aalazar/FIRE/GVB/' + str(sim) + '/output/hdf5/snapdir_' + str(snap_num) + '/'           
            if snap_num == 600:
                halo_path = '/DFS-L/DATA/cosmo/grenache/aalazar/FIRE/GVB/' + str(sim) + '/halo/rockstar_dm/hdf5/'
                halo = galaxy_tools.load_halo(halo_path, snap_num, host=True, filetype='hdf5', hostnumber=1)
            elif snap_num == 184:
                halo_path = '/DFS-L/DATA/cosmo/grenache/aalazar/FIRE/GVB/' + str(sim) + '/halo/rockstar_dm/catalog/'
                halo = galaxy_tools.load_halo(halo_path, snap_num, host=True, filetype='ascii', hostnumber=1)
            else:
                print("Invalid snap_num")
                continue
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            # print(f"Loading halo took {elapsed_time:.2f} seconds")
            
            star_snapdict, gas_snapdict = galaxy_tools.load_sim(sim_path, snap_num)
            
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            # print(f"Loading simulation data and masking to halo took {elapsed_time:.2f} seconds")

            for i in rvir_scales:
                star_center, gas_center, halo2 = galaxy_tools.mask_sim_to_halo(
                    star_snapdict, gas_snapdict, halo, orient=False, lim=True, limvalue=halo['rvir'].values[0] * i)
                for j in particle_fractions:
                    while repeats <= k:
                        star_center_copy = copy.deepcopy(star_center)
                        result = iter_RIT_3d(star_center_copy, particle_fraction=j)
                        if result:
                            results.append({
                                'simulation': sim,
                                'a': result['a'],
                                'b': result['b'],
                                'c': result['c'],
                                'b/a': result['b/a'],
                                'c/a': result['c/a'],
                                'c/b': result['c/b'],
                                'ellipticity': result['ellipticity'],
                                'triaxiality': result['triaxiality'],
                                'iterations': result['iterations'],
                                'initial_particles': result['initial_particles'],
                                'remaining_particles': result['remaining_particles'],
                                'max_radius': result['max_radius'],
                                'r_50': result['r_50'],
                                'mass': result['mass'],
                                'cm': result['cm'],
                                'age_range': result['age_range'],
                                'snap_num': snap_num, 
                                'redshift': redshift,
                                'rotation_matrix' : result['rotation_matrix'],
                                'remaining_galaxy': result['remaining_galaxy']
                            })
                            repeats += 1
        
        else:
            sim_path = '/DFS-L/DATA/cosmo/jgmoren1/FIREBox/FB15N1024/old_objects_1200/' + str(sim) + '.hdf5'
            star_snapdict, gas_snapdict = galaxy_tools.load_sim_FIREBox(sim_path)  
            coords = np.array(star_snapdict['Coordinates'])
            new_coords = coords - cm
            star_snapdict['Coordinates'] = new_coords.tolist()
            result = iter_RIT_3d(star_snapdict, initial_max_radius = cutoff_radius, particle_fraction=[1.0])
            # result = iter_RIT_3d(star_snapdict, particle_fraction=[1.0])
            sim = output_file.replace(".csv", "")
            if result:
                results.append({
                    'simulation': sim,
                    'a': result['a'],
                    'b': result['b'],
                    'c': result['c'],
                    'b/a': result['b/a'],
                    'c/a': result['c/a'],
                    'c/b': result['c/b'],
                    'ellipticity': result['ellipticity'],
                    'triaxiality': result['triaxiality'],
                    'iterations': result['iterations'],
                    'initial_particles': result['initial_particles'],
                    'remaining_particles': result['remaining_particles'],
                    'max_radius': result['max_radius'],
                    'r_50': result['r_50'],
                    'mass': result['mass'],
                    'cm': result['cm']+cm,
                    'age_range': result['age_range'],
                    'snap_num': snap_num, 
                    'redshift': redshift,
                    'rotation_matrix' : result['rotation_matrix'],
                    'remaining_galaxy': result['remaining_galaxy']
                })
                repeats += 1
            print('measured ' + str(sim))

        repeats = k
        
    # save_to_csv(output_file, results)
    csv_3d('outlier shapes', output_file, results) 
    return results        

import time

def measure_m12(sim_inputs, snap_num, host=1, age_range = 15):
    redshift = z(snap_num, '/DFS-L/DATA/cosmo/grenache/omyrtaj/fofie/snapshot_times.txt')
    for sim in sim_inputs:
        individual_result = []
        if host == 1: 
            individual_output_file = 'FIRE_' + str(sim) + '_' + str(age_range) + '_' + str(snap_num) + '.csv'
        else: 
            individual_output_file = 'FIRE_' + str(sim) + '_' + str(age_range) + '_' + str(snap_num) + '_2.csv'
        
        print(f"loading galaxy '{sim}'")
        start_time = time.time()
        
        halo_path = '/DFS-L/DATA/cosmo/grenache/omyrtaj/FIRE/' + str(sim) + '/halo/rockstar_dm/catalog_hdf5/'
        sim_path = '/DFS-L/DATA/cosmo/grenache/omyrtaj/FIRE/' + str(sim) + '/output/snapdir_' + str(snap_num) + '/'
        
        load_start_time = time.time()
        halo = galaxy_tools.load_halo(halo_path, snap_num, host=True, filetype='hdf5', hostnumber=host)
        star_snapdict, gas_snapdict = galaxy_tools.load_sim(sim_path, snap_num)
        load_end_time = time.time()

        # correct stellar age calculation
        star_snapdict['StellarAge'] = np.array(star_snapdict['StellarAge'])
        curr_time = Planck13.lookback_time(redshift).value
        star_snapdict['StellarAge'] -= curr_time

        # code to mask young stars, need to increase age range
        mask = star_snapdict['StellarAge'] <= age_range
        for key, value in star_snapdict.items():
            if isinstance(value, list):
                value = np.array(value)  # Convert lists to NumPy arrays
            if isinstance(value, np.ndarray) and len(value) == len(mask):
                star_snapdict[key] = value[mask]
            else:
                star_snapdict[key] = value
        if len(star_snapdict['StellarAge']) < 100:
            print('asdfasdfasdfasdf')
            
        star_center, gas_center, halo2 = galaxy_tools.mask_sim_to_halo(
                        star_snapdict, gas_snapdict, halo, orient=False, lim=True, limvalue=halo['rvir'].values[0] * 0.1)
        
        mask_end_time = time.time()
        
        result = iter_RIT_3d(star_center, initial_max_radius=50000000000000.0)
        iter_end_time = time.time()
        
        if result:
            individual_result.append({
                'simulation': sim,
                'a': result['a'],
                'b': result['b'],
                'c': result['c'],
                'b/a': result['b/a'],
                'c/a': result['c/a'],
                'c/b': result['c/b'],
                'ellipticity': result['ellipticity'],
                'triaxiality': result['triaxiality'],
                'iterations': result['iterations'],
                'initial_particles': result['initial_particles'],
                'remaining_particles': result['remaining_particles'],
                'max_radius': result['max_radius'],
                'r_50': result['r_50'],
                'mass': result['mass'],
                'cm': result['cm'],
                'snap_num': snap_num, 
                'redshift': redshift,
                'age_range': result['age_range'],
                'rotation_matrix': result['rotation_matrix'],
                'remaining_galaxy': result['remaining_galaxy']
            })
        
        print('measured galaxy ' + str(sim))
        
        csv_3d('m12 redshift shapes', individual_output_file, individual_result)
        write_end_time = time.time()
        
        print(f"Time taken for loading: {load_end_time - load_start_time:.2f} seconds")
        print(f"Time taken for masking: {mask_end_time - load_end_time:.2f} seconds")
        print(f"Total time taken for galaxy '{sim}': {write_end_time - start_time:.2f} seconds")

def z(snapshot, file_path):
    try:
        # Normalize the snapshot input by stripping leading zeros
        snapshot_str = str(snapshot).lstrip('0')
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
            for line in lines:
                if line.startswith("#") or not line.strip():
                    # Skip comments and empty lines
                    continue
                
                parts = line.split()
                if not parts[0].isdigit():
                    # Skip lines where the first part is not a digit
                    continue
                
                # Normalize the snapshot number in the file by stripping leading zeros
                snapshot_num_str = parts[0].lstrip('0')
                redshift = float(parts[2])
                
                if snapshot_num_str == snapshot_str:
                    return redshift

        raise ValueError(f"Snapshot {snapshot} not found in the file.")
    
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")
