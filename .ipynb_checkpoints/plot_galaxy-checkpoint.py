import numpy as np
import csv
import matplotlib

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython.display import set_matplotlib_formats

from mpl_toolkits.mplot3d import Axes3D

# set_matplotlib_formats('svg')
# baloga_3d: Inputs measured 3D galaxies and plots the axis ratios b/a, c/a, c/b vs the log10 of the largest axes of the projection a, a, b. 
# baloga_2d: Inputs 2D measured galaxy projections and plots the axis ratio vs. log10 of the largest axes (b/a vs. a)
# baca: Inputs measured 3D galaxies and plots them in b/a vs. c/a space. Note that it is essential to have FIRE or FIREBox in the file name in order for the function to plot them.  

# 2d scatter
# 2d hexbin
# 3d scatter
# 3d scatter with ellipsoid - must enter the three axes of the ellipsoid


def baloga_3d(csv_files):
    file_identifiers = [os.path.basename(csv_file).split('.')[0].replace('FIRE', '') for csv_file in csv_files]

    dfs = [pd.read_csv(csv_file) for csv_file in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    num_galaxies = len(df)

    required_columns = ['Max Radius', 'b/a', 'c/a', 'c/b']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in the dataframe: {', '.join(missing_columns)}")

    df['b/a * Max Radius'] = df['b/a'] * df['Max Radius']

    max_radius = np.tile(df['Max Radius'], 2).tolist() + df['b/a * Max Radius'].tolist()
    max_radius = np.log10(max_radius)

    combined_list = df['b/a'].tolist() + df['c/a'].tolist() + df['c/b'].tolist()

    csfont = {'size': '15'}
    csfont1 = { 'size': '13'}

    plt.figure(figsize=(8, 6))
    plt.hist2d(max_radius, combined_list, bins=20, range=[[0, 2], [0, 1]], cmap='inferno')
    plt.colorbar(label='Frequency')
    plt.xlabel('Max Radius', **csfont1)
    plt.ylabel('Axis Ratios', **csfont1)
    plt.title(f'FIRE & FIREBox 3D Axis Ratios vs. log10 of Max Radius (N={num_galaxies})', **csfont)
    # {", ".join(file_identifiers)}
    plt.show()

def baloga_2d(csv_files):
    file_identifiers = [os.path.basename(csv_file).split('.')[0].replace('FIRE', '') for csv_file in csv_files]

    dfs = [pd.read_csv(csv_file) for csv_file in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    num_galaxies = len(df)

    required_columns = ['Max Radius', 'b/a']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in the dataframe: {', '.join(missing_columns)}")

    df['b/a * Max Radius'] = df['b/a'] * df['Max Radius']
    max_radius = df['b/a * Max Radius'].tolist()
    max_radius = np.log10(max_radius)

    combined_list = df['b/a'].tolist()

    csfont = {'size': '15'}
    csfont1 = { 'size': '13'}

    plt.figure(figsize=(8, 6))
    plt.hist2d(max_radius, combined_list, bins=20, range=[[-1, 2], [0, 1]], cmap='inferno')
    plt.colorbar(label='Frequency')
    plt.xlabel('Max Radius', **csfont1)
    plt.ylabel('Axis Ratios', **csfont1)
    plt.title(f'FIRE {", ".join(file_identifiers)} 2D Axis Ratios vs. log10 of Max Radius (N={num_galaxies})', **csfont)

    plt.show()


def baca(csv_files):
    # Read CSV files and add a 'source' column based on the filename
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['source'] = os.path.basename(csv_file)
        dfs.append(df)
    
    df_all = pd.concat(dfs, ignore_index=True)
    num_galaxies = len(df_all)
    
    required_columns = ['b/a', 'c/a', 'mass']
    missing_columns = [col for col in required_columns if col not in df_all.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in the dataframe: {', '.join(missing_columns)}")
    
    df_firebox = df_all[df_all['source'].str.contains('FIREBox', na=False)]
    df_fire = df_all[df_all['source'].str.contains('FIRE', na=False) & ~df_all['source'].str.contains('FIREBox', na=False)]
    df_other = df_all[~df_all['source'].str.contains('FIRE', na=False)]
    
    ba_firebox = df_firebox['b/a']
    ca_firebox = df_firebox['c/a']
    mass_firebox = df_firebox['mass']
    log_mass_firebox = np.log10(mass_firebox)
    
    ba_fire = df_fire['b/a']
    ca_fire = df_fire['c/a']
    mass_fire = df_fire['mass']
    log_mass_fire = np.log10(mass_fire)
    
    ba_other = df_other['b/a']
    ca_other = df_other['c/a']
    mass_other = df_other['mass']
    log_mass_other = np.log10(mass_other)
    
    csfont = {'size': '15'}
    csfont1 = {'size': '13'}
    csfont2 = {'size': '15'}
    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_facecolor('black')
    
    scatter1 = plt.scatter(ba_firebox, ca_firebox, c=log_mass_firebox, alpha=0.8, s=15, cmap='inferno', marker='.', zorder=6, label='FIREBox')
    scatter2 = plt.scatter(ba_fire, ca_fire, c=log_mass_fire, alpha=1, s=55, cmap='inferno', marker='^', zorder=6, label='FIRE')
    scatter3 = plt.scatter(ba_other, ca_other, c=log_mass_other, alpha=1, s=100, cmap='inferno', marker='o', zorder=7, label='Other')
    
    vmin = df_all['mass'].apply(np.log10).min()
    vmax = df_all['mass'].apply(np.log10).max()
    scatter1.set_clim(vmin=vmin, vmax=vmax)
    scatter2.set_clim(vmin=vmin, vmax=vmax)
    scatter3.set_clim(vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar(scatter1)
    cbar.set_label('log10(Stellar Mass)', **csfont1)
    
    plt.legend(loc='best')
    
    plt.xlabel('b/a', color='black', **csfont1)
    plt.ylabel('c/a', color='black', **csfont1)
    
    # Determine the appropriate title based on the input files
    title = "Galaxies in b/a vs. c/a space from "
    if any("FIREBox" in os.path.basename(f) for f in csv_files):
        title += "FIREBox"
    elif any("FIRE" in os.path.basename(f) for f in csv_files):
        title += "FIRE"
    else:
        title += "FIRE"
    title += f" (N={num_galaxies})"
    
    plt.title(title, color='black', **csfont)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    x = np.arange(0.0, 1.01, 0.01)
    plt.plot(x, x, color='white', linestyle='-', zorder=5)
    x_filtered = x[x >= 0.5]
    y_filtered = 1 - x_filtered
    plt.plot(x_filtered, y_filtered, color='white', linestyle='-', zorder=5)
    plt.plot([1, 1], [0, 1], color='white', linestyle='-', zorder=5)
    
    plt.fill_between(x, x, 1 - x, where=(x >= 0.5), color='black', zorder=1)
    plt.fill_between(x, x, where=(x <= 0.5), color='black', zorder=1)
    plt.fill_between(x, 1 - x, where=(x >= 0.5), color='black', zorder=1)
    
    center = (1, 0)
    radius = 0.4
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta), color='white', linestyle='-', zorder=5)
    plt.fill_between(center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta), color='black', zorder=1)
    
    plt.text(0.7, 0.11, 'Disky', **csfont2, alpha=0.8, zorder=7, color='white')
    plt.text(0.3, 0.11, 'Elongated', **csfont2, alpha=0.8, zorder=7, color='white')
    plt.text(0.76, 0.72, 'Spheroidal', **csfont2, alpha=0.8, zorder=7, color='white')
    
    ax.tick_params(colors='black', which='both')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')

    plt.show()


def scatter_2d(galaxy, projection='xy', lim=None):
    """
    Create a 2D scatter plot of the final coordinates of the galaxy,
    with the specified projection and optional axis limits, with a black
    background limited to the plot area.

    Parameters
    ----------
    galaxy : dict
        Dictionary containing the final galaxy coordinates and masses.
    projection : str, optional
        The projection to plot ('xy', 'zx', 'yz').
    lim : float, optional
        The limit for the plot axes (if None, limits are set based on data).

    Returns
    -------
    None
        Displays the plot.
    """
    coords = galaxy['Coordinates']
    
    if projection == 'xy':
        x = coords[:, 0]
        y = coords[:, 1]
        xlabel, ylabel = 'x (kpc)', 'y (kpc)'
    elif projection == 'zx':  # switched from 'xz' to 'zx'
        x = coords[:, 2]
        y = coords[:, 0]
        xlabel, ylabel = 'z (kpc)', 'x (kpc)'
    elif projection == 'yz':
        x = coords[:, 1]
        y = coords[:, 2]
        xlabel, ylabel = 'y (kpc)', 'z (kpc)'
    else:
        raise ValueError("Invalid projection. Choose from 'xy', 'zx', or 'yz'.")
    
    # Compute the limit based on the largest absolute coordinate value if lim is not provided
    if lim is None:
        lim = np.max(np.abs(coords))
    
    # Create a scatter plot with a black background limited to the plot area
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=1, alpha=0.5, color='c')  # Adjust the marker size 's' and color as needed

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.set_xlabel(xlabel, color='black')
    ax.set_ylabel(ylabel, color='black')
    ax.set_title(f'2D Scatter Plot of Galaxy ({projection.upper()} Projection)', color='black')
    
    ax.patch.set_facecolor('black')  # Set the background color of the plot area

    plt.grid(False)  # Turn off grid lines
    
    plt.show()


def hexbin_2d(galaxy, projection='xy', gridsize=50, cmap='inferno'):
    """
    Create a 2D hexbin plot of the final coordinates of the galaxy,
    with the specified projection.

    Parameters
    ----------
    galaxy : dict
        Dictionary containing the final galaxy coordinates and masses.
    projection : str, optional
        The projection to plot ('xy', 'zx', 'yz').
    gridsize : int, optional
        The number of hexagons in the x-direction.
    cmap : str, optional
        The colormap to use for the plot.

    Returns
    -------
    None
        Displays the plot.
    """
    coords = galaxy['Coordinates']
    
    if projection == 'xy':
        x = coords[:, 0]
        y = coords[:, 1]
        xlabel, ylabel = 'x (kpc)', 'y (kpc)'
    elif projection == 'zx':  # switched from 'xz' to 'zx'
        x = coords[:, 2]
        y = coords[:, 0]
        xlabel, ylabel = 'z (kpc)', 'x (kpc)'
    elif projection == 'yz':
        x = coords[:, 1]
        y = coords[:, 2]
        xlabel, ylabel = 'y (kpc)', 'z (kpc)'
    else:
        raise ValueError("Invalid projection. Choose from 'xy', 'zx', or 'yz'.")
    
    # Create a hexbin plot
    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(x, y, gridsize=gridsize, cmap=cmap)
    cb = plt.colorbar(hb)
    cb.set_label('Counts')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'2D Hexbin Plot of Galaxy ({projection.upper()} Projection)')
    
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def scatter_3d(galaxy, axis_lengths=(1, 1, 1), lim=None):
    """
    Create a 3D scatter plot of the final coordinates of the galaxy,
    and plot an ellipsoid shell with specified axis lengths, with a black
    background limited to the plot area.

    Parameters
    ----------
    galaxy : dict
        Dictionary containing the final galaxy coordinates and masses.
    axis_lengths : tuple of floats, optional
        The lengths of the axes of the ellipsoid (a, b, c).
    lim : float, optional
        The limit for the plot axes (if None, limits are set based on data).

    Returns
    -------
    None
        Displays the plot.
    """
    coords = galaxy['Coordinates']
    
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    # Compute the limit based on the largest absolute coordinate value if lim is not provided
    if lim is None:
        lim = np.max(np.abs(coords))
    
    # Create a 3D scatter plot with a black background limited to the plot area
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.dist = 13
    ax.scatter(x, y, z, s=1, alpha=0.5, color='cyan')  # Adjust the marker size 's' and color as needed
    ax.set_xlabel('x (kpc)', color='white')
    ax.set_ylabel('y (kpc)', color='white')
    ax.set_zlabel('z (kpc)', color='white')
    
    ax.patch.set_facecolor('black')  # Set the background color of the plot area
    
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))  # Black background for the grid
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    
    # Customize tick parameters
    ax.xaxis.set_tick_params(colors='white')
    ax.yaxis.set_tick_params(colors='white')
    ax.zaxis.set_tick_params(colors='white')

    # Customize grid lines color and style
    ax.xaxis._axinfo['grid'].update(color='white', linestyle='-')
    ax.yaxis._axinfo['grid'].update(color='white', linestyle='-')
    ax.zaxis._axinfo['grid'].update(color='white', linestyle='-')


    # Customize grid transparency
    ax.xaxis.pane.set_alpha(1)
    ax.yaxis.pane.set_alpha(1)
    ax.zaxis.pane.set_alpha(1)
    ax.set_title('3D Scatter Plot of Galaxy', color='black')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    plt.grid(False)  # Turn off default grid lines
    
    plt.show()

def scatter_ellipsoid_3d(galaxy, axis_lengths=(1, 1, 1), lim=None):
    """
    Create a 3D scatter plot of the final coordinates of the galaxy,
    and plot an ellipsoid shell with specified axis lengths, with a black
    background and white grid lines.

    Parameters
    ----------
    galaxy : dict
        Dictionary containing the final galaxy coordinates and masses.
    axis_lengths : tuple of floats, optional
        The lengths of the axes of the ellipsoid (a, b, c).
    lim : float, optional
        The limit for the plot axes (if None, limits are set based on data).

    Returns
    -------
    None
        Displays the plot.
    """
    coords = galaxy['Coordinates']
    
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    # Create a 3D scatter plot with a black background limited to the plot area
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1, alpha=0.5, marker='o', color='cyan')  # Adjust marker size 's' and color as needed
    
    # Create ellipsoid
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    a, b, c = axis_lengths
    x_ellipsoid = a * np.outer(np.cos(u), np.sin(v))
    y_ellipsoid = b * np.outer(np.sin(u), np.sin(v))
    z_ellipsoid = c * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_wireframe(x_ellipsoid, y_ellipsoid, z_ellipsoid, color='red', alpha=0.25)

    # Compute the limit based on the largest absolute coordinate value if lim is not provided
    if lim is None:
        lim = np.max(np.abs(coords))
    ax.dist = 13
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    ax.set_xlabel('x (kpc)', color='white')
    ax.set_ylabel('y (kpc)', color='white')
    ax.set_zlabel('z (kpc)', color='white')
    

    ax.patch.set_facecolor('black')  # Set the background color of the plot area

    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))  # Black background for the grid
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    
    # Customize tick parameters
    ax.xaxis.set_tick_params(colors='white')
    ax.yaxis.set_tick_params(colors='white')
    ax.zaxis.set_tick_params(colors='white')

    # Customize grid lines color and style
    ax.xaxis._axinfo['grid'].update(color='white', linestyle='-')
    ax.yaxis._axinfo['grid'].update(color='white', linestyle='-')
    ax.zaxis._axinfo['grid'].update(color='white', linestyle='-')
    ax.set_title('3D Scatter Plot of Galaxy with Measured Ellipsoid', color='black')

    plt.grid(False)  # Turn off grid lines
    
    plt.show()
