import numpy as np
import csv
import matplotlib

import matplotlib.cm as cm
from astropy.cosmology import Planck13
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

import matplotlib.cm as cm
from scipy.integrate import quad

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
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'truncated_cmap', cmap(np.linspace(minval, maxval, n)))
    return new_cmap



def ca_a_scatter(csv_files, shift = None, title=None):
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['source'] = os.path.basename(csv_file)
        dfs.append(df)
    
    df_all = pd.concat(dfs, ignore_index=True)
    num_galaxies = len(df_all)
    
    required_columns = ['Max Radius', 'c/a', 'Age Range', 'Redshift']
    missing_columns = [col for col in required_columns if col not in df_all.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in the dataframe: {', '.join(missing_columns)}")
    df_all['Lookback_Time'] = df_all['Redshift'].apply(lookback_time)
    df_all['c/a * Max Radius'] = df_all['c/a'] * df_all['Max Radius']
    df_all['Max Radius log10'] = df_all['Max Radius']
    df_all['Age Range Avg'] = df_all['Age Range'].apply(lambda x: np.mean(eval(x)))
    if shift == True:
        df_all['Age Range Avg'] += df_all['Lookback_Time']
    csfont = {'size': '13'}
    csfont1 = {'size': '12'}
    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_facecolor('black')

    original_cmap = plt.get_cmap('inferno')
    colormap = truncate_colormap(original_cmap, minval=0.35, maxval=1.0)  # Adjust minval as needed

    min_age = df_all['Age Range Avg'].min()
    max_age = df_all['Age Range Avg'].max()
    normalize = plt.Normalize(vmin=min_age, vmax=max_age)
    
    scatter = plt.scatter(df_all['Max Radius log10'], df_all['c/a'], c=df_all['Age Range Avg'], cmap=colormap, norm=normalize, alpha=1.0, s=50)
    
    cbar = plt.colorbar(scatter)
    if shift == True:
         cbar.set_label('Lookback Time (Gyr)', **csfont1)
    else:
        cbar.set_label('Stellar Age (Gyr)', **csfont1)
    plt.xlabel('Major axis a (kpc)', **csfont1)
    plt.ylabel('Axis Ratios (c/a)', **csfont1)
    
    # Set x-axis and y-axis limits
    plt.xlim(left=0)
    plt.ylim(0, 1)
    
    if title is None:
        file_identifiers = [os.path.basename(csv_file).split('.')[0].replace('FIRE', '') for csv_file in csv_files]
        plt.title(f'FIRE {", ".join(file_identifiers)} 2D Axis Ratios vs. Max Radius (N={num_galaxies})', **csfont)
    else:
        plt.title(title, **csfont)
    
    plt.show()



def baca_age1(ax, csv_files, title=None, colorbar = True, ba = True, ca = True):
    """
    Plot galaxies' b/a vs. c/a space colored by average stellar age.
    
    Parameters:
    ax (matplotlib.axes.Axes): The axes object to plot on.
    csv_files (list of str): List of paths to the CSV files.
    title (str, optional): Title for the plot.
    
    Returns:
    None
    """
    # Read CSV files and add a 'source' column based on the filename
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['source'] = os.path.basename(csv_file)
        dfs.append(df)
    
    df_all = pd.concat(dfs, ignore_index=True)
    num_galaxies = len(df_all)
    
    required_columns = ['b/a', 'c/a', 'mass', 'Age Range']
    missing_columns = [col for col in required_columns if col not in df_all.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in the dataframe: {', '.join(missing_columns)}")
    
    # Calculate the average of each Age Range
    df_all['Age Range Avg'] = df_all['Age Range'].apply(lambda x: np.mean(eval(x)))
    
    # Sort by the Age Range Avg
    df_all = df_all.sort_values(by='Age Range Avg').reset_index(drop=True)
    # Categorize datasets based on filename patterns
    df_firebox = df_all[df_all['source'].str.contains('FIREBox', na=False)]
    df_firem10 = df_all[df_all['source'].str.contains('FIREm', na=False)]
    df_fire_m12 = df_all[df_all['source'].str.contains('FIRE_m12', na=False)]
    
    ba_firebox = df_firebox['b/a']
    ca_firebox = df_firebox['c/a']
    mass_firebox = df_firebox['mass']
    
    ba_firem10 = df_firem10['b/a']
    ca_firem10 = df_firem10['c/a']
    mass_firem10 = df_firem10['mass']
    
    ba_fire_m12 = df_fire_m12['b/a']
    ca_fire_m12 = df_fire_m12['c/a']
    mass_fire_m12 = df_fire_m12['mass']
    
    csfont = {'size': '15'}
    csfont1 = {'size': '13'}
    csfont2 = {'size': '15'}
    
    # Set the background color for the axes
    ax.set_facecolor('black')
    
    # Create a colormap that transitions from blue to red based on Age Range Avg
    original_cmap = plt.get_cmap('inferno')
    colormap = truncate_colormap(original_cmap, minval=0.35, maxval=1.0)  # Adjust minval as needed
    min_age = df_all['Age Range Avg'].min()
    max_age = df_all['Age Range Avg'].max()
    normalize = plt.Normalize(vmin=min_age, vmax=max_age)
    
    scatter1 = ax.scatter(ba_firebox, ca_firebox, c=df_firebox['Age Range Avg'], alpha=0.8, s=5, cmap=colormap, norm=normalize, marker='.', zorder=6, label='FIREBox')
    scatter2 = ax.scatter(ba_firem10, ca_firem10, c=df_firem10['Age Range Avg'], alpha=1, s=60, cmap=colormap, norm=normalize, marker='^', zorder=6, label='FIRE')
    scatter3 = ax.scatter(ba_fire_m12, ca_fire_m12, c=df_fire_m12['Age Range Avg'], alpha=1, s=60, cmap=colormap, norm=normalize, marker='^', zorder=7, label='FIRE_m12')
    if colorbar == True:
        cbar = plt.colorbar(scatter1, ax=ax)
        cbar.set_label('Stellar Age (Gyr)', **csfont1)
    
    # ax.legend(loc='best')
    if ba == True:
        ax.set_xlabel('b/a', color='black', **csfont1)
    if ca == True:
        ax.set_ylabel('c/a', color='black', **csfont1)
    
    # Set the title directly if provided, otherwise determine based on input files
    if title is None:
        title = "Galaxies in b/a vs. c/a space from "
        if any("FIREBox_satellite" in os.path.basename(f) for f in csv_files):
            title += "FIREBox"
        elif any("FIREm10" in os.path.basename(f) for f in csv_files):
            title += "FIRE"
        elif any("FIRE_m12" in os.path.basename(f) for f in csv_files):
            title += "FIRE_m12"
        title += f" (N={num_galaxies})"
    
    ax.set_title(title, color='black', **csfont)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    x = np.arange(0.0, 1.01, 0.01)
    ax.plot(x, x, color='white', linestyle='-', zorder=5)
    x_filtered = x[x >= 0.5]
    y_filtered = 1 - x_filtered
    ax.plot(x_filtered, y_filtered, color='white', linestyle='-', zorder=5)
    ax.plot([1, 1], [0, 1], color='white', linestyle='-', zorder=5)
    
    ax.fill_between(x, x, 1 - x, where=(x >= 0.5), color='black', zorder=1)
    ax.fill_between(x, x, where=(x <= 0.5), color='black', zorder=1)
    ax.fill_between(x, 1 - x, where=(x >= 0.5), color='black', zorder=1)
    
    center = (1, 0)
    radius = 0.4
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta), color='white', linestyle='-', zorder=5)
    ax.fill_between(center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta), color='black', zorder=1)
    
    ax.text(0.7, 0.11, 'Disky', **csfont2, alpha=0.8, zorder=7, color='white')
    ax.text(0.3, 0.11, 'Elongated', **csfont2, alpha=0.8, zorder=7, color='white')
    ax.text(0.76, 0.72, 'Spheroidal', **csfont2, alpha=0.8, zorder=7, color='white')
    
    ax.tick_params(colors='black', which='both')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')

def baca_lookback1(ax, csv_files, title=None, colorbar = True, ba = True, ca = True):
    def lookback_time(z):
        return Planck13.lookback_time(z).value

    # Read CSV files and add a 'source' column based on the filename
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['source'] = os.path.basename(csv_file)
        dfs.append(df)
    
    df_all = pd.concat(dfs, ignore_index=True)
    num_galaxies = len(df_all)
    
    required_columns = ['b/a', 'c/a', 'mass', 'Redshift']
    missing_columns = [col for col in required_columns if col not in df_all.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in the dataframe: {', '.join(missing_columns)}")
    
    # Calculate lookback time for each galaxy
    df_all['Lookback_Time'] = df_all['Redshift'].apply(lookback_time)
    df_all = df_all.sort_values(by='Lookback_Time').reset_index(drop=True)
    
    df_firebox = df_all[df_all['source'].str.contains('FIREBox', na=False)]
    df_firem10 = df_all[df_all['source'].str.contains('FIREm', na=False)]
    df_fire_m12 = df_all[df_all['source'].str.contains('FIRE_m12', na=False)]
    
    ba_firebox = df_firebox['b/a']
    ca_firebox = df_firebox['c/a']
    mass_firebox = df_firebox['mass']
    log_mass_firebox = np.log10(mass_firebox)
    
    ba_firem10 = df_firem10['b/a']
    ca_firem10 = df_firem10['c/a']
    mass_firem10 = df_firem10['mass']
    log_mass_firem10 = np.log10(mass_firem10)
    
    ba_fire_m12 = df_fire_m12['b/a']
    ca_fire_m12 = df_fire_m12['c/a']
    mass_fire_m12 = df_fire_m12['mass']
    log_mass_fire_m12 = np.log10(mass_fire_m12)
    
    # Create a colormap that transitions from blue to red based on lookback time
    colormap = cm.get_cmap('inferno')
    original_cmap = plt.get_cmap('inferno')
    colormap = truncate_colormap(original_cmap, minval=0.35, maxval=1.0)  # Adjust minval as needed
    min_lookback_time = df_all['Lookback_Time'].min()
    max_lookback_time = df_all['Lookback_Time'].max()
    normalize = plt.Normalize(vmin=min_lookback_time, vmax=max_lookback_time)
    
    ax.set_facecolor('black')
    scatter1 = ax.scatter(ba_firebox, ca_firebox, c=df_firebox['Lookback_Time'], alpha=0.8, s=5, cmap=colormap, norm=normalize, marker='.', zorder=6, label='FIREBox')
    scatter2 = ax.scatter(ba_firem10, ca_firem10, c=df_firem10['Lookback_Time'], alpha=1, s=60, cmap=colormap, norm=normalize, marker='^', zorder=6, label='FIRE')
    scatter3 = ax.scatter(ba_fire_m12, ca_fire_m12, c=df_fire_m12['Lookback_Time'], alpha=1, s=60, cmap=colormap, norm=normalize, marker='o', zorder=7, label='FIRE_m12')
    if colorbar == True:
        cbar = plt.colorbar(scatter1, ax=ax)
        cbar.set_label('Lookback Time (Gyr)', fontsize=13)
    
    # ax.legend(loc='best')
    if ba == True: 
        ax.set_xlabel('b/a', fontsize=13)
    if ca == True:
        ax.set_ylabel('c/a', fontsize=13)
    
    if title is None:
        title = "Galaxies in b/a vs. c/a space from "
        if any("FIREBox_satellite" in os.path.basename(f) for f in csv_files):
            title += "FIREBox"
        elif any("FIREm10" in os.path.basename(f) for f in csv_files):
            title += "FIRE"
        elif any("FIRE_m12" in os.path.basename(f) for f in csv_files):
            title += "FIRE_m12"
        title += f" (N={num_galaxies})"
    
    ax.set_title(title, fontsize=15)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Draw arrows connecting points ordered by lookback time
    for i in range(len(df_all) - 1):
        ax.arrow(df_all['b/a'][i], df_all['c/a'][i], 
                 df_all['b/a'][i+1] - df_all['b/a'][i], 
                 df_all['c/a'][i+1] - df_all['c/a'][i],
                 color='white', alpha=0.8, head_width=0.02, head_length=0.02, zorder = 6)
    
    x = np.arange(0.0, 1.01, 0.01)
    ax.plot(x, x, color='white', linestyle='-', zorder=5)
    x_filtered = x[x >= 0.5]
    y_filtered = 1 - x_filtered
    ax.plot(x_filtered, y_filtered, color='white', linestyle='-', zorder=5)
    ax.plot([1, 1], [0, 1], color='white', linestyle='-', zorder=5)
    
    ax.fill_between(x, x, 1 - x, where=(x >= 0.5), color='black', zorder=1)
    ax.fill_between(x, x, where=(x <= 0.5), color='black', zorder=1)
    ax.fill_between(x, 1 - x, where=(x >= 0.5), color='black', zorder=1)
    
    center = (1, 0)
    radius = 0.4
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta), color='white', linestyle='-', zorder=5)
    ax.fill_between(center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta), color='black', zorder=1)
    
    ax.text(0.7, 0.11, 'Disky', fontsize=15, alpha=0.8, zorder=7, color='white')
    ax.text(0.3, 0.11, 'Elongated', fontsize=15, alpha=0.8, zorder=7, color='white')
    ax.text(0.76, 0.72, 'Spheroidal', fontsize=15, alpha=0.8, zorder=7, color='white')
    
    ax.tick_params(colors='black', which='both')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')


# Example of usage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def baca_combine(csv_files1, csv_files2, csv_files3, title1=None, title2=None, title3=None):
    # Create a gridspec with different width ratios
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1.2, 1.2])  # Adjust the width ratio of the third subplot

    fig = plt.figure(figsize=(22, 6))

    # Create subplots with the specified gridspec
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Plot each graph
    baca_lookback1(ax1, csv_files1, title=title1, colorbar=True, ba=True, ca=True)
    baca_lookback1(ax2, csv_files2, title=title2, colorbar=True, ba=True, ca=True)
    baca_age1(ax3, csv_files3, title=title3, colorbar=True, ba=True, ca=True)

    plt.tight_layout()
    plt.show()


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

def baloga_2d(csv_files, title):
    file_identifiers = [os.path.basename(csv_file).split('.')[0].replace('FIRE', '') for csv_file in csv_files]

    dfs = [pd.read_csv(csv_file) for csv_file in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    num_galaxies = len(df)

    required_columns = ['Max Radius', 'c/a']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in the dataframe: {', '.join(missing_columns)}")

    df['c/a * Max Radius'] = df['c/a'] * df['Max Radius']
    max_radius = df['c/a * Max Radius'].tolist()
    max_radius = np.log10(max_radius)

    combined_list = df['c/a'].tolist()

    csfont = {'size': '15'}
    csfont1 = { 'size': '13'}

    plt.figure(figsize=(8, 6))
    plt.hist2d(max_radius, combined_list, bins=20, range=[[-1, 2], [0, 1]], cmap='inferno')
    plt.colorbar(label='Frequency')
    plt.xlabel('Max Radius', **csfont1)
    plt.ylabel('Axis Ratios', **csfont1)
    if title is None:
        plt.title(f'FIRE {", ".join(file_identifiers)} 2D Axis Ratios vs. log10 of Max Radius (N={num_galaxies})', **csfont)
    else: plt.title(title)

    plt.show()


def baca2(csv_files):
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
    
    # Categorize datasets based on filename patterns
    df_firebox = df_all[df_all['source'].str.contains('FIREBox', na=False)]
    df_firem10 = df_all[df_all['source'].str.contains('FIREm', na=False)]
    df_fire_m12 = df_all[df_all['source'].str.contains('FIRE_m12', na=False)]
    
    ba_firebox = df_firebox['b/a']
    ca_firebox = df_firebox['c/a']
    mass_firebox = df_firebox['mass']
    log_mass_firebox = np.log10(mass_firebox)
    
    ba_firem10 = df_firem10['b/a']
    ca_firem10 = df_firem10['c/a']
    mass_firem10 = df_firem10['mass']
    log_mass_firem10 = np.log10(mass_firem10)
    
    ba_fire_m12 = df_fire_m12['b/a']
    ca_fire_m12 = df_fire_m12['c/a']
    mass_fire_m12 = df_fire_m12['mass']
    log_mass_fire_m12 = np.log10(mass_fire_m12)
    
    csfont = {'size': '15'}
    csfont1 = {'size': '13'}
    csfont2 = {'size': '15'}
    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_facecolor('black')
    
    scatter1 = plt.scatter(ba_firebox, ca_firebox, c=log_mass_firebox, alpha=0.8, s=5, cmap='inferno', marker='.', zorder=6, label='FIREBox')
    scatter2 = plt.scatter(ba_firem10, ca_firem10, c=log_mass_firem10, alpha=1, s=60, cmap='inferno', marker='^', zorder=6, label='FIRE')
    scatter3 = plt.scatter(ba_fire_m12, ca_fire_m12, c=log_mass_fire_m12, alpha=1, s=60, cmap='inferno', marker='o', zorder=7, label='FIRE_m12')
    
    vmin = df_all['mass'].apply(np.log10).min()-1
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
    if any("FIREBox_satellite" in os.path.basename(f) for f in csv_files):
        title += "FIREBox"
    elif any(("FIREm10" in os.path.basename(f) for f in csv_files) or ("FIREm10" in os.path.basename(f) for f in csv_files)):
        title += "FIRE"
    elif any("FIRE_m12" in os.path.basename(f) for f in csv_files):
        title += "FIRE_m12"
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

def baca_age(csv_files, title=None):
    # Read CSV files and add a 'source' column based on the filename
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['source'] = os.path.basename(csv_file)
        dfs.append(df)
    
    df_all = pd.concat(dfs, ignore_index=True)
    num_galaxies = len(df_all)
    
    required_columns = ['b/a', 'c/a', 'mass', 'Age Range']
    missing_columns = [col for col in required_columns if col not in df_all.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in the dataframe: {', '.join(missing_columns)}")
    
    # Calculate the average of each Age Range
    df_all['Age Range Avg'] = df_all['Age Range'].apply(lambda x: np.mean(eval(x)))
    
    # Sort by the Age Range Avg
    df_all = df_all.sort_values(by='Age Range Avg').reset_index(drop=True)
    
    # Categorize datasets based on filename patterns
    df_firebox = df_all[df_all['source'].str.contains('FIREBox', na=False)]
    df_firem10 = df_all[df_all['source'].str.contains('FIREm', na=False)]
    df_fire_m12 = df_all[df_all['source'].str.contains('FIRE_m12', na=False)]
    
    ba_firebox = df_firebox['b/a']
    ca_firebox = df_firebox['c/a']
    mass_firebox = df_firebox['mass']
    log_mass_firebox = np.log10(mass_firebox)
    
    ba_firem10 = df_firem10['b/a']
    ca_firem10 = df_firem10['c/a']
    mass_firem10 = df_firem10['mass']
    log_mass_firem10 = np.log10(mass_firem10)
    
    ba_fire_m12 = df_fire_m12['b/a']
    ca_fire_m12 = df_fire_m12['c/a']
    mass_fire_m12 = df_fire_m12['mass']
    log_mass_fire_m12 = np.log10(mass_fire_m12)
    
    csfont = {'size': '15'}
    csfont1 = {'size': '13'}
    csfont2 = {'size': '15'}
    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_facecolor('black')
    
    # Create a colormap that transitions from blue to red based on the sorted row index
    colormap = cm.get_cmap('inferno')
    min_age = df_all['Age Range Avg'].min()
    max_age = df_all['Age Range Avg'].max()
    normalize = plt.Normalize(vmin=min_age, vmax=max_age)
    
    scatter1 = plt.scatter(ba_firebox, ca_firebox, c=df_firebox['Age Range Avg'], alpha=0.8, s=5, cmap='inferno', norm=normalize, marker='.', zorder=6, label='FIREBox')
    scatter2 = plt.scatter(ba_firem10, ca_firem10, c=df_firem10['Age Range Avg'], alpha=1, s=60, cmap='inferno', norm=normalize, marker='^', zorder=6, label='FIRE')
    scatter3 = plt.scatter(ba_fire_m12, ca_fire_m12, c=df_fire_m12['Age Range Avg'], alpha=1, s=60, cmap='inferno', norm=normalize, marker='o', zorder=7, label='FIRE_m12')
    
    cbar = plt.colorbar(scatter1)
    cbar.set_label('Stellar Age (Gyr)', **csfont1)
    
    plt.legend(loc='best')
    
    plt.xlabel('b/a', color='black', **csfont1)
    plt.ylabel('c/a', color='black', **csfont1)
    
    # Determine the appropriate title based on the input files
    if title is None:
        title = "Galaxies in b/a vs. c/a space from "
        if any("FIREBox_satellite" in os.path.basename(f) for f in csv_files):
            title += "FIREBox"
        elif any("FIREm10" in os.path.basename(f) for f in csv_files):
            title += "FIRE"
        elif any("FIRE_m12" in os.path.basename(f) for f in csv_files):
            title += "FIRE_m12"
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

def lookback_time(z):
        return Planck13.lookback_time(z).value
    
def baca_lookback(csv_files, title=None):
    """
    Plot galaxies' b/a vs. c/a space colored by lookback time and connect points with arrows.
    
    Parameters:
    csv_files (list of str): List of paths to the CSV files.
    title (str, optional): Title for the plot.
    H0 (float, optional): Hubble constant in km/s/Mpc. Default is 70.
    Omega_m (float, optional): Matter density parameter. Default is 0.3.
    Omega_Lambda (float, optional): Dark energy density parameter. Default is 0.7.
    
    Returns:
    None
    """

    def lookback_time(z):
        return Planck13.lookback_time(z).value
    # Read CSV files and add a 'source' column based on the filename
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['source'] = os.path.basename(csv_file)
        dfs.append(df)
    
    df_all = pd.concat(dfs, ignore_index=True)
    num_galaxies = len(df_all)
    
    required_columns = ['b/a', 'c/a', 'mass', 'Redshift']
    missing_columns = [col for col in required_columns if col not in df_all.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in the dataframe: {', '.join(missing_columns)}")
    
    # Calculate lookback time for each galaxy
    df_all['Lookback_Time'] = df_all['Redshift'].apply(lookback_time)
    
    # Sort by the lookback time
    df_all = df_all.sort_values(by='Lookback_Time').reset_index(drop=True)
    
    # Categorize datasets based on filename patterns
    df_firebox = df_all[df_all['source'].str.contains('FIREBox', na=False)]
    df_firem10 = df_all[df_all['source'].str.contains('FIREm', na=False)]
    df_fire_m12 = df_all[df_all['source'].str.contains('FIRE_m12', na=False)]
    
    ba_firebox = df_firebox['b/a']
    ca_firebox = df_firebox['c/a']
    mass_firebox = df_firebox['mass']
    log_mass_firebox = np.log10(mass_firebox)
    
    ba_firem10 = df_firem10['b/a']
    ca_firem10 = df_firem10['c/a']
    mass_firem10 = df_firem10['mass']
    log_mass_firem10 = np.log10(mass_firem10)
    
    ba_fire_m12 = df_fire_m12['b/a']
    ca_fire_m12 = df_fire_m12['c/a']
    mass_fire_m12 = df_fire_m12['mass']
    log_mass_fire_m12 = np.log10(mass_fire_m12)
    
    csfont = {'size': '15'}
    csfont1 = {'size': '13'}
    csfont2 = {'size': '15'}
    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_facecolor('black')
    
    # Create a colormap that transitions from blue to red based on lookback time
    colormap = cm.get_cmap('inferno')
    min_lookback_time = df_all['Lookback_Time'].min()
    max_lookback_time = df_all['Lookback_Time'].max()
    normalize = plt.Normalize(vmin=min_lookback_time, vmax=max_lookback_time)
    
    scatter1 = plt.scatter(ba_firebox, ca_firebox, c=df_firebox['Lookback_Time'], alpha=0.8, s=5, cmap='inferno', norm=normalize, marker='.', zorder=6, label='FIREBox')
    scatter2 = plt.scatter(ba_firem10, ca_firem10, c=df_firem10['Lookback_Time'], alpha=1, s=60, cmap='inferno', norm=normalize, marker='^', zorder=6, label='FIRE')
    scatter3 = plt.scatter(ba_fire_m12, ca_fire_m12, c=df_fire_m12['Lookback_Time'], alpha=1, s=60, cmap='inferno', norm=normalize, marker='o', zorder=7, label='FIRE_m12')
    
    cbar = plt.colorbar(scatter1)
    cbar.set_label('Lookback Time (Gyr)', **csfont1)
    
    plt.legend(loc='best')
    
    plt.xlabel('b/a', color='black', **csfont1)
    plt.ylabel('c/a', color='black', **csfont1)
    
    # Set the title directly if provided, otherwise determine based on input files
    if title is None:
        title = "Galaxies in b/a vs. c/a space from "
        if any("FIREBox_satellite" in os.path.basename(f) for f in csv_files):
            title += "FIREBox"
        elif any("FIREm10" in os.path.basename(f) for f in csv_files):
            title += "FIRE"
        elif any("FIRE_m12" in os.path.basename(f) for f in csv_files):
            title += "FIRE_m12"
        title += f" (N={num_galaxies})"
    
    plt.title(title, color='black', **csfont)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Draw arrows connecting points ordered by lookback time
    for i in range(len(df_all) - 1):
        plt.arrow(df_all['b/a'][i], df_all['c/a'][i], 
                  df_all['b/a'][i+1] - df_all['b/a'][i], 
                  df_all['c/a'][i+1] - df_all['c/a'][i],
                  color='white', alpha=0.8, head_width=0.02, head_length=0.02, zorder = 6)
    
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
def radius_check(csv_file, title = None):
    # Define the lookback time calculation function
    def lookback_time(z):
        return Planck13.lookback_time(z).value
    
    # Load data from the CSV file
    df = pd.read_csv(csv_file)
    
    # Calculate Lookback Time and add it as a new column
    df['Lookback_Time'] = df['Redshift'].apply(lookback_time)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Lookback_Time'], df['Max Radius'], color='b')
    plt.xlabel('Lookback Time (Gyr)')
    plt.ylabel('Max Radius')
    if title == None:
        plt.title('Max Radius as a function of Lookback Time')
    else: 
        plt.title(title)
    plt.grid(True)
    plt.show()

import matplotlib.image as mpimg  # Add this import to handle image loading

def baca4(csv_files, image_path = 'graphic.png'):
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
    if (missing_columns):
        raise ValueError(f"Missing columns in the dataframe: {', '.join(missing_columns)}")
    
    # Categorize datasets based on filename patterns
    df_firebox = df_all[df_all['source'].str.contains('FIREBox', na=False)]
    df_firem10 = df_all[df_all['source'].str.contains('FIREm', na=False)]
    df_fire_m12 = df_all[df_all['source'].str.contains('FIRE_m12', na=False)]
    
    ba_firebox = df_firebox['b/a']
    ca_firebox = df_firebox['c/a']
    mass_firebox = df_firebox['mass']
    log_mass_firebox = np.log10(mass_firebox)
    
    ba_firem10 = df_firem10['b/a']
    ca_firem10 = df_firem10['c/a']
    mass_firem10 = df_firem10['mass']
    log_mass_firem10 = np.log10(mass_firem10)
    
    ba_fire_m12 = df_fire_m12['b/a']
    ca_fire_m12 = df_fire_m12['c/a']
    mass_fire_m12 = df_fire_m12['mass']
    log_mass_fire_m12 = np.log10(mass_fire_m12)
    
    csfont = {'size': '15'}
    csfont1 = {'size': '13'}
    csfont2 = {'size': '15'}
    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_facecolor('black')
    
    # Overlay the external image
    img = mpimg.imread(image_path)  # Load the image
    ax.imshow(img, extent=[0, 1, 0, 1], aspect='auto', alpha=1, zorder=4)  # Adjust extent and alpha as needed
    
    scatter1 = plt.scatter(ba_firebox, ca_firebox, c=log_mass_firebox, alpha=0.8, s=5, cmap='inferno', marker='.', zorder=1, label='FIREBox')
    scatter2 = plt.scatter(ba_firem10, ca_firem10, c=log_mass_firem10, alpha=1, s=60, cmap='inferno', marker='^', zorder=1, label='FIRE')
    scatter3 = plt.scatter(ba_fire_m12, ca_fire_m12, c=log_mass_fire_m12, alpha=1, s=60, cmap='inferno', marker='o', zorder=1, label='FIRE_m12')
    
    vmin = df_all['mass'].apply(np.log10).min()-1
    vmax = df_all['mass'].apply(np.log10).max()
    scatter1.set_clim(vmin=vmin, vmax=vmax)
    scatter2.set_clim(vmin=vmin, vmax=vmax)
    scatter3.set_clim(vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar(scatter1)
    cbar.set_label('log10(Stellar Mass)', **csfont1)
    
    plt.xlabel('b/a', color='black', **csfont1)
    plt.ylabel('c/a', color='black', **csfont1)
    
    # Determine the appropriate title based on the input files
    title = "Galaxies as Ellipsoids in b/a vs. c/a Space"
    
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

def baca3(csv_files):
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
    
    # Categorize datasets based on filename patterns
    df_firebox = df_all[df_all['source'].str.contains('FIREBox', na=False)]
    df_firem10 = df_all[df_all['source'].str.contains('FIREm', na=False)]
    df_fire_m12 = df_all[df_all['source'].str.contains('FIRE_m12', na=False)]
    
    ba_firebox = df_firebox['b/a']
    ca_firebox = df_firebox['c/a']
    mass_firebox = df_firebox['mass']
    log_mass_firebox = np.log10(mass_firebox)
    
    ba_firem10 = df_firem10['b/a']
    ca_firem10 = df_firem10['c/a']
    mass_firem10 = df_firem10['mass']
    log_mass_firem10 = np.log10(mass_firem10)
    
    ba_fire_m12 = df_fire_m12['b/a']
    ca_fire_m12 = df_fire_m12['c/a']
    mass_fire_m12 = df_fire_m12['mass']
    log_mass_fire_m12 = np.log10(mass_fire_m12)
    
    csfont = {'size': '15'}
    csfont1 = {'size': '13'}
    csfont2 = {'size': '15'}
    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_facecolor('black')
    
    scatter1 = plt.scatter(ba_firebox, ca_firebox, c=log_mass_firebox, alpha=0.8, s=5, cmap='inferno', marker='.', zorder=1, label='FIREBox')
    scatter2 = plt.scatter(ba_firem10, ca_firem10, c=log_mass_firem10, alpha=1, s=60, cmap='inferno', marker='^', zorder=1, label='FIRE')
    scatter3 = plt.scatter(ba_fire_m12, ca_fire_m12, c=log_mass_fire_m12, alpha=1, s=60, cmap='inferno', marker='o', zorder=1, label='FIRE_m12')
    
    vmin = df_all['mass'].apply(np.log10).min()-1
    vmax = df_all['mass'].apply(np.log10).max()
    scatter1.set_clim(vmin=vmin, vmax=vmax)
    scatter2.set_clim(vmin=vmin, vmax=vmax)
    scatter3.set_clim(vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar(scatter1)
    cbar.set_label('log10(Stellar Mass)', **csfont1)
    
    
    plt.xlabel('b/a', color='black', **csfont1)
    plt.ylabel('c/a', color='black', **csfont1)
    
    # Determine the appropriate title based on the input files
    title = "Galaxies as Ellipsoids in b/a vs. c/a Space"
    
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



def baca_redshift(csv_files, title=None):
    # Read CSV files and add a 'source' column based on the filename
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['source'] = os.path.basename(csv_file)
        dfs.append(df)
    
    df_all = pd.concat(dfs, ignore_index=True)
    num_galaxies = len(df_all)
    
    required_columns = ['b/a', 'c/a', 'mass', 'Redshift']
    missing_columns = [col for col in required_columns if col not in df_all.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in the dataframe: {', '.join(missing_columns)}")
    
    # Sort by the redshift
    df_all = df_all.sort_values(by='Redshift').reset_index(drop=True)
    
    # Categorize datasets based on filename patterns
    df_firebox = df_all[df_all['source'].str.contains('FIREBox', na=False)]
    df_firem10 = df_all[df_all['source'].str.contains('FIREm', na=False)]
    df_fire_m12 = df_all[df_all['source'].str.contains('FIRE_m12', na=False)]
    
    ba_firebox = df_firebox['b/a']
    ca_firebox = df_firebox['c/a']
    mass_firebox = df_firebox['mass']
    log_mass_firebox = np.log10(mass_firebox)
    
    ba_firem10 = df_firem10['b/a']
    ca_firem10 = df_firem10['c/a']
    mass_firem10 = df_firem10['mass']
    log_mass_firem10 = np.log10(mass_firem10)
    
    ba_fire_m12 = df_fire_m12['b/a']
    ca_fire_m12 = df_fire_m12['c/a']
    mass_fire_m12 = df_fire_m12['mass']
    log_mass_fire_m12 = np.log10(mass_fire_m12)
    
    csfont = {'size': '15'}
    csfont1 = {'size': '13'}
    csfont2 = {'size': '15'}
    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_facecolor('black')
    
    # Create a colormap that transitions from blue to red based on redshift
    colormap = cm.get_cmap('inferno')
    min_redshift = df_all['Redshift'].min()
    max_redshift = df_all['Redshift'].max()
    normalize = plt.Normalize(vmin=min_redshift, vmax=max_redshift)
    
    scatter1 = plt.scatter(ba_firebox, ca_firebox, c=df_firebox['Redshift'], alpha=0.8, s=5, cmap='inferno', norm=normalize, marker='.', zorder=6, label='FIREBox')
    scatter2 = plt.scatter(ba_firem10, ca_firem10, c=df_firem10['Redshift'], alpha=1, s=60, cmap='inferno', norm=normalize, marker='^', zorder=6, label='FIRE')
    scatter3 = plt.scatter(ba_fire_m12, ca_fire_m12, c=df_fire_m12['Redshift'], alpha=1, s=60, cmap='inferno', norm=normalize, marker='o', zorder=7, label='FIRE_m12')
    
    cbar = plt.colorbar(scatter1)
    cbar.set_label('Redshift', **csfont1)
    
    plt.legend(loc='best')
    
    plt.xlabel('b/a', color='black', **csfont1)
    plt.ylabel('c/a', color='black', **csfont1)
    
    # Set the title directly if provided, otherwise determine based on input files
    if title is None:
        title = "Galaxies in b/a vs. c/a space from "
        if any("FIREBox_satellite" in os.path.basename(f) for f in csv_files):
            title += "FIREBox"
        elif any("FIREm10" in os.path.basename(f) for f in csv_files):
            title += "FIRE"
        elif any("FIRE_m12" in os.path.basename(f) for f in csv_files):
            title += "FIRE_m12"
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

def histogram_2d(coords, projection='xy', bins=100, lim=None, title=None, ax=None, show_labels=True, cmap='viridis', ellipse_params=None, colorbar=False):
    """
    Create a 2D histogram of the coordinates, with the specified projection and optional axis limits.
    
    Optionally, add an ellipse to the plot and a white line labeled '5 kpc' at the bottom.

    Parameters
    ----------
    coords : array-like
        Array of shape (n, 3) containing the x, y, z coordinates.
    projection : str, optional
        The projection to plot ('xy', 'zx', 'yz').
    bins : int or [int, int], optional
        The number of bins for the histogram (default is 100).
    lim : float, optional
        The limit for the plot axes (if None, limits are set based on data).
    title : str, optional
        The title of the plot.
    ax : matplotlib.axes.Axes, optional
        The axes object to plot on. If None, a new figure and axes will be created.
    show_labels : bool, optional
        Whether to display x and y labels. Default is True.
    cmap : str, optional
        The colormap to use for the histogram. Default is 'viridis'.
    ellipse_params : dict, optional
        Dictionary with keys 'center', 'major', 'minor', 'angle' to define the ellipse. Default is None.
    colorbar : bool, optional
        Whether to display the colorbar. Default is True.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object of the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))  # Make the figure square

    if projection == 'xy':
        x = coords[:, 0]
        y = coords[:, 1]
        xlabel, ylabel = 'x (kpc)', 'y (kpc)'
        ar = 'b/a'
        if ellipse_params:
            center = (ellipse_params['center'][0], ellipse_params['center'][1])
            axis_ratio = ellipse_params['minor'] / ellipse_params['major']
    elif projection == 'zx':
        x = coords[:, 2]
        y = coords[:, 0]
        xlabel, ylabel = 'z (kpc)', 'x (kpc)'
        ar = 'c/a'
        if ellipse_params:
            center = (ellipse_params['center'][2], ellipse_params['center'][0])
            axis_ratio = ellipse_params['minor'] / ellipse_params['major']
    elif projection == 'yz':
        x = coords[:, 1]
        y = coords[:, 2]
        xlabel, ylabel = 'y (kpc)', 'z (kpc)'
        ar = 'c/b'
        if ellipse_params:
            center = (ellipse_params['center'][1], ellipse_params['center'][2])
            axis_ratio = ellipse_params['minor'] / ellipse_params['major']
    else:
        raise ValueError("Invalid projection. Choose from 'xy', 'zx', or 'yz'.")

    if lim is None:
        lim = np.max(np.abs(coords))
    
    # Create 2D histogram
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[-lim, lim], [-lim, lim]])

    # Log scale (add 1 to avoid log(0))
    H = np.log10(H + 1)

    # Plot the 2D histogram
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    img = ax.imshow(H.T, extent=extent, origin='lower', cmap=cmap, aspect='equal')  # Set aspect to 'equal'

    if show_labels:
        ax.set_xlabel(xlabel, color='white')
        ax.set_ylabel(ylabel, color='white')
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')

    if title:
        ax.set_title(title)

    ax.patch.set_facecolor('black')  # Set the background color of the plot area

    plt.grid(False)  # Turn off grid lines

    # Add color bar if requested
    if colorbar:
        cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('log10(count + 1)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Add ellipse if parameters are provided
    if ellipse_params:
        ellipse = patches.Ellipse(
            center,
            width=2*ellipse_params['major'],
            height=2*ellipse_params['minor'],
            angle=ellipse_params.get('angle', 0),
            edgecolor='red',
            linewidth=2,
            facecolor='none'
        )
        ax.add_patch(ellipse)

        # Add axis ratio text
        
        axis_ratio_text = ar + f": {axis_ratio:.2f}"
        ax.text(-lim + 0.05 * (2 * lim), lim - 0.1 * (2 * lim), axis_ratio_text, color='white', fontsize=10, ha='left')

    # Add a white line labeled '5 kpc' at the bottom left of the plot
    fixed_position_x = -lim + 0.05 * (2 * lim)  # 5% offset from the left
    fixed_position_y = -lim + 0.05 * (2 * lim)  # 5% offset from the bottom
    line_length = 5
    ax.plot([fixed_position_x, fixed_position_x + line_length], [fixed_position_y, fixed_position_y], color='white', lw=4)
    ax.text(fixed_position_x + line_length / 2, fixed_position_y + 0.05 * (lim), '5 kpc', color='white', fontsize=10, ha='center')

    return ax



import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
from ast import literal_eval

def preprocess_center_of_mass(center_of_mass_str):
    """
    Convert the center of mass string from CSV into a properly formatted NumPy array.
    """
    formatted_str = center_of_mass_str.strip()
    formatted_str = re.sub(r'\s+', ' ', formatted_str)
    formatted_str = formatted_str.replace('[ ', '[').replace(' ]', ']')
    formatted_str = re.sub(r',\s*,', ', ', formatted_str)
    formatted_str = formatted_str.replace(' ', ',').replace('[,', '[').replace(',]', ']')
    return np.zeros(3)
    #return np.array(literal_eval(formatted_str))

def preprocess_rotation_matrix(rotation_matrix_str):
    """
    Convert the rotation matrix string from CSV into a NumPy array.
    """
    formatted_str = rotation_matrix_str.strip()
    formatted_str = re.sub(r'\s+', ' ', formatted_str)
    formatted_str = formatted_str.replace('[ ', '[').replace(' ]', ']')
    formatted_str = re.sub(r',\s*,', ', ', formatted_str)
    formatted_str = formatted_str.replace(' ', ',').replace('[,', '[').replace(',]', ']')
    return np.array(literal_eval(formatted_str))

def compute_ellipse_params(center_of_mass, max_radius, axis_ratios, rotation_matrix, projection):
    """
    Compute ellipse parameters for a given projection.
    """
    if projection == 'xy':
        major = max_radius
        minor = max_radius * axis_ratios[0]  # b/a
        angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * 180 / np.pi
        angle = 0
    elif projection == 'zx':
        major = max_radius
        minor = max_radius * axis_ratios[1]  # c/a
        angle = np.arctan2(rotation_matrix[2, 0], rotation_matrix[0, 0]) * 180 / np.pi
        angle = 90
    elif projection == 'yz':
        major = max_radius * axis_ratios[0]  # b/a
        minor = max_radius * axis_ratios[1]  # c/a
        angle = np.arctan2(rotation_matrix[2, 1], rotation_matrix[1, 1]) * 180 / np.pi
        angle = 0
    else:
        raise ValueError("Invalid projection. Choose from 'xy', 'zx', or 'yz'.")

    return {
        'center': center_of_mass,
        'major': major,
        'minor': minor,
        'angle': angle
    }

def find_global_limit(folder_path):
    """
    Find the global limit for the axis by searching all CSV files for the largest value of any coordinate.
    """
    max_val = -np.inf
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and 'INS' not in f]
    
    for csv_file in csv_files:
        csv_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(csv_path)
        max_coord = df[['x', 'y', 'z']].max().max()
        max_val = max(max_val, max_coord)
    
    return max_val

def plot_evolution_hist2d(folder_path, ellipse_csv, show_labels=True):
    """
    Create a 3xN grid of 2D scatter plots from CSV files in the specified folder and ellipse data from another CSV file.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing CSV files with coordinates.
    ellipse_csv : str
        Path to the CSV file containing ellipse data.
    show_labels : bool, optional
        Whether to display x and y labels in the plots. Default is True.
    """
    # Determine the global limit for the axis
    global_limit = find_global_limit(folder_path)

    # Get sorted list of CSV files excluding files with 'INS' in their names
    csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv') and 'INS' not in f])

    num_files = len(csv_files)
    if num_files == 0:
        print("No CSV files found in the specified folder.")
        return

    # Load ellipse data
    ellipse_data = pd.read_csv(ellipse_csv)

    # Create a figure with a 3xN grid of subplots
    fig, axes = plt.subplots(3, num_files, figsize=(num_files * 4, 12))

    for i, csv_file in enumerate(csv_files):
        # Load coordinates from CSV file
        csv_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(csv_path)
        coords = df[['x', 'y', 'z']].values
        
        # Extract galaxy name from the CSV file name
        galaxy_name = csv_file.split('_')[0]
        snap_num = csv_file.split('_')[-1].replace('.csv', '')
        redshift = z(snap_num, '/DFS-L/DATA/cosmo/grenache/omyrtaj/fofie/snapshot_times.txt')
        galaxy_lim = global_limit / (1 + redshift)

        # Get ellipse parameters for the current snapshot
        ellipse_row = ellipse_data[ellipse_data['Snapshot'] == int(snap_num)]
        if not ellipse_row.empty:
            max_radius = ellipse_row['Max Radius'].values[0]
            axis_ratios = ellipse_row[['b/a', 'c/a', 'c/b']].values[0]
            center_of_mass_str = ellipse_row['Center of Mass'].values[0]
            rotation_matrix_str = ellipse_row['Rotation Matrix'].values[0]

            # Preprocess the data
            center_of_mass = preprocess_center_of_mass(center_of_mass_str)
            rotation_matrix = preprocess_rotation_matrix(rotation_matrix_str)

            ellipse_params_xy = compute_ellipse_params(center_of_mass, max_radius, axis_ratios, rotation_matrix, 'xy')
            ellipse_params_zx = compute_ellipse_params(center_of_mass, max_radius, axis_ratios, rotation_matrix, 'zx')
            ellipse_params_yz = compute_ellipse_params(center_of_mass, max_radius, axis_ratios, rotation_matrix, 'yz')
        else:
            ellipse_params_xy = ellipse_params_zx = ellipse_params_yz = None

        # Plot projections with titles
        histogram_2d(coords, 'xy', ax=axes[0, i], title=title, show_labels=show_labels, ellipse_params=ellipse_params_xy, lim=galaxy_lim)
        histogram_2d(coords, 'zx', ax=axes[1, i], show_labels=show_labels, ellipse_params=ellipse_params_zx, lim=galaxy_lim)
        histogram_2d(coords, 'yz', ax=axes[2, i], show_labels=show_labels, ellipse_params=ellipse_params_yz, lim=galaxy_lim)
    axes[0, 0].set_title(f"{galaxy_name} at Lookback Time {lt} Gyr", fontsize=12)
    
    # Set column titles
    for ax, csv_file in zip(axes[0], csv_files):
        galaxy_name = csv_file.split('_')[0]
        snap_num = int(csv_file.split('_')[-1].replace('.csv', ''))
        
        redshift = z(snap_num, '/DFS-L/DATA/cosmo/grenache/omyrtaj/fofie/snapshot_times.txt')
        lt = round(lookback_time(redshift), 2)
        ax.set_title(f"{galaxy_name} at Lookback Time {lt} Gyr", fontsize=12)

    # Set row titles
    row_titles = ['XY Projection', 'ZX Projection', 'YZ Projection']
    for ax, title in zip(axes[:, 0], row_titles):
        ax.set_ylabel(title, rotation=90, fontsize=12, labelpad=15)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

import os
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
from ast import literal_eval

def plot_star_populations_hist2d(folder_path, ellipse_csv, start_index=None, end_index=None, show_labels=True):
    """
    Create a 3xN grid of 2D scatter plots from CSV files in the specified folder and ellipse data from another CSV file.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing CSV files with coordinates.
    ellipse_csv : str
        Path to the CSV file containing ellipse data.
    start_index : int, optional
        Starting index of the subplots to be displayed.
    end_index : int, optional
        Ending index of the subplots to be displayed.
    show_labels : bool, optional
        Whether to display x and y labels in the plots. Default is True.
    """
    # Extract the numeric value (age) from the filename for sorting
    def extract_age(filename):
        return float(filename.split('_')[2])

    # Get sorted list of CSV files excluding files with 'INS' in their names
    csv_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.csv') and 'INS' not in f],
        key=extract_age, reverse = True
    )

    # Get the specified range of files and reverse the order
    if start_index is not None and end_index is not None:
        csv_files = csv_files[start_index:end_index + 1]#[::-1]  # Reverse the sliced list

    num_files = len(csv_files)
    if num_files == 0:
        print("No CSV files found in the specified range.")
        return

    # Load ellipse data
    ellipse_data = pd.read_csv(ellipse_csv)

    # Create a figure with a 3xN grid of subplots
    fig, axes = plt.subplots(3, num_files, figsize=(num_files * 4, 12))
    
    for i, csv_file in enumerate(csv_files):
        # Load coordinates from CSV file
        csv_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(csv_path)
        coords = df[['x', 'y', 'z']].values

        # Extract information from the filename
        file_name = os.path.basename(csv_path)
        galaxy_name = file_name.split('_')[0]
        population_id = file_name.split('_')[2]
        snapshot_number = file_name.split('_')[-1].replace('.csv', '')
            
        # Convert snapshot number to redshift
        redshift = z(snapshot_number, '/DFS-L/DATA/cosmo/grenache/omyrtaj/fofie/snapshot_times.txt')
        galaxy_lim = find_global_limit(folder_path) / (1 + redshift) * 1.1

        # Round the first value of each item's Age Range in ellipse_data
        ellipse_data['Rounded Age'] = ellipse_data['Age Range'].apply(lambda x: round(literal_eval(x)[0], 1))
        population_age = float(population_id)
        ellipse_row = ellipse_data[ellipse_data['Rounded Age'] == population_age]

        if not ellipse_row.empty:
            max_radius = ellipse_row['Max Radius'].values[0]
            axis_ratios = ellipse_row[['b/a', 'c/a', 'c/b']].values[0]
            center_of_mass_str = ellipse_row['Center of Mass'].values[0]
            rotation_matrix_str = ellipse_row['Rotation Matrix'].values[0]

            center_of_mass = preprocess_center_of_mass(center_of_mass_str)
            rotation_matrix = preprocess_rotation_matrix(rotation_matrix_str)

            ellipse_params_xy = compute_ellipse_params(center_of_mass, max_radius, axis_ratios, rotation_matrix, 'xy')
            ellipse_params_yz = compute_ellipse_params(center_of_mass, max_radius, axis_ratios, rotation_matrix, 'yz')
            ellipse_params_zx = compute_ellipse_params(center_of_mass, max_radius, axis_ratios, rotation_matrix, 'zx')
        else:
            ellipse_params_xy = ellipse_params_yz = ellipse_params_zx = None

        histogram_2d(coords, 'xy', ax=axes[0, i], show_labels=show_labels, ellipse_params=ellipse_params_xy, lim=galaxy_lim)
        histogram_2d(coords, 'yz', ax=axes[1, i], show_labels=show_labels, ellipse_params=ellipse_params_yz, lim=galaxy_lim)
        histogram_2d(coords, 'zx', ax=axes[2, i], show_labels=show_labels, ellipse_params=ellipse_params_zx, lim=galaxy_lim)

    for i, ax in enumerate(axes[0]):
        file_name = os.path.basename(csv_files[i])
        galaxy_name = file_name.split('_')[0]
        population_id = file_name.split('_')[2]
        pop_end = min(float(population_id) + 0.5, 13.79)
        snapshot_number = file_name.split('_')[-1].replace('.csv', '')

        redshift = z(snapshot_number, '/DFS-L/DATA/cosmo/grenache/omyrtaj/fofie/snapshot_times.txt')
        
        ax.set_title(f"{galaxy_name} Shapes of Stars Born During \n Lookback Time {population_id} - {pop_end} Gyr (z = 0)", fontsize=11)

    row_titles = ['XY Projection', 'YZ Projection', 'ZX Projection']
    for ax, title in zip(axes[:, 0], row_titles):
        ax.set_ylabel(title, rotation=90, fontsize=12, labelpad=15)

    plt.tight_layout()
    plt.show()


def plot_evolution_hist2d1(folder_path, ellipse_csv, start_index=None, end_index=None, show_labels=True):
    """
    Create a 3xN grid of 2D scatter plots from CSV files in the specified folder and ellipse data from another CSV file.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing CSV files with coordinates.
    ellipse_csv : str
        Path to the CSV file containing ellipse data.
    start_index : int, optional
        Starting index of the subplots to be displayed.
    end_index : int, optional
        Ending index of the subplots to be displayed.
    show_labels : bool, optional
        Whether to display x and y labels in the plots. Default is True.
    """
    # Determine the global limit for the axis
    global_limit = find_global_limit(folder_path)

    # Get sorted list of CSV files excluding files with 'INS' in their names
    csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv') and 'INS' not in f])
    
    # Get the specified range of files
    if start_index is not None and end_index is not None:
        csv_files = csv_files[start_index:end_index + 1]

    num_files = len(csv_files)
    if num_files == 0:
        print("No CSV files found in the specified range.")
        return

    # Load ellipse data
    ellipse_data = pd.read_csv(ellipse_csv)

    # Create a figure with a 3xN grid of subplots
    fig, axes = plt.subplots(3, num_files, figsize=(num_files * 4, 12))

    for i, csv_file in enumerate(csv_files):
        # Load coordinates from CSV file
        csv_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(csv_path)
        coords = df[['x', 'y', 'z']].values

        # Extract galaxy name from the CSV file name
        galaxy_name = csv_file.split('_')[0]
        snap_num = csv_file.split('_')[-1].replace('.csv', '')
        redshift = z(snap_num, '/DFS-L/DATA/cosmo/grenache/omyrtaj/fofie/snapshot_times.txt')
        galaxy_lim = global_limit / (1 + redshift)

        # Get ellipse parameters for the current snapshot
        ellipse_row = ellipse_data[ellipse_data['Snapshot'] == int(snap_num)]
        if not ellipse_row.empty:
            max_radius = ellipse_row['Max Radius'].values[0]
            axis_ratios = ellipse_row[['b/a', 'c/a', 'c/b']].values[0]
            center_of_mass_str = ellipse_row['Center of Mass'].values[0]
            rotation_matrix_str = ellipse_row['Rotation Matrix'].values[0]

            # Preprocess the data
            center_of_mass = preprocess_center_of_mass(center_of_mass_str)
            rotation_matrix = preprocess_rotation_matrix(rotation_matrix_str)

            ellipse_params_xy = compute_ellipse_params(center_of_mass, max_radius, axis_ratios, rotation_matrix, 'xy')
            ellipse_params_yz = compute_ellipse_params(center_of_mass, max_radius, axis_ratios, rotation_matrix, 'yz')
            ellipse_params_zx = compute_ellipse_params(center_of_mass, max_radius, axis_ratios, rotation_matrix, 'zx')
        else:
            ellipse_params_xy = ellipse_params_yz = ellipse_params_zx = None

        # Plot projections with titles
        histogram_2d(coords, 'xy', ax=axes[0, i], show_labels=show_labels, ellipse_params=ellipse_params_xy, lim=galaxy_lim)
        histogram_2d(coords, 'yz', ax=axes[1, i], show_labels=show_labels, ellipse_params=ellipse_params_yz, lim=galaxy_lim)
        histogram_2d(coords, 'zx', ax=axes[2, i], show_labels=show_labels, ellipse_params=ellipse_params_zx, lim=galaxy_lim)

    # Set column titles for the first column only (XY projections)
    for i, ax in enumerate(axes[0]):
        csv_file = csv_files[i]
        galaxy_name = csv_file.split('_')[0]
        snap_num = int(csv_file.split('_')[-1].replace('.csv', ''))

        redshift = z(snap_num, '/DFS-L/DATA/cosmo/grenache/omyrtaj/fofie/snapshot_times.txt')
        lt = round(lookback_time(redshift), 2)
        ax.set_title(f"{galaxy_name} at Lookback Time {lt} Gyr", fontsize=12)

    # Set row titles
    row_titles = ['XY Projection', 'YZ Projection', 'ZX Projection']
    for ax, title in zip(axes[:, 0], row_titles):
        ax.set_ylabel(title, rotation=90, fontsize=12, labelpad=15)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def scatter_2d(coords, projection='xy', lim=None, title=None, ax=None, show_labels=False):
    """
    Create a 2D scatter plot of the coordinates, with the specified projection and optional axis limits.
    
    Parameters
    ----------
    coords : array-like
        Array of shape (n, 3) containing the x, y, z coordinates.
    projection : str, optional
        The projection to plot ('xy', 'zx', 'yz').
    lim : float, optional
        The limit for the plot axes (if None, limits are set based on data).
    title : str, optional
        The title of the plot.
    ax : matplotlib.axes.Axes, optional
        The axes object to plot on. If None, a new figure and axes will be created.
    show_labels : bool, optional
        Whether to display x and y labels. Default is True.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object of the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    if projection == 'xy':
        x = coords[:, 0]
        y = coords[:, 1]
        xlabel, ylabel = 'x (kpc)', 'y (kpc)'
    elif projection == 'zx':
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
    
    ax.scatter(x, y, s=1, alpha=0.5, color='c')  # Adjust the marker size 's' and color as needed

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    if show_labels:
        ax.set_xlabel(xlabel, color='black')
        ax.set_ylabel(ylabel, color='black')
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    if title:
        ax.set_title(title, color='black')
    
    ax.patch.set_facecolor('black')  # Set the background color of the plot area

    plt.grid(False)  # Turn off grid lines
    
    return ax

def plot_evolution_hist2d(folder_path, ellipse_csv, show_labels=True):
    """
    Create a 3xN grid of 2D scatter plots from CSV files in the specified folder and ellipse data from another CSV file.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing CSV files with coordinates.
    ellipse_csv : str
        Path to the CSV file containing ellipse data.
    show_labels : bool, optional
        Whether to display x and y labels in the plots. Default is True.
    """
    # Determine the global limit for the axis
    global_limit = find_global_limit(folder_path)

    # Get sorted list of CSV files excluding files with 'INS' in their names
    csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv') and 'INS' not in f])

    num_files = len(csv_files)
    if num_files == 0:
        print("No CSV files found in the specified folder.")
        return

    # Load ellipse data
    ellipse_data = pd.read_csv(ellipse_csv)

    # Create a figure with a 3xN grid of subplots
    fig, axes = plt.subplots(3, num_files, figsize=(num_files * 4, 12))
    for i, csv_file in enumerate(csv_files):
        # Load coordinates from CSV file
        csv_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(csv_path)
        coords = df[['x', 'y', 'z']].values

        # Extract galaxy name from the CSV file name
        galaxy_name = csv_file.split('_')[0]
        snap_num = csv_file.split('_')[-1].replace('.csv', '')
        redshift = z(snap_num, '/DFS-L/DATA/cosmo/grenache/omyrtaj/fofie/snapshot_times.txt')
        galaxy_lim = global_limit / (1 + redshift)
        # Title for the first column
        title = f"{galaxy_name} (stellar age < 0.1Gyr) projections at snapshot {snap_num}" if i == 0 else None

        # Get ellipse parameters for the current snapshot
        ellipse_row = ellipse_data[ellipse_data['Snapshot'] == int(snap_num)]
        if not ellipse_row.empty:
            max_radius = ellipse_row['Max Radius'].values[0]
            axis_ratios = ellipse_row[['b/a', 'c/a', 'c/b']].values[0]
            center_of_mass_str = ellipse_row['Center of Mass'].values[0]
            rotation_matrix_str = ellipse_row['Rotation Matrix'].values[0]

            # Preprocess the data
            center_of_mass = preprocess_center_of_mass(center_of_mass_str)
            rotation_matrix = preprocess_rotation_matrix(rotation_matrix_str)

            ellipse_params_xy = compute_ellipse_params(center_of_mass, max_radius, axis_ratios, rotation_matrix, 'xy')
            ellipse_params_zx = compute_ellipse_params(center_of_mass, max_radius, axis_ratios, rotation_matrix, 'zx')
            ellipse_params_yz = compute_ellipse_params(center_of_mass, max_radius, axis_ratios, rotation_matrix, 'yz')
        else:
            ellipse_params_xy = ellipse_params_zx = ellipse_params_yz = None

        # Plot projections with titles
        histogram_2d(coords, 'xy', ax=axes[0, i], title=title, show_labels=show_labels, ellipse_params=ellipse_params_xy, lim=galaxy_lim)
        histogram_2d(coords, 'zx', ax=axes[1, i], show_labels=show_labels, ellipse_params=ellipse_params_zx, lim=galaxy_lim)
        histogram_2d(coords, 'yz', ax=axes[2, i], show_labels=show_labels, ellipse_params=ellipse_params_yz, lim=galaxy_lim)

    # Set column titles for XY projections
    for i, ax in enumerate(axes[0]):
        csv_file = csv_files[i]
        galaxy_name = csv_file.split('_')[0]
        snap_num = int(csv_file.split('_')[-1].replace('.csv', ''))
        
        redshift = z(snap_num, '/DFS-L/DATA/cosmo/grenache/omyrtaj/fofie/snapshot_times.txt')
        lt = round(lookback_time(redshift), 2)
        ax.set_title(f"{galaxy_name} at Lookback Time {lt} Gyr", fontsize=12)

    # Set row titles
    row_titles = ['XY Projection', 'ZX Projection', 'YZ Projection']
    for ax, title in zip(axes[:, 0], row_titles):
        ax.set_ylabel(title, rotation=90, fontsize=12, labelpad=15)
    
    # Adjust layout
    plt.tight_layout()
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
    
    # Create a hexbin plot with logarithm of counts
    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(x, y, gridsize=gridsize, cmap=cmap)
    cb = plt.colorbar(hb)
    cb.set_label('Log Counts')

    # Transform counts to log scale
    counts = hb.get_array()
    cb.set_ticks([np.min(counts), np.max(counts)])
    cb.set_ticklabels([f'{int(np.exp(min(counts)))}', f'{int(np.exp(max(counts)))}'])

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
