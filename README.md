# FIRE-Geometry
LOOOOOL HAHA HAH AHAHAHAHAAHHAHA

Last updated Jul 25 2024

mg.py: a collection of functions to measure galaxy shapes. the main function is iter_RIT_3D, which iteratively calculates the reduced inertia tensor to measure a galaxy's shape. Other functions such as measure_m12, find_galaxy_shape3d, and outlier3d are for personal use to deal with different galaxy file paths and formatting (e.g. masked FIRE m12's from Patrick, older snapshots from Olti, FIREBox from Jorge).

plot_galaxy.py : baca_age and baca_lookback are the main ones to create b/a vs. c/a diagrams. baca_age1 and baca_lookback1 are slightly modified versions that help with baca_combine, which creates the row of 3 diagrams shown in figures.ipynb. baloga3d and baloga2d create b/a vs. log(a) diagrams either from grabbing all the axis ratios (b/a, c/a, c/b) of a measured 3D galaxy or from a set of 3D-projected-into-2D galaxies measured using a 2d iterative reduced inertia tensor or a Sersic fit. There are also functions such as hexbin, scatter_2d, scatter_3d, scatter_ellipsoid_3d that help visualize a galaxy before and after measurement. 

deviations.ipynb: a statistical test to see how many particles we need to accurately measure the shape of a galaxy. We designate the measurement with all particles within 0.1 r_vir to be the "True" shape and measure how the shape deviates from the true shape as we randomly delete off fractions of particles. The conclusion is that we need roughly 1000 particles to provide an accurate estimate of the shape. 

FIRE_3D.ipynb: measuring FIRE2 zoom-in m10-m11 galaxies

FIREBox_3D.ipynb: measuring FIREBox galaxies

FIRE_Pm12_3d.ipynb: measuring pre-masked FIRE2 zoom-in m12 galaxies

FIRE_m12_redshifts.ipynb: measuring the evolution of FIRE2 zoom-in m12 galaxy morphology over redshift

FIRE_m12_redshifts_ys.ipynb: measuring the evolution of FIRE2 zoom-in m12 galaxy morphology over redshift of ONLY young stars (<0.1 Gyr)

FIRE_m11_redshifts.ipynb: measuring the evolution of FIRE2 zoom-in m11 galaxy morphology over redshift (I need to find old snapshots, I only have them at z=0)

format_files.ipynb: this is literally just concatenating all the CSV's in one folder together into a new CSV file. I realize that I probably should've had just set it to append to an existing csv, but this works for now. 

identify_outliers.ipynb: here, I identify FIREBox galaxies that may not be measuring correctly because the galaxy is actually 2 or 3 galaxies in the same file. I try to pick through them by reading the CSV files measured by FIREBox_3D.ipynb and picking out the ones with abnormally large max_radius. Then, I plot them, and if they turn out to be multiple galaxies, I measure them individually as detailed in outliers_ipynb.

measure_outliers.ipynb: I measure the multiple-galaxy systems by visually guessing a center and radius for each galaxy in the system by using the plots from identify_outliers.ipynb. 

projections.ipynb: randomly projects a galaxy and measures its 2D shape using a 2d iterative reduced inertia tensor. 

stellar_age.ipynb: This measures the shapes of the populations of the stellar ages of a galaxy at some certain redshift. For example, we can bin the stellar ages as 0-0.5, 0.5-1.0, ..., 13.5-14.0 Gyr and measure each shape of the stellar age population. 

tests.ipynb: it's in the name

