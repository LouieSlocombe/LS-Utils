#!/usr/bin/python3
import os
import re
import shutil
import subprocess

import numpy as np
import scipy
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
from ase import io
from ase.visualize.plot import plot_atoms
from scipy import signal
from scipy import special

if os.name == 'nt':
    from findiff import FinDiff
    import sympy as sym

from matplotlib import pyplot as plt

from scipy import linalg
import math
# import numba
from scipy.signal import find_peaks
# import cupy as cp
# import pyfftw
import multiprocessing

#######################################################################################################################
# USEFUL PLOTTING and I/O
# nice plot
def n_plot(xlab, ylab, xs=14, ys=14):
    """
    Makes a plot look nice by introducing ticks, labels, and making it tight
    :param xlab: x axis label
    :param ylab: y axis label
    :param xs: x axis text size
    :param ys: y axis text size
    :return: None
    """
    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', labelsize=ys - 2, direction='in', length=6, width=2)
    plt.tick_params(axis='both', which='minor', labelsize=ys - 2, direction='in', length=4, width=2)
    plt.tick_params(axis='both', which='both', top=True, right=True)
    plt.xlabel(xlab, fontsize=xs)
    plt.ylabel(ylab, fontsize=ys)
    plt.tight_layout()
    return None


# Check OS then plot
def os_plot_show(os_name='nt'):
    """
    Checks the system OS. This is to prevent plotting to HPC.
    nt is windows
    Can trick by making the OS name something else to prevent it from plotting to screen

    :param os_name: The name of the operating system
    :return: None
    """
    # Check if the os is windows
    if os.name == os_name:
        plt.show()
    plt.close()
    return None


# Save 3d
def save_3d(data, dir, header):
    """
    Save 3d data to a given directory
    Taken from:
    https://stackoverflow.com/questions/3685265/how-to-write-a-multidimensional-array-to-a-text-file
    :param data: input 3d data
    :param dir: directory to save to
    :param header: any comments to save to file
    :return:
    """

    # Write the array to disk
    with open(dir, 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape :' + str(list(np.shape(data))) + ': ' + header + '\n')

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in data:
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            np.savetxt(outfile, data_slice)  # , fmt='%-7.2f')

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')
    return None


# Plotting 3D scatter plots
def scatter_3d(xx, yy, zz, x_lab='X', y_lab='Y', z_lab='Z', fig_name=None):
    """
    Now replaces plot_3d_xyz
    Plots given 3D data
    :param xx: x data
    :param yy: y data
    :param zz: z data
    :param x_lab: x label
    :param y_lab: y label
    :param z_lab: z label
    :param fig_name: file to save to
    :return:
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xx, yy, zz, c='r', marker='o')
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    ax.set_zlabel(z_lab)
    if fig_name != None:
        plt.savefig(fig_name)
    os_plot_show()
    return None


# Pull specific data by using a sub string as a keyword
def pull(Arr, sub):
    """
    If there is a list of strings it pulls out numerical values from lines which contain the substring
    :param Arr: input array
    :param sub: sub string
    :return: list of values
    """
    # Strip all the non-numbers
    return [re.sub("[^0123456789\.]", "", i) for i in Arr if sub in i]


# plots the last image from a traj file
def plot_traj(file_traj='prod.traj', rad=.8):
    """
    Plots a .traj file and saves as a pdf to the current working dir
    :param file_traj: input file
    :param rad: radii of each atom
    :return: None
    """
    # remove and re-apply file extenstions
    file_traj = os.path.splitext(file_traj)[0] + '.traj'
    atoms = io.read(file_traj, -1)
    fig, axarr = plt.subplots(1, 4, figsize=(15, 5))
    plot_atoms(atoms, axarr[0], radii=rad, rotation=('0x,0y,0z'))
    plot_atoms(atoms, axarr[1], radii=rad, rotation=('90x,45y,0z'))
    plot_atoms(atoms, axarr[2], radii=rad, rotation=('45x,45y,0z'))
    plot_atoms(atoms, axarr[3], radii=rad, rotation=('90x,0y,0z'))
    axarr[0].set_axis_off()
    axarr[1].set_axis_off()
    axarr[2].set_axis_off()
    axarr[3].set_axis_off()
    fig.savefig(os.path.splitext(file_traj)[0] + ".pdf")
    os_plot_show()
    return None


# converts a .traj to a xyz file
def convert_traj_2_xyz(file_traj='react.traj', file_xyz='test.xyz'):
    """
    Converts a .traj file to a .xyz formatted file
    :param file_traj: trajectory file
    :param file_xyz: xyz file to save to
    :return:
    """
    # remove and re-apply file extenstions
    file_traj = os.path.splitext(file_traj)[0] + '.traj'
    file_xyz = os.path.splitext(file_xyz)[0] + '.xyz'

    # Grab the cell from file
    a = io.read(file_traj)

    symbols = a.get_chemical_symbols()
    N = len(symbols)
    positions = a.get_positions()
    A = [str(N)]
    A.append(' ')
    # loop over the elements
    for i in range(N):
        # put together the symbol and the xyz on one string line
        A.append(str(symbols[i]) + '   ' + str(positions[i]).strip('[]'))
    # Save the file
    np.savetxt(file_xyz, A, format('%s'))
    return None


# Fixes the time axis, giving a good prefix
# Converts the time array to something that is more manageable
def time_axis_fix(t_arr, implied_units=1.0):
    max_val = max(t_arr) * implied_units
    min_val = max(t_arr) * implied_units

    # predefined prefixes
    prefix = {'y': 1e-24,  # yocto
              'z': 1e-21,  # zepto
              'a': 1e-18,  # atto
              'f': 1e-15,  # femto
              'p': 1e-12,  # pico
              'n': 1e-9,  # nano
              'u': 1e-6,  # micro
              'm': 1e-3,  # mili
              'c': 1e-2,  # centi
              'd': 1e-1,  # deci
              'k': 1e3,  # kilo
              'M': 1e6,  # mega
              'G': 1e9,  # giga
              'T': 1e12,  # tera
              'P': 1e15,  # peta
              'E': 1e18,  # exa
              'Z': 1e21,  # zetta
              'Y': 1e24,  # yotta
              }

    # grab vals and keys
    vals = [i for i in prefix.values()]
    keys = [i for i in prefix.keys()]

    # Find the closest value
    dif = abs(np.log(vals) - np.log(max_val))

    # Find the location
    loc = np.where(dif == min(dif))[0][0]

    # Fix the time axis
    t_arr = t_arr * implied_units / vals[loc]

    return np.array(t_arr), keys[loc]


#######################################################################################################################
# COORDINATE TRANSFORMS AND CLUSTERING
# Collect up arrays of x,y,z and outputs a 2D array of coordinate triplets
def collect_xyz_2_coords(x, y, z):
    """
    Collect up arrays of x,y,z and outputs a 2D array of coordinate triplets
    :param x: x data
    :param y: y data
    :param z: z data
    :return: array of [x,z,y]
    """
    return np.stack((np.array(x), np.array(y), np.array(z), np.ones(len(x))), axis=-1)


# Splits up 2D array of coordinate triplets into rows of x,y,z
def uncollect_xyz_2_coords(arr):
    """
    Splits up 2D array of coordinate triplets into rows of x,y,z
    :param arr: 2d array
    :return: x,y,z
    """
    # Find the shape
    sh = arr.shape
    # Check if the matrix has the stack of ones at the end
    if sh[1] == 4:
        x, y, z, ones = np.split(arr, 4, axis=1)
        return np.transpose(x)[0], np.transpose(y)[0], np.transpose(z)[0]
    # Doesnt have ones
    elif sh[1] == 3:
        x, y, z = np.split(arr, 3, axis=1)
        return np.ravel(x), np.ravel(y), np.ravel(z)


# Plane polar coordinates converter
def plane_polar_coords(A):
    """
    cartesian coordinates to Plane polar coordinates converter
    :param A: input vector of cartesian coordinates
    :return: polar coordinates vector
    """
    # Handles different array shapes and ranks
    sh = A.shape
    if A.ndim == 2 and (sh[0] > 1) and (sh[1] >= 2):
        xx = A[:, 0]
        yy = A[:, 1]
        zz = A[:, 2]
    else:
        xx = A[0]
        yy = A[1]
        zz = A[2]
    r = np.sqrt(np.square(xx) + np.square(yy))
    phi = np.arctan2(yy, xx)
    # Handles different array shapes and ranks
    if A.ndim == 2 and (sh[0] > 1) and (sh[1] >= 3):
        return np.column_stack((r, phi))
    else:
        return [r, phi]


# Inverse plane polar coordinates converter
def i_plane_polar_coords(A):
    """
    plane polar coordinates to cartesian coordinates
    :param A: input vector
    :return: output vector
    """
    # Handles different array shapes and ranks
    sh = A.shape
    if A.ndim == 2 and (sh[0] > 1) and (sh[1] >= 2):
        r = A[:, 0]
        phi = A[:, 1]
    else:
        r = A[0]
        phi = A[1]
    xx = r * np.cos(phi)
    yy = r * np.sin(phi)
    zz = 0.0
    # Handles different array shapes and ranks
    if A.ndim == 2 and (sh[0] > 1) and (sh[1] >= 3):
        return np.column_stack((xx, yy, zz, np.ones(len(xx))))
    else:
        return [xx, yy, zz]


# Spherical polar coordinates converter
def spherical_coords(A):
    """
    Cartesian coordinates to spherical polar coordinates
    :param A: input vector
    :return: output vector
    """
    # Handles different array shapes and ranks
    sh = A.shape
    if A.ndim == 2 and (sh[0] > 1) and (sh[1] >= 3):
        xx = A[:, 0]
        yy = A[:, 1]
        zz = A[:, 2]
    else:
        xx = A[0]
        yy = A[1]
        zz = A[2]
    r = np.sqrt(np.square(xx) + np.square(yy) + np.square(zz))
    theta = np.arccos(np.divide(zz, r))
    phi = np.arctan2(yy, xx)
    # Handles different array shapes and ranks
    if A.ndim == 2 and (sh[0] > 1) and (sh[1] >= 3):
        return np.column_stack((r, phi, theta))
    else:
        return [r, phi, theta]


# Inverse spherical polar coordinates converter
def i_spherical_coords(A):
    """
    spherical polar coordinates to cartesian coordinates
    :param A: input vector
    :return: output vector
    """
    # Handles different array shapes and ranks
    sh = np.shape(A)
    if np.ndim(A) == 2 and (sh[0] > 1) and (sh[1] >= 3):
        r = A[:, 0]
        phi = A[:, 1]
        theta = A[:, 2]
    else:
        r = A[0]
        phi = A[1]
        theta = A[2]
    xx = r * np.sin(theta) * np.cos(phi)
    yy = r * np.sin(theta) * np.sin(phi)
    zz = r * np.cos(theta)
    # Handles different array shapes and ranks
    if np.ndim(A) == 2 and (sh[0] > 1) and (sh[1] >= 3):
        return np.column_stack((xx, yy, zz, np.ones(len(xx))))
    else:
        return [xx, yy, zz]


# Deals with stacks of coordinates in applies np.dot
def trans_mat_stacks(A, x):
    """
    helper function which deals with stacks of coordinates in applies np.dot
    in the form A.x
    :param A: input array
    :param x: transformation array
    :return: translated
    """
    mat_shape = np.shape(x)
    if mat_shape[0] > 0 and mat_shape[1] == 4:
        rtn = np.zeros_like(x)
        for i, val in enumerate(x):
            rtn[i, :] = np.dot(A, val)
    else:
        rtn = np.dot(A, x)
    return rtn


# Performs a transformation using a, b, c
def trans_mat_tran(x, vec):
    """
    Performs a transformation using an input vector of [a, b, c]
    Taken from:
    https://www.tutorialspoint.com/computer_graphics/3d_transformation.htm
    :param x: input vector
    :param vec: translation vector
    :return: translated vector
    """
    a = vec[0]
    b = vec[1]
    c = vec[2]
    # translation matrix
    T = np.zeros([4, 4])
    T[0][0] = 1.0
    T[0][1] = 0.0
    T[0][2] = 0.0
    T[0][3] = a

    T[1][0] = 0.0
    T[1][1] = 1.0
    T[1][2] = 0.0
    T[1][3] = b

    T[2][0] = 0.0
    T[2][1] = 0.0
    T[2][2] = 1.0
    T[2][3] = c

    T[3][0] = 0.0
    T[3][1] = 0.0
    T[3][2] = 0.0
    T[3][3] = 1.0
    return trans_mat_stacks(T, x)


# Roation about the x axis
def trans_mat_rot_x(x, theta):
    """
    Performs a rotation about the x axis
    Taken from:
    https://www.tutorialspoint.com/computer_graphics/3d_transformation.htm
    :param x: input array
    :param theta: degree to rotate by
    :return: transformed result
    """
    # Rotation matrix
    T = np.zeros([4, 4])
    T[0][0] = 1.0
    T[0][1] = 0.0
    T[0][2] = 0.0
    T[0][3] = 0.0

    T[1][0] = 0.0
    T[1][1] = np.cos(theta)
    T[1][2] = -np.sin(theta)
    T[1][3] = 0.0

    T[2][0] = 0.0
    T[2][1] = np.sin(theta)
    T[2][2] = np.cos(theta)
    T[2][3] = 0.0

    T[3][0] = 0.0
    T[3][1] = 0.0
    T[3][2] = 0.0
    T[3][3] = 1.0
    return trans_mat_stacks(T, x)


# Roation about the y axis
def trans_mat_rot_y(x, theta):
    """
    Performs a rotation about the y axis
    Taken from:
    https://www.tutorialspoint.com/computer_graphics/3d_transformation.htm
    :param x: input array
    :param theta: degree to rotate by
    :return: transformed result
    """
    # Rotation matrix
    T = np.zeros([4, 4])
    T[0][0] = np.cos(theta)
    T[0][1] = 0.0
    T[0][2] = np.sin(theta)
    T[0][3] = 0.0

    T[1][0] = 0.0
    T[1][1] = 1.0
    T[1][2] = 0.0
    T[1][3] = 0.0

    T[2][0] = -np.sin(theta)
    T[2][1] = 0.0
    T[2][2] = np.cos(theta)
    T[2][3] = 0.0

    T[3][0] = 0.0
    T[3][1] = 0.0
    T[3][2] = 0.0
    T[3][3] = 1.0
    return trans_mat_stacks(T, x)


# Roation about the z axis
def trans_mat_rot_z(x, theta):
    """
    Performs a rotation about the z axis
    Taken from:
    https://www.tutorialspoint.com/computer_graphics/3d_transformation.htm
    :param x: input array
    :param theta: degree to rotate by
    :return: transformed result
    """
    # Rotation matrix
    T = np.zeros([4, 4])
    T[0][0] = np.cos(theta)
    T[0][1] = -np.sin(theta)
    T[0][2] = 0.0
    T[0][3] = 0.0

    T[1][0] = np.sin(theta)
    T[1][1] = np.cos(theta)
    T[1][2] = 0.0
    T[1][3] = 0.0

    T[2][0] = 0.0
    T[2][1] = 0.0
    T[2][2] = 1.0
    T[2][3] = 0.0

    T[3][0] = 0.0
    T[3][1] = 0.0
    T[3][2] = 0.0
    T[3][3] = 1.0
    return trans_mat_stacks(T, x)


# Calculates COM for given mass and position
def com_basic(mass, pos):
    """
    Calculates the center of mass for given vectors of mass and position
    :param mass: mass vector
    :param pos: position vector
    :return: center of mass
    """
    return np.divide(np.sum(np.dot(mass, pos)), np.sum(mass))


# Determines the COM for a given set of coordinates
def trans_mat_com(x):
    """
    Helper function which determines the COM for a given set of collected coordinates [x,y,z]
    ASSUMES THE MASS VECTOR IS UNITY
    :param x: [x,y,z]
    :return: com in the form [x,y,z]
    """
    x, y, z = uncollect_xyz_2_coords(x)
    com_x = com_basic(np.ones_like(x), x)
    com_y = com_basic(np.ones_like(y), y)
    com_z = com_basic(np.ones_like(z), z)
    return [com_x, com_y, com_z]


# Moves things to origin using com, translates by a vector, then transforms back away from origin
def trans_mat_com_tran(x, vec):
    """
    Moves things to origin using com, translates by a vector, then transforms back away from origin
    :param x: [x,y,z]
    :param vec: translation vector
    :return: Transformed vector
    """
    # Determine com location
    com = trans_mat_com(x)
    # Move com to origin
    x = trans_mat_tran(x, np.dot(-1.0, com))
    # Translate by input vector
    x = trans_mat_tran(x, vec)
    # Move com back to original com location
    x = trans_mat_tran(x, np.dot(+1, com))
    return com, x


# Moves things to origin using com, rotates by theta, then transforms back away from origin
def trans_mat_com_rot(x, theta, axis):
    """
    Moves things to origin using com, rotates by theta, then transforms back away from origin
    :param x: [x,y,z]
    :param theta: degree to rotate by
    :param axis: axis to perform the rotation
    :return: Transformed vector
    """
    # Determine com location
    com = trans_mat_com(x)
    # Move com to origin
    x = trans_mat_tran(x, np.dot(-1, com))
    # Rotates
    if axis == 'x':
        x = trans_mat_rot_x(x, theta)
    elif axis == 'y':
        x = trans_mat_rot_y(x, theta)
    elif axis == 'z':
        x = trans_mat_rot_z(x, theta)
    else:
        print('Problem...')
    # Move com back to original com location
    x = trans_mat_tran(x, np.dot(+1, com))
    return com, x


# Find the optimal rigid transformation
def rigid_transform_3d_find(A, B):
    """
    Finding the optimal rotation and translation between
    two sets of corresponding 3D point data, so that they are aligned

    Adapted from https://nghiaho.com/?page_id=671

    :param A: vector A expects Nx3 matrix of points
    :param B: vector B expects Nx3 matrix of points
    :return: R,t
    R = 3x3 rotation matrix
    t = 3x1 column vector
    """
    # Ensure that the shape is the same
    assert A.shape == B.shape
    # Assert matrix type
    A = np.matrixlib.defmatrix.matrix(A)
    B = np.matrixlib.defmatrix.matrix(B)

    N = A.shape[0]  # total points

    # Find the centre of A and B
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA) * BB
    # H = AA.T * BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = Vt.T * U.T

    t = -R * centroid_A.T + centroid_B.T
    return R, t


# Apply the optimal rigid transformation
def rigid_transform_3d_apply(A, B, R, t):
    """
    Applies a rigid transformation of A to try and match B, a comparison of the new vector and B is then made
    https://en.wikipedia.org/wiki/Rigid_transformation
    Adapted from https://nghiaho.com/?page_id=671
    :param A: vector to move
    :param B: Exemplar vector
    :param R: optimum rotation
    :param t: Optimum translation
    :return: Moved vector A and the rmse
    """
    # Assert matrix type
    A = np.matrixlib.defmatrix.matrix(A)
    B = np.matrixlib.defmatrix.matrix(B)
    # Find the size
    n = A.shape[0]
    # Apply the optimal transform
    A2 = (R * A.T) + np.tile(t, (1, n))
    A2 = A2.T
    # Find the error
    err = np.subtract(A2, B)
    rmse = np.sqrt(np.sum(np.square(err)) / n)
    return A2, rmse


# Random rigid scramble transformation
def rigid_transform_3d_scramble(A):
    """
    Applies a rigid rotation and translation scramble to input vector A
    Adapted from https://nghiaho.com/?page_id=671
    :param A: Input vector to scramble
    :return: Scrambled input vector, rotation, and translation
    """
    # Assert matrix type
    A = np.matrixlib.defmatrix.matrix(A)
    # Random rotation and translation
    R = np.mat(np.random.rand(3, 3))
    t = np.mat(np.random.rand(3, 1))

    # Make R a proper rotation matrix, force orthonormal
    U, S, Vt = np.linalg.svd(R)
    R = U * Vt

    # Remove reflection
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = U * Vt

    # Find the size
    n = A.shape[0]
    # Apply the scramble
    B = R * A.T + np.tile(t, (1, n))
    return B.T, R, t


# for a given set of coordinates clusters into two molecules
def cluster(x, y, z, ad_locs):
    """
    For a given set of coordinates clusters into two molecules

    Improvements:
    Needs improvements in the checking algo
    Needs better way to output data
    Generalise to auto-detect Hbonds or from given bond length
    Generalise to work for any number of clusters
    :param x: x vector
    :param y: y vector
    :param z: z vector
    :param ad_locs: H-bonds to fragment by
    :return: the locations of two clusters and the cutoff radius to fragment the clusters
    """
    n_atoms = len(x)
    # Collect data into coords
    tmp = collect_xyz_2_coords(x, y, z)
    coords = tmp[:, :-1]

    # Determine the shortest bond of the hydrogen bonds given
    r = []
    for val in ad_locs:
        tmp = spherical_coords(abs(coords[val[1]] - coords[val[2]]))[0]
        tmp1 = spherical_coords(abs(coords[val[0]] - coords[val[2]]))[0]
        r.append(max(tmp, tmp1))
    print('')
    # Pick the smallest one
    r_cutoff = min(r)

    # pick some location
    a = [0]

    flags = 0
    # Outer loop over kill, confirm nothing is missed
    while True:
        l1 = len(a)
        # Loop over the atoms
        for j in a:
            # Loop over all the atoms in the list
            for i in range(n_atoms):
                select = j
                # Avoid re-adding of values
                if i in a:
                    continue
                # Calculate the distance between the two chosen sites
                dist = spherical_coords(abs(coords[select] - coords[i]))[0]
                # Check if the distance is less than the cut off
                if dist < r_cutoff:
                    a.append(i)
        l2 = len(a)
        # check if the appended atoms changes
        if l1 == l2:
            flags += 1
        # Dont leave until nothing changes for 5 iterations, overkill...
        if flags == 5:
            break

    # Select the rest of the atoms
    b = [i for i in range(n_atoms) if i not in a]
    # returns the locations of two clusters and the cutoff radius
    return a, b, r_cutoff


#######################################################################################################################
# FILE PATH MANIPULATION STUFF
# List only the files in a directory
def file_list(mypath=os.getcwd()):
    """
    List only the files in a directory given by mypath
    :param mypath: specified directory, defaults to current directory
    :return: returns a list of files
    """
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    return onlyfiles


# List only the top level folders in a directory
def folder_list(mypath=os.getcwd()):
    """
    List only the top level folders in a directory given by mypath
    NOTE THIS IS THE SAME AS top_dirs_list
    :param mypath: specified directory, defaults to current directory
    :return: returns a list of folders
    """
    onlyfolders = [f for f in os.listdir(mypath) if os.path.isdir(os.path.join(mypath, f))]
    return onlyfolders


# List only files which contain a substring
def sub_file_list(mypath, sub_str):
    """
    List only files which contain a given substring
    :param mypath: specified directory
    :param sub_str: string to filter by
    :return: list of files which have been filtered
    """
    return [i for i in file_list(mypath) if sub_str in i]


# List only folders which contain a substring
def sub_folder_list(mypath, sub_str):
    """
    List only folders which contain a given substring
    :param mypath: specified directory
    :param sub_str: string to filter by
    :return: list of folders which have been filtered
    """
    return [i for i in folder_list(mypath) if sub_str in i]


# Bring the path back one
def parent_folder(mypath=os.getcwd()):
    """
    Bring the path back by one
    :param mypath: specified directory, defaults to current directory
    :return: parent path
    """
    return os.path.abspath(os.path.join(mypath, os.pardir))


# Backs up a file if it exists
def file_bck(fpath):
    """
    Backs up a file if it exists
    :param fpath: file to check/backup
    :return: None
    """
    if os.path.exists(fpath) == True:
        bck = fpath.split('.')
        assert len(bck) == 2
        dst = bck[0] + '_bck.' + bck[1]
        shutil.copyfile(fpath, dst)
    return None


# Removes all files in a given directory
def file_remove(fpath, f_exit=True):
    """
    Removes all files in a given directory. Can abort if failure detected
    :param fpath:
    :return: None
    """
    for filename in os.listdir(fpath):
        file_path = os.path.join(fpath, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            # Leaves if failure
            if f_exit:
                exit()
    return None


#######################################################################################################################
# MATHS

# Normalise a vector
def normalise(vec):
    """
    Normalises a given eigenvector
    :param vec: input vector of which to normalise
    :return: normalised vector
    """
    # Finds the magnitude by self dot then sqrt
    mag = np.sqrt(np.dot(vec, vec))
    # Normalises
    vec = np.divide(vec, mag)
    return vec


def norm_wf(psi, x):
    """
    Normalises a set of wavefunctions
    :param psi: (list of several or just one) wavefunction
    :param x: x range to integrate over
    :return: normalised wavefunction
    """
    rank = len(np.shape(psi))
    # Enforce normalisation
    if rank == 2:
        # psi = np.array([(p / np.sqrt(np.trapz(p * p.conj(), x))).real for p in psi], dtype=complex)
        psi = np.array([(p / np.sqrt(scipy.integrate.simps(p * p.conj(), x))).real for p in psi], dtype=complex)
    else:
        # psi = np.array((psi / np.sqrt(np.trapz(psi * psi.conj(), x))).real, dtype=complex)
        psi = np.array((psi / np.sqrt(scipy.integrate.simps(psi * psi.conj(), x))).real, dtype=complex)
    return psi


# Determines the probability
def prob(l_low, u_lim, f_rho, x):
    """
    # extract the real diagonal terms
    # tmp = np.diag(np.real(A*np.conjugate(A)))
    tmp = np.diag(np.real(A))
    # Need to pick the location
    tmp = tmp[l_low:u_lim]
    # Integrate over the required range
    # p = scipy.integrate.simps(tmp)
    p = np.sum(tmp)
    """
    p = np.real(scipy.integrate.simps(np.diag(f_rho)[l_low:u_lim], x[l_low:u_lim]))
    return p


def expect_x(f_rho, x):  # expect_x
    # return np.real(np.trace(np.multiply(f_rho, A)))
    return np.real(scipy.integrate.simps(np.multiply(np.diag(f_rho), x), x))


def trace(f_rho, x):
    # pick out the psi components
    # psi = np.diag(mat)
    # tr = np.real(np.sqrt(np.trapz(np.diag(mat), x)))
    tr = np.real(scipy.integrate.simps(np.diag(f_rho), x))
    return tr


def vn_entropy(f_rho, x):
    f_rho = f_rho * np.log(f_rho)
    rtn = -np.trace(f_rho)
    rtn = -trace(f_rho, x)
    return rtn


def von_neumann_entropy(density_matrix, cutoff=10):
    """
    https://arxiv.org/pdf/1209.2575.pdf
    https://cs.stackexchange.com/questions/56261/computing-von-neumann-entropy-efficiently
    https://en.wikipedia.org/wiki/Von_Neumann_entropy
    :param density_matrix:
    :param cutoff:
    :return:
    """
    x = np.mat(density_matrix)
    one = np.identity(x.shape[0])
    base = one - x
    power = base * base
    result = np.trace(base)
    for k in range(2, cutoff):
        result -= np.trace(power) / (k * k - k)
        power = power.dot(base)

    # Twiddly hacky magic.
    a = cutoff
    for k in range(3):
        d = (a + 1) / (4 * a * (a - 1))
        result -= np.trace(power) * d
        power = power.dot(power)
        result -= np.trace(power) * d
        a *= 2
    result -= np.trace(power) / (a - 1) * 0.75
    return result / np.log(2)  # convert from nats to bits


def trace_ratio(f_rho, x):
    rat = trace(np.square(f_rho), x) / np.square(trace(f_rho, x))
    rat = np.trace(np.square(f_rho)) / np.square(np.trace(f_rho))
    return rat


# Check if a matrix is symmetric
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    """
    Check if the input matrix is symmetric, compares real to real and imag to imag!
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
    :param a: input matrix
    :param rtol: The relative tolerance parameter
    :param atol: The absolute tolerance parameter
    :return: bool, true or false
    """
    val = np.real(a)
    comp_real = np.allclose(val, val.T, rtol=rtol, atol=atol)
    val = np.imag(a)
    comp_imag = np.allclose(val, val.T, rtol=rtol, atol=atol)
    return comp_real, comp_imag


# matrix logarithm via eigen decomposition
def logm_eigen(A):
    """
    Calculate the matrix logarithm using the eigen decomposition method
    :param A: Input matrix
    :return:
    """
    _, V = np.linalg.eig(A)
    V_inv = np.linalg.inv(V)
    A_dash = np.matmul(np.matmul(V_inv, A), V)
    tmp = np.matmul(np.matmul(V_inv, np.log(A_dash)), V)
    print('shape', np.shape(tmp))
    return tmp


# Summing all the off diagonal
def norm_l1(a):
    """
    Summing all the off diagonal
    :param a: input matrix
    :return: l1 matrix norm
    """
    # sum all the elements and subtract the trace
    return np.sum(a) - np.trace(a)


# My attempt at the bessel function
def bessel_myn(z, k, nl=100):
    """
    My attempt at the bessel function
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jv.html#scipy.special.jv
    https://en.wikipedia.org/wiki/Bessel_function
    :param z: Order
    :param k: Argument
    :param nl: limit of sum
    :return:
    """
    rtn = 0
    for l in range(nl):
        rtn += rtn + (((-1.0) ** l) / ((2 ** (2 * l + k)) * math.factorial(l) * math.factorial(l))) * z ** (2 * l + k)
    return rtn


# Bessel function of the first kind of real order and complex argument.
def bessel_func(z, k):
    """
    Bessel function of the first kind of real order and complex argument.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jv.html#scipy.special.jv
    https://en.wikipedia.org/wiki/Bessel_function
    :param z: Order
    :param k: Argument
    :return:
    """
    return scipy.special.jv(z, k)


# Modified Bessel function of the first kind.
def besseli_func(z, k):
    """
    https://docs.sympy.org/latest/modules/functions/special.html#sympy.functions.special.bessel.besseli
    :param z: Order
    :param k: Argument
    :return:
    """
    return 1.0j ** (-z) * bessel_func(z, 1.0j * k)


# Generates the cheb polynomials
def cheb_poly(x, nk):
    """
    https://en.wikipedia.org/wiki/Chebyshev_polynomials
    http://mathworld.wolfram.com/ChebyshevPolynomialoftheFirstKind.html
    :param x: data
    :param nk: order of polynomial
    :return: p(x)
    """
    rank = len(np.shape(x))
    if rank >= 2:
        N = np.shape(x)[0]
        T_1k = np.identity(N)
    else:
        T_1k = 1
    T_k = x
    # Check what needs to be returned
    if nk == 0:  # very first term
        return T_1k
    elif nk == 1:  # next term
        return T_k
    else:  # subsequent terms
        for k in range(nk - 1):
            # print('k value',k)
            T_k1 = 2 * x * T_k - T_1k
            T_1k = T_k
            T_k = T_k1
        return T_k


# Determines the matrix exponential via the cheby expansion
def expm_cheb_old(x, n_k=4):
    """
    Determines the matrix exponential via the cheby expansion
    :param x: input data
    :param n_k: order of expansion
    :return:
    """
    # define identity matrix
    I = np.identity(len(x), dtype=complex)
    n_k = 4
    s = 0

    w, _ = np.linalg.eig(x)
    lamdba_min = np.min(w)
    lamdba_max = np.max(w)
    x_bar = (lamdba_max + lamdba_min) * 0.5
    x_delta = (lamdba_max - lamdba_min) * 0.5
    x_norm = ()

    for k in range(1, n_k):
        ran = np.zeros(k, dtype=int)
        ran[k - 1] = 1
        cheb = np.polynomial.chebyshev.Chebyshev(ran)
        print(cheb)
        coef = np.polynomial.chebyshev.cheb2poly(cheb.coef)
        print(coef)
        T_k = np.polynomial.chebyshev.chebval(x, coef)
        print(T_k)
        tmp = (1.0j ** k) * bessel_func(k, -1.0j) * T_k
        s += tmp

    return bessel_func(0, 1.0j) * I + 2 * s


# Determines the matrix exponential via the cheby expansion
def expm_cheb(x, nk=10):
    """
    Determines the matrix exponential via the cheby expansion
    :param x: input data
    :param n_k: order of expansion
    :return:
    """
    N = np.shape(x)[0]
    val0 = bessel_func(1.0j, 0) * np.identity(N, dtype=complex)
    val = np.zeros_like(val0, dtype=complex)
    for fk in range(1, nk + 1, 1):
        # print('outer k:',fk)
        val += (1.0j ** fk) * bessel_func(-1.0j, fk) * cheb_poly(x, fk)
        # val += (1.0j ** fk) * bessel_func(-1.0j, fk) * cheb_poly((x/(2**fk)), fk)
    return val0 + 2.0 * val
    # return (val0 + 2.0 * val)**(2.0*(nk+1))


# Determines the matrix exponential via the pade approximation
def expm_pade(a):
    """
    # Determines the matrix exponential via the pade approximation
    :param a: input data
    :return:
    """
    return linalg.expm(a)


# Calculate the matrix exponential using the eigen decomposition method
def expm_eigen(a):
    """
    Calculate the matrix exponential using the eigen decomposition method
    :param a: Input matrix
    :return:
    """
    d, Y = np.linalg.eig(a)
    Yinv = np.linalg.pinv(Y)
    D = np.diag(np.exp(d))
    Y = np.asmatrix(Y)
    D = np.asmatrix(D)
    Yinv = np.asmatrix(Yinv)
    return Y * D * Yinv


# Calculate the matrix exponential using the power expansion
def expm_power(a, n=10):
    """
    # Calculate the matrix exponential using the power expansion
    :param x: input data
    :param n: order of expansion
    :return:
    """
    N = np.shape(a)[0]
    res = np.identity(N)
    for i in range(n - 1):
        res = res + np.divide(np.power(a, i + 1), math.factorial(i + 1))
    return res


def nufftfreqs(M, df=1):
    """Compute the frequency range used in nufft for M frequency bins"""
    return df * np.arange(-(M // 2), M - (M // 2))


def nudft(x, y, M, df=1.0, iflag=1):
    """Non-Uniform Direct Fourier Transform"""
    sign = -1 if iflag < 0 else 1
    return (1 / len(x)) * np.dot(y, np.exp(sign * 1j * nufftfreqs(M, df) * x[:, np.newaxis]))


def _compute_grid_params(M, eps):
    # Choose Msp & tau from eps following Dutt & Rokhlin (1993)
    if eps <= 1E-33 or eps >= 1E-1:
        raise ValueError("eps = {0:.0e}; must satisfy "
                         "1e-33 < eps < 1e-1.".format(eps))
    ratio = 2 if eps > 1E-11 else 3
    Msp = int(-np.log(eps) / (np.pi * (ratio - 1) / (ratio - 0.5)) + 0.5)
    Mr = max(ratio * M, 2 * Msp)
    lambda_ = Msp / (ratio * (ratio - 0.5))
    tau = np.pi * lambda_ / M ** 2
    return Msp, Mr, tau


def nufft_python(x, c, M, df=1.0, eps=1E-15, iflag=1):
    """Fast Non-Uniform Fourier Transform with Python"""
    Msp, Mr, tau = _compute_grid_params(M, eps)
    N = len(x)

    # Construct the convolved grid
    ftau = np.zeros(Mr, dtype=c.dtype)
    Mr = ftau.shape[0]
    hx = 2 * np.pi / Mr
    mm = np.arange(-Msp, Msp)
    for i in range(N):
        xi = (x[i] * df) % (2 * np.pi)
        m = 1 + int(xi // hx)
        spread = np.exp(-0.25 * (xi - hx * (m + mm)) ** 2 / tau)
        ftau[(m + mm) % Mr] += c[i] * spread

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = (1 / Mr) * np.fft.fft(ftau)
    else:
        Ftau = np.fft.ifft(ftau)
    Ftau = np.concatenate([Ftau[-(M // 2):], Ftau[:M // 2 + M % 2]])

    # Deconvolve the grid using convolution theorem
    k = nufftfreqs(M)
    return (1 / N) * np.sqrt(np.pi / tau) * np.exp(tau * k ** 2) * Ftau


def nufft_numpy(x, y, M, df=1.0, iflag=1, eps=1E-15):
    """Fast Non-Uniform Fourier Transform"""
    Msp, Mr, tau = _compute_grid_params(M, eps)
    N = len(x)

    # Construct the convolved grid ftau:
    # this replaces the loop used above
    ftau = np.zeros(Mr, dtype=y.dtype)
    hx = 2 * np.pi / Mr
    xmod = (x * df) % (2 * np.pi)
    m = 1 + (xmod // hx).astype(int)
    mm = np.arange(-Msp, Msp)
    mpmm = m + mm[:, np.newaxis]
    spread = y * np.exp(-0.25 * (xmod - hx * mpmm) ** 2 / tau)
    np.add.at(ftau, mpmm % Mr, spread)

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = (1 / Mr) * np.fft.fft(ftau)
    else:
        Ftau = np.fft.ifft(ftau)
    Ftau = np.concatenate([Ftau[-(M // 2):], Ftau[:M // 2 + M % 2]])

    # Deconvolve the grid using convolution theorem
    k = nufftfreqs(M)
    return (1 / N) * np.sqrt(np.pi / tau) * np.exp(tau * k ** 2) * Ftau


# @numba.jit(nopython=True)
def build_grid(x, c, tau, Msp, ftau):
    Mr = ftau.shape[0]
    hx = 2 * np.pi / Mr
    for i in range(x.shape[0]):
        xi = x[i] % (2 * np.pi)
        m = 1 + int(xi // hx)
        for mm in range(-Msp, Msp):
            ftau[(m + mm) % Mr] += c[i] * np.exp(-0.25 * (xi - hx * (m + mm)) ** 2 / tau)
    return ftau


def nufft_numba(x, c, M, df=1.0, eps=1E-15, iflag=1):
    """Fast Non-Uniform Fourier Transform with Numba"""
    Msp, Mr, tau = _compute_grid_params(M, eps)
    N = len(x)

    # Construct the convolved grid
    ftau = build_grid(x * df, c, tau, Msp,
                      np.zeros(Mr, dtype=c.dtype))

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = (1 / Mr) * np.fft.fft(ftau)
    else:
        Ftau = np.fft.ifft(ftau)
    Ftau = np.concatenate([Ftau[-(M // 2):], Ftau[:M // 2 + M % 2]])

    # Deconvolve the grid using convolution theorem
    k = nufftfreqs(M)
    return (1 / N) * np.sqrt(np.pi / tau) * np.exp(tau * k ** 2) * Ftau


# @numba.jit(nopython=True)
def build_grid_fast(x, c, tau, Msp, ftau, E3):
    Mr = ftau.shape[0]
    hx = 2 * np.pi / Mr

    # precompute some exponents
    for j in range(Msp + 1):
        E3[j] = np.exp(-(np.pi * j / Mr) ** 2 / tau)

    # spread values onto ftau
    for i in range(x.shape[0]):
        xi = x[i] % (2 * np.pi)
        m = 1 + int(xi // hx)
        xi = (xi - hx * m)
        E1 = np.exp(-0.25 * xi ** 2 / tau)
        E2 = np.exp((xi * np.pi) / (Mr * tau))
        E2mm = 1
        for mm in range(Msp):
            ftau[(m + mm) % Mr] += c[i] * E1 * E2mm * E3[mm]
            E2mm *= E2
            ftau[(m - mm - 1) % Mr] += c[i] * E1 / E2mm * E3[mm + 1]
    return ftau


def nufft_numba_fast(x, c, M, df=1.0, eps=1E-15, iflag=1):
    """Fast Non-Uniform Fourier Transform with Numba"""
    Msp, Mr, tau = _compute_grid_params(M, eps)
    N = len(x)

    # Construct the convolved grid
    ftau = build_grid_fast(x * df, c, tau, Msp,
                           np.zeros(Mr, dtype=c.dtype),
                           np.zeros(Msp + 1, dtype=x.dtype))

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = (1 / Mr) * np.fft.fft(ftau)
    else:
        Ftau = np.fft.ifft(ftau)
    Ftau = np.concatenate([Ftau[-(M // 2):], Ftau[:M // 2 + M % 2]])

    # Deconvolve the grid using convolution theorem
    k = nufftfreqs(M)
    return (1 / N) * np.sqrt(np.pi / tau) * np.exp(tau * k ** 2) * Ftau


#######################################################################################################################
# ODE and PDE solvers

# Initialize k vector up to Nyquist wavenumber
def k_space(nx, dx, sort=False):
    """
    Initialize k vector up to Nyquist wavenumber
    :param nx: size of x vector
    :param dx: step length of x vector
    :param sort: flag to return the sorted k vector for nice plotting
    :return: associated k vector
    """
    kmax = np.pi / dx
    dk = kmax / (nx / 2)
    k = np.arange(float(nx))
    k[: int(nx / 2)] = k[: int(nx / 2)] * dk
    k[int(nx / 2):] = k[: int(nx / 2)] - kmax
    if sort:
        k = np.sort(k)
    return k


# Solves pde using fft, my slow algo
def pde_fft_myn_n(f, dx, pow=1, ax=0, filter=None):
    """
    Solve general rank <=2 matrix pde using the Pseudo-spectral FFT method.
    Assumes a uniform grid.

    See for theory:
    https://en.wikipedia.org/wiki/Pseudo-spectral_method
    https://www.coursera.org/lecture/computers-waves-simulations/w5v5-solving-the-1d-2d-wave-equation-with-python-p5Etj
    Optimise me using:
    https://jakevdp.github.io/blog/2015/02/24/optimizing-python-with-numpy-and-numba/

    :param f: input matrix
    :param dx: step length of matrix
    :param pow: power/order of the derivative
    :param ax: axis to perform the derivative
    :param filter: apply a filter to remove high frequency components
    :return: PDE result
    """

    # Determine size and rank
    nx = np.shape(f)[0]
    rank = len(np.shape(f))
    # Assert limits
    assert rank <= 2  # only 1d or 2d arrays
    assert ax == 0 or ax == 1  # only actions on the correct axis
    if rank == 2:
        assert f.shape[0] == f.shape[1], 'Expected matrix to be a square like'
    # Determine k
    k = k_space(nx, dx)
    # Fourier transform
    ff = np.fft.fftn(f)
    # Apply the fft to the other axis
    if ax == 1:
        ff = np.transpose(ff)
    # Apply the derivative in Fourier space
    ff = np.power(1.0j * k, pow) * ff
    if filter == 0:
        # Filter highest freq
        ff[0] = 0
    # Apply the fft to the other axis
    if ax == 1:
        ff = np.transpose(ff)
    # Inverse transform
    df_num = np.fft.ifftn(ff)
    return df_num


# Solves pde using fft, my slow algo
def pde_fft_myn_n_pyfft(f, dx, pow=1, ax=0, filter=None):
    """
    Solve general rank <=2 matrix pde using the Pseudo-spectral FFT method.
    Assumes a uniform grid.

    See for theory:
    https://en.wikipedia.org/wiki/Pseudo-spectral_method
    https://www.coursera.org/lecture/computers-waves-simulations/w5v5-solving-the-1d-2d-wave-equation-with-python-p5Etj
    Optimise me using:
    https://jakevdp.github.io/blog/2015/02/24/optimizing-python-with-numpy-and-numba/

    :param f: input matrix
    :param dx: step length of matrix
    :param pow: power/order of the derivative
    :param ax: axis to perform the derivative
    :param filter: apply a filter to remove high frequency components
    :return: PDE result
    """

    # Determine size and rank
    nx = np.shape(f)[0]
    rank = len(np.shape(f))
    # Assert limits
    assert rank <= 2  # only 1d or 2d arrays
    assert ax == 0 or ax == 1  # only actions on the correct axis
    if rank == 2:
        assert f.shape[0] == f.shape[1], 'Expected matrix to be a square like'

    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
    # Configure PyFFTW to use all cores (the default is single-threaded)
    pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()

    # Determine k
    k = k_space(nx, dx)
    # Fourier transform
    ff = pyfftw.interfaces.numpy_fft.fftn(f)
    # Apply the fft to the other axis
    if ax == 1:
        ff = np.transpose(ff)
    # Apply the derivative in Fourier space
    ff = np.power(1.0j * k, pow) * ff
    if filter == 0:
        # Filter highest freq
        ff[0] = 0
    # Apply the fft to the other axis
    if ax == 1:
        ff = np.transpose(ff)
    # Inverse transform
    df_num = pyfftw.interfaces.numpy_fft.ifftn(ff)
    return df_num


# Solves pde using fft, my slow algo
def pde_fft_myn_n_gpu(f, dx, pow=1, ax=0, filter=None):
    """
    Solve general rank <=2 matrix pde using the Pseudo-spectral FFT method.
    Assumes a uniform grid.

    See for theory:
    https://en.wikipedia.org/wiki/Pseudo-spectral_method
    https://www.coursera.org/lecture/computers-waves-simulations/w5v5-solving-the-1d-2d-wave-equation-with-python-p5Etj
    Optimise me using:
    https://jakevdp.github.io/blog/2015/02/24/optimizing-python-with-numpy-and-numba/

    :param f: input matrix
    :param dx: step length of matrix
    :param pow: power/order of the derivative
    :param ax: axis to perform the derivative
    :param filter: apply a filter to remove high frequency components
    :return: PDE result
    """

    # Determine size and rank
    nx = np.shape(f)[0]
    rank = len(np.shape(f))
    # Assert limits
    assert rank <= 2  # only 1d or 2d arrays
    assert ax == 0 or ax == 1  # only actions on the correct axis
    if rank == 2:
        assert f.shape[0] == f.shape[1], 'Expected matrix to be a square like'
    # Determine k
    k = k_space(nx, dx)

    # Convert to gpu
    k = cp.asarray(k)
    f = cp.asarray(f)

    # Fourier transform
    ff = cp.fft.fftn(f)
    # Apply the fft to the other axis
    if ax == 1:
        ff = cp.transpose(ff)
    # Apply the derivative in Fourier space
    ff = cp.power(1.0j * k, pow) * ff
    if filter == 0:
        # Filter highest freq
        ff[0] = 0
    # Apply the fft to the other axis
    if ax == 1:
        ff = cp.transpose(ff)
    # Inverse transform
    df_num = cp.fft.ifftn(ff)
    return cp.asnumpy(df_num)


# Simple under-the-bonnet 1d fft derivative
def fourier_derivative(f, k, pow):
    # Fourier derivative
    ff = np.power(1.0j * k, pow) * np.fft.fft(f)
    df_num = np.fft.ifft(ff)
    return df_num


# Simple under the bonnet 1d fft derivative
def fourier_derivative_pyfft(f, k, pow):
    # Forward FFT
    ff = pyfftw.interfaces.numpy_fft.fft(f)
    # Fourier derivative
    ff = np.multiply(np.power(np.multiply(1.0j, k), pow), ff)
    df_num = pyfftw.interfaces.numpy_fft.ifft(ff)
    return cp.asnumpy(df_num)


# Simple under the bonnet 1d fft derivative
def fourier_derivative_gpu(f, k, pow):
    # Fourier derivative
    ff = cp.multiply(cp.power(cp.multiply(1.0j, k), pow), cp.fft.fft(f))
    df_num = cp.fft.ifft(ff)
    return cp.asnumpy(df_num)


# Solves pde using fft, direct example from course
def pde_fft_ex_n(f, dx, pow=1, ax=0):
    """
    Solve general rank <=2 matrix pde using the Pseudo-spectral FFT method.
    Assumes a uniform grid.

    See for theory:
    https://en.wikipedia.org/wiki/Pseudo-spectral_method
    https://www.coursera.org/lecture/computers-waves-simulations/w5v5-solving-the-1d-2d-wave-equation-with-python-p5Etj
    Optimise me using:
    https://jakevdp.github.io/blog/2015/02/24/optimizing-python-with-numpy-and-numba/

    :param f: input matrix
    :param dx: step length of matrix
    :param pow: power/order of the derivative
    :param ax: axis to perform the derivative
    :return: PDE result
    """
    # Determine size and rank
    nx = np.shape(f)[0]
    rank = len(np.shape(f))
    # Assert limits
    assert rank <= 2  # only 1d or 2d arrays
    assert ax == 0 or ax == 1  # only actions on the correct axis
    if rank == 2:
        assert f.shape[0] == f.shape[1], 'Expected matrix to be a square like'

    # Determine k space
    k = k_space(len(f), dx)

    if rank == 2:
        df_num = np.zeros((nx, nx), dtype='complex128')  # nx,nz
        if ax == 0:
            for i in np.arange(nx):
                df_num[i, :] = fourier_derivative(f[i, :], k, pow)
        if ax == 1:
            for j in np.arange(nx):
                df_num[:, j] = fourier_derivative(np.transpose(f[:, j]), k, pow)
    else:
        df_num = fourier_derivative(f, k, pow)
    return df_num


# Solves pde using fft, direct example from course
def pde_fft_ex_n_pyfft(f, dx, pow=1, ax=0):
    """
    Solve general rank <=2 matrix pde using the Pseudo-spectral FFT method.
    Assumes a uniform grid.

    See for theory:
    https://en.wikipedia.org/wiki/Pseudo-spectral_method
    https://www.coursera.org/lecture/computers-waves-simulations/w5v5-solving-the-1d-2d-wave-equation-with-python-p5Etj
    Optimise me using:
    https://jakevdp.github.io/blog/2015/02/24/optimizing-python-with-numpy-and-numba/

    :param f: input matrix
    :param dx: step length of matrix
    :param pow: power/order of the derivative
    :param ax: axis to perform the derivative
    :return: PDE result
    """
    # Determine size and rank
    nx = np.shape(f)[0]
    rank = len(np.shape(f))
    # Assert limits
    assert rank <= 2  # only 1d or 2d arrays
    assert ax == 0 or ax == 1  # only actions on the correct axis
    if rank == 2:
        assert f.shape[0] == f.shape[1], 'Expected matrix to be a square like'

    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
    # Configure PyFFTW to use all cores (the default is single-threaded)
    pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()

    # Determine k space
    k = k_space(len(f), dx)
    if rank == 2:
        df_num = np.zeros((nx, nx), dtype='complex128')  # nx,nz
        if ax == 0:
            for i in np.arange(nx):
                df_num[i, :] = fourier_derivative_pyfft(f[i, :], k, pow)
        if ax == 1:
            for j in np.arange(nx):
                df_num[:, j] = fourier_derivative_pyfft(np.transpose(f[:, j]), k, pow)
    else:
        df_num = fourier_derivative_pyfft(f, k, pow)
    return df_num


# Solves pde using fft, direct example from course
def pde_fft_ex_n_gpu(f, dx, pow=1, ax=0):
    """
    Solve general rank <=2 matrix pde using the Pseudo-spectral FFT method.
    Assumes a uniform grid.

    See for theory:
    https://en.wikipedia.org/wiki/Pseudo-spectral_method
    https://www.coursera.org/lecture/computers-waves-simulations/w5v5-solving-the-1d-2d-wave-equation-with-python-p5Etj
    Optimise me using:
    https://jakevdp.github.io/blog/2015/02/24/optimizing-python-with-numpy-and-numba/

    :param f: input matrix
    :param dx: step length of matrix
    :param pow: power/order of the derivative
    :param ax: axis to perform the derivative
    :return: PDE result
    """
    # Determine size and rank
    nx = np.shape(f)[0]
    rank = len(np.shape(f))
    # Assert limits
    assert rank <= 2  # only 1d or 2d arrays
    assert ax == 0 or ax == 1  # only actions on the correct axis
    if rank == 2:
        assert f.shape[0] == f.shape[1], 'Expected matrix to be a square like'

    # Determine k space
    k = k_space(len(f), dx)

    # Convert to gpu
    k = cp.asarray(k)
    f = cp.asarray(f)
    if rank == 2:
        df_num = cp.zeros((nx, nx), dtype='complex128')  # nx,nz
        if ax == 0:
            for i in cp.arange(nx):
                df_num[i, :] = np.asarray(fourier_derivative_gpu(f[i, :], k, pow))
        if ax == 1:
            for j in cp.arange(nx):
                df_num[:, j] = fourier_derivative_gpu(cp.transpose(f[:, j]), k, pow)
    else:
        df_num = fourier_derivative_gpu(f, k, pow)
    return cp.asnumpy(df_num)


# Solves pde using central finite differences
def pde_fine_diff(f, dx, pow=1, ax=0, acc=4):
    """
    Solves pde using central finite differences
    https://github.com/maroba/findiff
    :param f: input matrix
    :param dx: step length of matrix
    :param pow: power/order of the derivative
    :param ax: axis to perform the derivative
    :param acc: accuracy to expand to
    :return: PDE result
    """
    rank = len(np.shape(f))
    """
    if rank == 2:
        # swap the axis
        if ax == 0:
            ax = 1
        elif ax == 1:
            ax = 0
        else:
            exit("Check axis")
    """
    d2_dx2 = FinDiff(ax, dx, pow, acc=acc)
    return d2_dx2(f)


# Helper function to pde_cheb_1d
def get_cheby_matrix(nx):
    """
    Function for setting up the Chebyshev derivative matrix D_ij
    :param nx: Size of input matrix
    :return: derivative matrix
    """
    cx = np.zeros(nx + 1)
    x = np.zeros(nx + 1)
    for ix in range(0, nx + 1):
        x[ix] = np.cos(np.pi * ix / nx)

    cx[0] = 2.
    cx[nx] = 2.
    cx[1:nx] = 1.

    D = np.zeros((nx + 1, nx + 1))
    for i in range(0, nx + 1):
        for j in range(0, nx + 1):
            if i == j and i != 0 and i != nx:
                D[i, i] = -x[i] / (2.0 * (1.0 - x[i] * x[i]))
            else:
                D[i, j] = (cx[i] * (-1.) ** (i + j)) / (cx[j] * (x[i] - x[j]))

    D[0, 0] = (2. * nx ** 2 + 1.) / 6.
    D[nx, nx] = -D[0, 0]
    return D


# Solves pde using Chebyshev polys, direct example from course
def pde_cheb_1d(f):
    """
    Solves pde using Chebyshev polys, direct example from course
    https://www.coursera.org/lecture/computers-waves-simulations/w5v5-solving-the-1d-2d-wave-equation-with-python-p5Etj

    This needs to be tested!
    :param f: input matrix
    :return: PDE result
    """
    # Initialize differentiation matrix
    D = get_cheby_matrix(len(f) - 1)
    # Calculate numerical derivative using differentiation matrix D_{ij}
    return D @ f


#######################################################################################################################
# GAUSSIANS AND BASIS FUNCTIONS

def gaussian_min(x, x0, p0, sigma, h_bar):
    """
    Minimum uncertainty gaussian taken from
    https://socratic.org/questions/why-does-a-gaussian-wave-packet-take-on-the-minimum-value-of-the-heisenberg-unce
    :param x: input vector
    :param x0: inital x
    :param p0: inital p
    :param sigma: spread
    :param h_bar: reduced pc
    :return: psi
    """
    return 1 / (np.cbrt(2 * np.pi * sigma ** 2)) * np.exp(-np.square(x - x0) / (4 * sigma ** 2) + 1.0j * p0 * x / h_bar)


def gaussian_packet_sol(x, a):
    """
    MANUALLY NORMALISE ME
    General gaussian wavepacket
    https://www.nbi.ku.dk/english/theses/bachelor-theses/jon-brogaard/Jon_Brogaard_Bachelorthesis_2015.pdf
    :param x: input vector
    :param a: spread
    :return: psi
    """
    return np.exp(-a * x ** 2)


def gaussian_packet_sol(x, t, a, m, h_bar):
    """
    Gaussian wavepacket solution
    https://www.nbi.ku.dk/english/theses/bachelor-theses/jon-brogaard/Jon_Brogaard_Bachelorthesis_2015.pdf
    :param x: input vector
    :param t: time
    :param a: spread
    :param m: mass
    :param h_bar: reduced pc
    :return: psi
    """
    tau = m / (2 * h_bar * a)  # characterisic time
    return np.cbrt(2 * a / np.pi) * np.exp(np.divide(-a * x ** 2, np.sqrt(1 + (1.0j * t / tau))))


# Single gaussian used for inital conditions
def gaussian_godbeer(x, x0, alpha):
    """
    Gaussian function used by Adam Godbeer in his thesis
    :param x: vector space
    :param x0: x location of peak
    :param alpha: gaussian width
    :return:
    """
    tmp = (alpha / np.pi) ** 0.25 * np.exp(-alpha * (x - x0) ** 2.0)
    return tmp


# gaussian from Wikipedia
def gaussian_wiki(x, mu, sigma):
    """
    Gaussian from wikipedia
    https://en.wikipedia.org/wiki/Gaussian_function
    :param x: vector space
    :param mu: Expectation value
    :param sigma: variance
    :return:
    """
    return 1.0 / (sigma * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2.0)


# Double gaussian used for initial conditions
def gaussian_double(x, x0, alpha):
    """
    Sum of two gaussians, Note this is un-normalised
    :param x:
    :param x0: list of minima
    :param alpha: list of spread
    :return:
    """
    assert len(x0) == len(alpha) == 2
    tmp = gaussian_godbeer(x, x0[0], alpha[0]) + gaussian_godbeer(x, x0[1], alpha[1])
    return tmp


# Simple gaussian
def gaus(x, x0=0.0, v_type=None):
    """
    Simple gaussian
    :param x: x vector
    :param x0: peak location
    :param v_type: variable type, can return sympy expression
    :return:
    """
    if v_type == 'sym':
        rtn = sym.exp(-(x + x0) ** 2) ** 2.0
    else:
        rtn = np.exp(-(x + x0) ** 2) ** 2.0
    return rtn


# Simple 2d gaussian
def gaus_2d(x, y, x0=1.0, y0=1.0, a=1.0, sig_x=0.1, sig_y=0.1, v_type=None):
    """
    Simple 2d gaussian
    :param x: x grid
    :param y: y grid
    :param a: Height
    :param x0: x peak location
    :param y0: y peak location
    :param sig_x: x spread
    :param sig_y: y spread
    :param v_type: variable type, can return sympy expression
    :return:
    """
    if v_type == 'sym':
        rtn = a * sym.exp(-((x - x0) ** 2 / (2 * sig_x) + (y - y0) ** 2 / (2 * sig_y)))
    else:
        rtn = a * np.exp(-((x - x0) ** 2 / (2 * sig_x) + (y - y0) ** 2 / (2 * sig_y)))
    return rtn


#######################################################################################################################
# POTENTIALS AND UTILS

def v_harmonic(t, x, args):
    mass = args[0]
    omega = args[1]
    return 0.5 * mass * np.square(x * omega)


# OLD LST/QST b3lyp
def v_b3lyp_zero(t, x, args):
    """
    Old LST/QST b3lyp which has been rescaled in y to be zero
    :param t: time
    :param x: input vector
    :param args: other args
    :return: p(x)
    """
    tmp = 0.1297 * x ** 5 + 0.4762 * x ** 4 - 0.4058 * x ** 3 - 1.004 * x ** 2 + 0.5622 * x + 0.8135
    tmp5 = np.multiply(0.1297, np.power(x, 5))
    tmp4 = np.multiply(0.4762, np.power(x, 4))
    tmp3 = np.multiply(-0.4058, np.power(x, 3))
    tmp2 = np.multiply(-1.004, np.power(x, 2))
    tmp1 = np.multiply(0.5622, x)
    tmp0 = 0.8135
    val = np.sum([tmp5, tmp4, tmp3, tmp2, tmp1, tmp0])
    return tmp


# Analytical potential of a quantum simple harmonic oscillator
def v_q_anal(t, x, args):
    """
    Analytical potential of a quantum simple harmonic oscillator
    :param t: time
    :param x: input vector
    :param args: other args
    :return: p(x)
    """
    omega = 5e-4
    omega = 1e-3
    m = 938.2720813e6 * 0.01
    a = 1.0
    tmp = 0.5 * m * omega ** 2.0 * (np.abs(x) - a) ** 2.0
    return tmp


# Empty potential
def v_empty(t, x, args):
    """
    Empty potential used for testing and for a free particle
    :param t: time
    :param x: input vector
    :param args: other args
    :return: 0
    """
    return np.multiply(0, x)


def v_quartic(t, x, args):
    """
    General quartic potential
    :param t: time
    :param x: input vector
    :param args: unpack as follows, well minima, forward barrier of the function, asymmetry between p(x) at -a and +a
    :return: p(x)
    """
    a, barrier, asym = args
    zi = x / a
    val = barrier * (a ** 2.0 - zi ** 2.0) ** 2.0 + (asym * zi * 0.5)
    return val - np.min(val)


# 4th order polynomial
def v_4_order_poly(x, a0, a1, a2, a3, a4):
    """
    4th order polynomial
    :param x: x data
    :param a0: x0 coeff
    :param a1: x1 coeff
    :param a2: x2 coeff
    :param a3: x3 coeff
    :param a4: x4 coeff
    :return: p(x)
    """
    return a4 * x ** 4.0 + a3 * x ** 3.0 + a2 * x ** 2.0 + a1 * x + a0


# double quartic
def v_double_quartic(x, a0, a1, a2, a3, a4):
    """
    https://en.wikipedia.org/wiki/Quartic_function
    :param x: x data
    :param a0: x0 coeff
    :param a1: x1 coeff
    :param a2: x2 coeff
    :param a3: x3 coeff
    :param a4: x4 coeff
    :return: p(x)
    """
    # Well Depth
    p = (8.0 * a2 * a4 - 3.0 * a3 ** 2) / (8.0 * a4 ** 2)
    # Asymmetry
    q = (a3 ** 3 - 4 * a2 * a3 * a4 + 8 * a1 * a4 ** 2) / (8 * a4 ** 3)
    # Offset
    r = (-3 * a3 ** 4 + 256 * a0 * a4 ** 3 - 64 * a1 * a3 * a4 ** 2 + 16 * a2 * a3 ** 2 * a4) / (256 * a4 ** 4)
    r = 0
    b = a3 / a4
    y = x + b / 4
    return y ** 4 + p * y ** 2 + q * y + r


# biquadratic
def v_biquadratic(x, barrier, asym, a=1.0):
    """
    https://en.wikipedia.org/wiki/Quartic_function#Biquadratic_equation
    :param x:
    :param barrier: forward barrier of the function
    :param asym: asymmetry between p(x) at -a and +a
    :param a: well minima
    :return: p(x)
    """
    # scale the coordinate
    zi = x / a
    return barrier * (a ** 2.0 - zi ** 2.0) ** 2.0 + (asym / 2.0) * zi


# 2d potential
def v_2d_coupled(x, y, D, G):
    """
    Prototypical 2d potential
    Taken from:
    Entanglement and co-tunneling of two equivalent protons in hydrogen bond pairs
    http://dx.doi.org/10.1063/1.5000681
    :param x: x data
    :param y: y data
    :param D: parameter, sensible values -3 to 2
    :param G: parameter, sensible values 0 to 2
    :return: value, p(x,y)
    """
    assert np.abs(D) < 2 * G
    R = np.divide(3 + D, (1 - D))
    dx = np.sqrt(np.divide((1 + G), (1 - D)))
    dy = np.sqrt(np.divide((1 - G), (1 - D)))
    tmp = (1 - D) * (np.power(np.power(x, 2) - np.power(dx, 2), 2)
                     + np.power(np.power(y, 2) - np.power(dy, 2), 2)
                     + 2.0 * R * np.power(x, 2) * np.power(y, 2))
    return tmp


def v_phi4(t, x, args):
    """
    Taken from
    https://iopscience.iop.org/article/10.1088/0305-4470/31/37/013/pdf

    :param t: time
    :param x: position
    :param args:(a,b) a: asym b: barrier
    :return:
    """
    a = args[0]
    b = args[1]
    rtn = b * 0.25 * np.power(x, 4) - 0.5 * a * np.power(x, 2)
    return rtn


def v_h_bond1(t, x, args):
    """
    Taken from
    https://iopscience.iop.org/article/10.1088/0305-4470/31/37/013/pdf
    :param t: time
    :param x:
    :param args:
    :return:
    """
    v0 = args[0]
    a = args[1]
    alpha = args[2]
    rtn = v0 * (0.5 * a ** 2 * np.cosh(2 * alpha * x) - 2 * a * np.cosh(alpha * x))
    return rtn


def v_h_bond2(t, x, args):
    """
    Taken from
    https://iopscience.iop.org/article/10.1088/0953-8984/8/23/022
    :param t: time
    :param x:
    :param args:
    :return:
    """
    v0 = args[0]  # multplier
    a = args[1]  # asym-like, 0<a, large numbers correspond to a large aym
    b = args[2]  # barrier-like, large numbers result in large barriers, also increases the well seperation
    alpha = args[3]  # width/minima location, 0<a<10 large numbers give closer minima
    rtn = v0 * (0.5 * np.square((1 / b) * np.cosh(2 * alpha * x) - 1) + a * np.cosh(alpha * x))
    return rtn


def v_morse(t, x, args):
    """
    Morse potential
    DOI: 10.1103/PhysRevA.92.042122
    :param t: time
    :param x: postion
    :param args: depth, width, displacement
    :return:
    """
    depth = args[0]
    a = args[1]
    x1 = args[2]
    return depth * (np.exp(-2 * a * (x - x1)) - 2. * np.exp(-a * (x - x1)))


def v_d_morse(t, x, args):
    """
    Double back to back morse potential
    DOI: 10.1103/PhysRevA.92.042122
    :param t: time
    :param x: postion
    :param args: depth, width, displacement
    :return:
    """
    d = args[0]
    a = args[1]
    x1 = args[2]
    m1 = d[0] * (np.exp(-2 * a[0] * (x - x1[0])) - 2. * np.exp(-a[0] * (x - x1[0])))
    m2 = d[1] * (np.exp(-2 * a[1] * (x1[1] - x)) - 2. * np.exp(-a[1] * (x1[1] - x)))
    return m1 + m2


def v_at_rough(t, x, args):
    # unscaled
    if args[0] == 'ns':
        z = [0.00324324, -0.02464209, 0.05403909, -0.02067262, 0.00077699]
    elif args[0] == 's':
        z = [0.02215459, -0.0155033, -0.03819983, 0.0253078, 0.02662674]
    else:
        print('problem with the pot choice')
        exit()
    p = np.poly1d(z)
    return p(x)


def v_box(t, x, args):
    """
    Box potential

    :param t: none
    :param x: input x
    :param args: height of the box
    :return: pot
    """

    if args is None:
        args = [100, 10]
    depth = args[0]
    thick = args[1]

    rtn = np.zeros(len(x))

    rtn[:thick] = depth
    rtn[-thick:] = depth
    return rtn


def v_square(t, x, args):
    """
    |_|-|_|
    :param t: None
    :param x:
    :param args:
    :return:
    """
    if args is None:
        args = [2.0, 4.0, -1]
    v = np.zeros_like(x)
    n = len(v)
    a = args[0]
    b = args[1]
    v0 = args[2]
    v = np.zeros_like(x)
    for i in range(n):
        if x[i] > -a - b / 2. and x[i] < -b / 2.:
            v[i] = v0
        elif x[i] > b / 2. and x[i] < b / 2. + a:
            v[i] = v0
    return v


def v_parabolic(t, x, args):
    v0 = args[0]
    m = args[1]
    omega = args[2]

    v = v0 - 0.5 * m * omega ** 2 * x ** 2
    cnd = np.sqrt((2 * v0) / (m * omega ** 2))
    rtn = np.zeros(len(x))
    for i in range(len(x)):
        if abs(x[i]) <= cnd:
            rtn[i] = v[i]
    return rtn


def v_parabolic_tun_anal(args):
    v0 = args[0]
    e = args[1]
    h_bar = args[2]
    omega = args[3]

    alpha = (2 * np.pi) / (h_bar * omega)
    te = 1 / (1 + np.exp(alpha * (v0 - e)))
    return te


def v_eckart_sym(t, x, args):
    """

    :param t:
    :param x:
    :param args:
    :return:
    """

    v0 = args[0]
    a = args[1]
    return v0 / (np.square(np.cosh(x / a)))


def v_eckart_sym_tun_anal(v0, a, w, m, h_bar):
    alpha = (a / (2.0 * h_bar)) * np.sqrt(2 * w * m)
    # print(alpha)
    ins = ((4.0 * a ** 2.0) / (h_bar ** 2.0)) * 2 * v0 * m - 1.0
    # print(ins)
    delta = 0.5 * np.sqrt(ins)
    # print(delta)
    p = (np.cosh(4 * np.pi * alpha) - 1.0) / (np.cosh(4 * np.pi * alpha) + np.cosh(2 * np.pi * delta))
    return p


def v_eckart_asym(t, x, args):
    """

    :param t:
    :param x:
    :param args:
    :return:
    """
    v1 = args[0]
    v2 = args[1]
    f = args[2]

    a = v1 - v2
    b = np.power(np.power(v1, 0.5) + np.power(v2, 0.5), 2.0)
    x_max = np.log((b + a) / (b - a))
    # print('x max:',x_max)

    l = 2 * np.pi * np.power(-2 / f, 0.5) / (np.power(v1, -0.5) + np.power(v2, -0.5))
    y = - np.exp(2 * np.pi * x / l)
    v = -y * (a - b / (1 - y)) / (1 - y)
    v = -(a * y) / (1.0 - y) - (b * y) / (1.0 - y) ** 2.0
    return v


def v_eckart_asym_j(f_x, f_dv1, f_dv2, f_F):
    # https://sci-hub.tw/https://pubs.acs.org/doi/10.1021/j100809a040
    # https://reactionmechanismgenerator.github.io/RMG-Py/reference/kinetics/eckart.html

    A = f_dv1 - f_dv2
    B = (f_dv1 ** 0.5 + f_dv2 ** 0.5) ** 2.0
    L = 2.0 * np.pi * (-2.0 / f_F) ** 0.5 / (f_dv1 ** -0.5 + f_dv2 ** -0.5)
    y = -np.exp(2.0 * np.pi * f_x / L)
    return -(A * y) / (1.0 - y) - (B * y) / (1.0 - y) ** 2.0


def v_eckart_asym_tun_anal(args):
    """
    A Method of Calculating Tunneling Corrections For Eckart Potential Barriers

    https://nvlpubs.nist.gov/nistpubs/jres/086/jresv86n4p357_A1b.pdf

    :return:
    """
    h_bar = args[0]  # reduced planks constant
    m = args[1]  # effective mass
    k_b = args[2]  # Boltzmann constant
    t = args[3]  # Temperature
    e = args[4]  # Incident energy [LIST]
    pot1 = args[5]  # v1
    pot2 = args[6]  # v2
    f = args[7]  # second derive / width param

    # Conversion to keep true to the equations
    h = h_bar * (2 * np.pi)

    v = np.sqrt(-f / m) / (2 * np.pi)
    # print("Imaginary freq: ", v)
    u = (h * v) / (k_b * t)

    alpha1 = (2 * np.pi * pot1) / (h * v)
    alpha2 = (2 * np.pi * pot2) / (h * v)

    eta = (e - pot1) / (k_b * t)
    v1 = pot1 / (k_b * t)
    v2 = pot2 / (k_b * t)

    d = np.sqrt(4.0 * alpha1 * alpha2 - np.pi ** 2) / (2 * np.pi)
    # print(d)

    if type(d) == complex:
        # print('value complex')
        d_big = np.cos(2 * np.pi * np.absolute(d))
    else:
        # print('value not complex')
        d_big = np.cosh(2 * np.pi * d)

    c = (1.0 / 8.0) * np.pi * u * np.power(np.power(alpha1, -0.5) + np.power(alpha2, -0.5), 2)
    a1 = 0.5 * np.sqrt((eta + v1) / c)
    a2 = 0.5 * np.sqrt((eta + v2) / c)

    k = (np.cosh(2 * np.pi * (a1 + a2)) - np.cosh(2 * np.pi * (a1 - a2))) / (np.cosh(2 * np.pi * (a1 + a2)) + d_big)
    return k


def v_eckart_asym_new(x, v1, v2, f, x0):
    """
    https://reactionmechanismgenerator.github.io/RMG-Py/reference/kinetics/eckart.html
    https://sci-hub.tw/https://pubs.acs.org/doi/10.1021/j100809a040
    :param x:
    :param v1:
    :param v2:
    :param f:
    :param x0:
    :return:
    """
    x = np.subtract(x, x0)

    a = v1 - v2
    b = np.power(np.power(v1, 0.5) + np.power(v2, 0.5), 2.0)
    # x_max = np.log((b+a)/(b-a))
    # print('x max:',x_max)

    l = 2 * np.pi * np.power(-2 / f, 0.5) / (np.power(v1, -0.5) + np.power(v2, -0.5))
    y = - np.exp(2 * np.pi * x / l)
    # v = -y * (a - b / (1 - y)) / (1 - y)

    # https://sci-hub.tw/https://pubs.acs.org/doi/10.1021/j100809a040
    v = -(a * y) / (1.0 - y) - (b * y) / (1.0 - y) ** 2.0
    return v


def qm_rate_int(x, beta, trans_func, args):
    return trans_func(*args) * np.exp(-beta * x)


def qm_rate(args):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html#scipy.integrate.quad
    :param args:
    :return:
    """
    k_b = args[0]
    temp = args[1]
    v_max = args[2]
    beta = 1 / (k_b * temp)
    func = None
    func_args = None
    val, _ = scipy.integrate.quad(qm_rate_int, 0, np.inf, args=(beta, func, func_args))
    return beta * np.exp(beta * v_max) * val


# Scales and translates
def scale_trans(x, min=None, max=None, a=-1.0, b=1.0):
    """
    Linearly maps a vector from [min,max] to [a,b]
    https://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio
    :param x: input vector
    :param min: minimum of vector
    :param max: maximum of vector
    :param a: small value to map to
    :param b: large value to map to
    :return: scaled vector
    """
    if min == None:
        min = np.min(x)
    if max == None:
        max = np.max(x)
    return (((b - a) * (x - min)) / (max - min)) + a


#######################################################################################################################
# OTHER
# Terminal shell command caller
def terminal_call(coms, f_print=True):
    """
    Sends a command to command line
    :param coms:
    :param f_print:
    :return:
    """
    if f_print == True:
        print('Sending: ', coms + '\n')
    subprocess.call([coms])
    return None


# Asks the os the user
def get_user():
    """
    Asks the os who the user is
    :return: string of the user
    """
    return os.popen('whoami').read()


# Helper to natural_keys
def atoi(text):
    """
    helper function of natural_keys
    :param text: input text
    :return: ??
    """
    return int(text) if text.isdigit() else text


# Human sorts a list of strings
def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    :param text:
    :return: sorted list
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


# finds the location of minima in a 2D array
def detect_local_minima(arr):
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when the pixel's value is the neighborhood maximum, 0 otherwise)
    Taken from:
    https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    :param arr: input array
    :return: location of the detected minima
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
    # apply the local minimum filter; all locations of minimum value
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood) == arr)
    # local_min is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    #
    # we create the mask of the background
    background = (arr == 0)
    #
    # a little technicality: we must erode the background in order to
    # successfully subtract it from local_min, otherwise a line will
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    #
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_min mask
    detected_minima = np.subtract(local_min.astype(np.float32), eroded_background.astype(np.float32))
    return np.where(detected_minima)


# finds the location of maxima
def detect_maxima_loc(yy, verbosity=0):
    """
    For a given vector finds a local maxima, hence a well barrier
    :param yy: input vector, for example the potential
    :param verbosity: printing control
    :return: location of the maxima
    """
    # tmp = scipy.signal.argrelextrema(yy, np.greater)[0]
    tmp = find_peaks(yy)[0]
    if np.sum(yy) == 0:
        if verbosity >= 2:
            print('System contains a free particle!')
        tmp = 0
    else:
        if len(tmp) != 1:
            print('Problem finding single barrier')
            print('Found n maxima:', len(tmp))
            tmp = 0
        else:
            tmp = tmp[0]
            if verbosity >= 2:
                print('barrier found at loc ', tmp, yy[tmp])
    return tmp


# Finds the location of the minima
def detect_minima_loc(yy, verbosity=0):
    """
    For a given vector finds a local minima, hence a well minima
    :param yy: input vector, for example the potential
    :param verbosity: printing control
    :return: location of the minima
    """
    tmp = scipy.signal.argrelextrema(yy, np.less)[0]
    if np.sum(yy) == 0:
        if verbosity >= 2:
            print('System contains a free particle!')
        tmp = 0
    else:
        if len(tmp) != 2:
            if verbosity >= 1:
                print('Problem finding double minima')
                print('Found n maxima:', len(tmp))
            exit()
        else:
            if verbosity >= 2:
                print('minima found at locs, with vals ', tmp, yy[tmp])
    return tmp


# Convert an element name to an atomic number
def ele_name_2_num(ele):
    """
    Elements from:
    http://www.elementalmatter.info/number-of-electrons.htm

    grab the elements with
    import numpy as np
    import os
    path = os.path.join(r'C:/Users/ls00338/Google Drive/PhD', 'ele.csv')
    ele = np.loadtxt(path, delimiter=',', dtype=str, usecols=(1))
    num = np.loadtxt(path, delimiter=',', dtype=str, usecols=(2))
    for i in range(len(ele)):
        print('"' + ele[i] + '":' + str(num[i])[:-1]+',')

    :param ele: input elemental name
    :return: atomic number
    """

    # Dictionary of all the elements
    dic = {"H": 1,
           "He": 2,
           "Li": 3,
           "Be": 4,
           "B": 5,
           "C": 6,
           "N": 7,
           "O": 8,
           "F": 9,
           "Ne": 10,
           "Na": 11,
           "Mg": 12,
           "Al": 13,
           "Si": 14,
           "P": 15,
           "S": 16,
           "Cl": 17,
           "Ar": 18,
           "K": 19,
           "Ca": 20,
           "Sc": 21,
           "Ti": 22,
           "V": 23,
           "Cr": 24,
           "Mn": 25,
           "Fe": 26,
           "Co": 27,
           "Ni": 28,
           "Cu": 29,
           "Zn": 30,
           "Ga": 31,
           "Ge": 32,
           "As": 33,
           "Se": 34,
           "Br": 35,
           "Kr": 36,
           "Rb": 37,
           "Sr": 38,
           "Y": 39,
           "Zr": 40,
           "Nb": 41,
           "Mo": 42,
           "Tc": 43,
           "Ru": 44,
           "Rh": 45,
           "Pd": 46,
           "Ag": 47,
           "Cd": 48,
           "In": 49,
           "Sn": 50,
           "Sb": 51,
           "Te": 52,
           "I": 53,
           "Xe": 54,
           "Cs": 55,
           "Ba": 56,
           "La": 57,
           "Ce": 58,
           "Pr": 59,
           "Nd": 60,
           "Pm": 61,
           "Sm": 62,
           "Eu": 63,
           "Gd": 64,
           "Tb": 65,
           "Dy": 66,
           "Ho": 67,
           "Er": 68,
           "Tm": 69,
           "Yb": 70,
           "Lu": 71,
           "Hf": 72,
           "Ta": 73,
           "W": 74,
           "Re": 75,
           "Os": 76,
           "Ir": 77,
           "Pt": 78,
           "Au": 79,
           "Hg": 80,
           "Tl": 81,
           "Pb": 82,
           "Bi": 83,
           "Po": 84,
           "At": 85,
           "Rn": 86,
           "Fr": 87,
           "Ra": 88,
           "Ac": 89,
           "Th": 90,
           "Pa": 91,
           "U": 92,
           "Np": 93,
           "Pu": 94,
           "Am": 95,
           "Cm": 96,
           "Bk": 97,
           "Cf": 98,
           "Es": 99,
           "Fm": 100,
           "Md": 101,
           "No": 102,
           "Lr": 103,
           "Rf": 104,
           "Db": 105,
           "Sg": 106,
           "Bh": 107,
           "Hs": 108,
           "Mt": 109,
           "Ds": 110,
           "Rg": 111,
           "Uub": 112,
           "Uut": 113,
           "Uuq": 114,
           "Uup": 115,
           "Uuh": 116,
           "Uus": 117,
           "Uuo": 118}
    # Check if the input is a list
    if isinstance(ele, (list,)):
        rtn = np.array([dic[i] for i in ele])
    # Check if the input is an array
    elif isinstance(ele, (np.ndarray,)):
        rtn = np.array([dic[i] for i in ele])
    # Sort otherwise
    else:
        rtn = dic[ele]
    return rtn


def find_nearest(array, value):
    """
    Finds the nearest value
    https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    :param array:
    :param value:
    :return:
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


# Round to significant digits
def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


def is_odd(num):
    return num & 0x1
