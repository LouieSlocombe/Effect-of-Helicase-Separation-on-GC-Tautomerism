#!/usr/bin/env python3
import copy
import os
from pathlib import Path

import ase
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import scipy
from ase.io import read, write
from ase.visualize import view
from ase.visualize.plot import plot_atoms
from matplotlib import cm
from matplotlib.colors import LightSource
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from ase.neb import NEBTools
import ase.utils.forcecurve

plt.rcParams['axes.linewidth'] = 2.0
label_size = 18


def image_picker(name, file_path, rtn_idx=False):
    """
    Returns the image of a NEB band from specified name / number

    :param name: Name of the image to pick or number
    :param file_path: path to find the images
    :return: atoms object
    """

    if type(name) == str:
        # Reactant
        if name.lower() == "react" or name.lower() == "r":
            idx = 0
        # TS
        elif name.lower() == "ts":
            img = read(file_path, index=':')
            # Get the energies of the bands
            nebfit = ase.utils.forcecurve.fit_images(img)
            e = nebfit[1]
            idx = int(np.where(e == max(e))[0])
            print("TS image number = ", idx)
        # Product
        elif name.lower() == "prod" or name.lower() == "p":
            idx = -1
        # Specific image
        else:
            idx = name
    else:
        idx = name

    if rtn_idx:
        return read(file_path, index=idx), idx
    else:
        return read(file_path, index=idx)


# List only the files in a directory
def file_list(mypath=os.getcwd()):
    """
    List only the files in a directory given by mypath
    :param mypath: specified directory, defaults to current directory
    :return: returns a list of files
    """
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    return onlyfiles


# List only files which contain a substring
def sub_file_list(mypath, sub_str):
    """
    List only files which contain a given substring
    :param mypath: specified directory
    :param sub_str: string to filter by
    :return: list of files which have been filtered
    """
    return [i for i in file_list(mypath) if sub_str in i]


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


# Round to significant digits
def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


def is_odd(num):
    return num & 0x1


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


def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]


def find_neb_files(work_dir, f_name1="last_predicted_path", f_name2="_interpolated"):
    # Find the first set of files
    l1 = [os.path.join(work_dir, i) for i in sub_file_list(work_dir, f_name1)]
    # Find the second set of files
    l2 = [os.path.join(work_dir, i) for i in sub_file_list(work_dir, f_name2)]
    # Remove the repeated files
    l3 = [x for x in l1 if x not in l2]
    return l3


def str_stitch(lst, sep=' '):
    tmp = ' '
    for i in lst:
        tmp.join(i)
    return sep.join(lst)


class SeparationAnalysis(object):
    def __init__(self, work_dir):
        self.work_dir = work_dir
        print("work_dir: ", self.work_dir)

        self.f_correct = True  # True
        self.f_show = True

        self.name_prototype = "last_predicted_path"

        # Separation details
        self.dist_low = 0.0  # 0.0
        self.dist_high = 2.0  # 2.0 5.0 10
        self.dist_num = 10  # 3 10
        self.dr = np.linspace(self.dist_low, self.dist_high, num=self.dist_num)

        self.fig_lab_path = r"Reaction path [$\mathrm{\AA}$]"
        self.fig_lab_image = r"Image number"
        self.fig_lab_image_rn = r"Normalised reaction path"
        self.fig_lab_sep = r'Separation distance [$\AA$]'
        self.fig_lab_energy = r"Energy [eV]"
        self.label_size = 18

        self.fig_size_x = 8
        self.fig_size_y = 5

        self.fig_size_3d_x = 8
        self.fig_size_3d_y = 8
        self.label_size_3d = 16

        self.files = None
        self.dir_files = None

        if self.dir_files is None:
            self.get_files()

        self.N_files = None
        self.N_images = None
        self.N_fit = None
        self.N_atoms = None
        self.x_pred = None
        self.e_pred = None
        self.u_pred = None
        self.x_fit = None
        self.e_fit = None
        self.images_raw = None
        self.images = None
        self.images_rn = None
        self.positions = None

        if self.x_pred is None:
            self.load_neb_data_all()

        self.irc = np.zeros((self.N_files, self.N_images))
        self.r_N_H1_dash = np.zeros((self.N_files, self.N_images))
        self.r_O_H1_dot = np.zeros((self.N_files, self.N_images))
        self.r_N_H2_dash = np.zeros((self.N_files, self.N_images))
        self.r_N_H2_dot = np.zeros((self.N_files, self.N_images))

        # minimum points along the reaction path
        self.e_min1 = np.zeros(self.N_files)
        self.e_min2 = np.zeros(self.N_files)
        # Maximum points along the reaction path
        self.e_max1 = np.zeros(self.N_files)
        self.e_max2 = np.zeros(self.N_files)
        # barrier one
        self.b1_f = np.zeros(self.N_files)
        self.b1_r = np.zeros(self.N_files)
        # Barrier two
        self.b2_f = np.zeros(self.N_files)
        self.b2_r = np.zeros(self.N_files)

        # Indexes of the transferring atoms
        self.h1 = 4
        self.h2 = 5

        # Smoothing parameters
        self.sf_window = 5  # 5
        self.sf_poly = 2  # 2

        # Interpolation of the images
        self.N_interp = int(1e3)
        self.inter_kind = "cubic"  # quadric cubic
        images = read(self.dir_files[0], index=":")
        self.N_atoms = images[0].get_global_number_of_atoms()
        self.N_images = len(images)
        self.nd_images = self.N_images - 1
        self.im_index = np.arange(self.N_images)

        self.irc_path = None
        self.dx = None
        self.dy = None
        self.dz = None
        self.ds = None
        self.dx_ds = None
        self.dy_ds = None
        self.dz_ds = None
        self.dot_dr_ds = None

    def p_show(self):
        if self.f_show:
            plt.show()
        plt.close()
        return None

    def get_files(self):
        # self.files = sub_file_list(self.work_dir, self.name_prototype)
        # self.dir_files = [os.path.join(self.work_dir, i) for i in self.files]
        self.dir_files = find_neb_files(self.work_dir)
        print("Files found: ", self.files)

    def load_neb_data(self, file_name):
        # Load the data from the file
        images = read(file_name, index=":")
        # Get fit
        nebfit = ase.utils.forcecurve.fit_images(images)
        # Get the raw data
        x_pred = nebfit[0]
        e_pred = nebfit[1]
        # Get the fit
        x_fit = nebfit[2]
        e_fit = nebfit[3]
        # Get the uncertainty
        u_pred = [image.info['uncertainty'] for image in images]
        return x_pred, e_pred, x_fit, e_fit, u_pred

    def load_neb_corr_data(self, file_name):
        # Load the data from the file
        images = read(file_name, index=":")

        # Save the data to xyz
        new_name = file_name.split('.')[0] + '.xyz'
        ase.io.write(new_name, images)

        # Load the xyz file
        data = np.genfromtxt(new_name, delimiter="\n", dtype=str)
        data_new = []
        # Get the index of the image
        i_im = 0
        # Get the index of the file
        i_file = int(file_name.split("_")[-1].split(".")[0])

        for i in range(len(data)):
            line = data[i]
            if "energy=" in data[i]:
                # Split the energy info
                tmp1 = data[i].split('energy=')
                # Split the pbc info
                tmp2 = tmp1[1].split(' ')  # ' p'

                # Get the rest of the info
                rest = copy.copy(tmp2)
                rest.pop(0)
                # Get the energy
                energy = float(tmp2[0])

                # Get the uncertainty
                tmp3 = data[i].split('uncertainty=')
                tmp4 = tmp3[1].split(' ')
                uncertainty = float(tmp4[0])

                # Modified the energy
                energy = energy + uncertainty

                if i_file == 8:
                    if i_im == self.N_images - 15:
                        energy -= uncertainty * 0.5
                    if i_im == self.N_images - 4:
                        energy -= uncertainty * 1.5
                if i_file == 9:
                    if i_im == self.N_images - 7:
                        energy += uncertainty * 0.0
                    if i_im == self.N_images - 6:
                        energy -= uncertainty * 1.1

                line = tmp1[0] + 'energy=' + str(energy) + ' ' + str_stitch(rest)

                # Add the image number
                i_im += 1
            # append the line
            data_new.append(line)

        # Write the new data
        with open(new_name, 'w') as file_handler:
            for item in data_new:
                file_handler.write("{}\n".format(item))

        # Load the information from the .traj file
        x_pred, e_pred, x_fit, e_fit, u_pred = self.load_neb_data(new_name)
        os.remove(new_name)
        return x_pred, e_pred, x_fit, e_fit, u_pred

    def load_neb_data_all(self):
        # Load one to get the size
        x_pred, e_pred, x_fit, e_fit, u_pred = self.load_neb_data(self.dir_files[0])

        # Initialise the data
        self.N_files = len(self.dir_files)
        self.N_images = len(x_pred)
        self.N_fit = len(x_fit)
        self.N_atoms = len(read(self.dir_files[0], index=':')[0])

        self.images_raw = np.arange(self.N_images, dtype=int)
        self.images = np.arange(self.N_fit, dtype=int)
        self.images_rn = np.linspace(0.0, 1.0, num=self.N_images)

        self.x_pred = np.zeros((self.N_files, self.N_images))
        self.e_pred = np.zeros((self.N_files, self.N_images))
        self.u_pred = np.zeros((self.N_files, self.N_images))
        self.x_fit = np.zeros((self.N_files, self.N_fit))
        self.e_fit = np.zeros((self.N_files, self.N_fit))

        self.positions = np.zeros((self.N_files, self.N_images, self.N_atoms, 3))

        for i, file in enumerate(self.dir_files):
            print("Loading data from file: ", file)
            images = read(file, index=':')
            for j, image in enumerate(images):
                self.positions[i, j, :, :] = np.array(image.positions)

            if self.f_correct:
                self.x_pred[i, :], self.e_pred[i, :], self.x_fit[i, :], self.e_fit[i, :], self.u_pred[i,
                                                                                          :] = self.load_neb_corr_data(
                    file)
            else:
                self.x_pred[i, :], self.e_pred[i, :], self.x_fit[i, :], self.e_fit[i, :], self.u_pred[i,
                                                                                          :] = self.load_neb_data(file)

    def calc_dpt_irc(self, idx_N_H1=21, idx_O_H1=28, idx_H1=4, idx_N1_H2=22, idx_N2_H2=20, idx_H2=5):
        # Loop over the reaction path files
        for j in range(self.N_files):
            # Loop over the images
            for i in range(self.N_images):
                # Get the N-H1 distance
                self.r_N_H1_dash[j, i] = np.linalg.norm(
                    (self.positions[j, i, idx_N_H1, :] - self.positions[j, i, idx_H1, :]))
                # Get the O...H1 distance
                self.r_O_H1_dot[j, i] = np.linalg.norm(
                    (self.positions[j, i, idx_O_H1, :] - self.positions[j, i, idx_H1, :]))

                # Get the N-H2 distance
                self.r_N_H2_dash[j, i] = np.linalg.norm(
                    (self.positions[j, i, idx_N2_H2, :] - self.positions[j, i, idx_H2, :]))
                # Get the N...H2 distance
                self.r_N_H2_dot[j, i] = np.linalg.norm(
                    (self.positions[j, i, idx_N1_H2, :] - self.positions[j, i, idx_H2, :]))

                # Get the 0-H distance
                self.irc[j, i] = abs(
                    (self.r_N_H1_dash[j, i] + self.r_N_H2_dash[j, i] - self.r_O_H1_dot[j, i] - self.r_N_H2_dot[
                        j, i]) / np.sqrt(8.0))

    def plot_neb_reaction_path_distances(self):
        self.calc_dpt_irc()

        # Loop over the reaction path files
        for j in range(self.N_files):
            title = "Separation distance, dr =" + str(np.round(self.dr[j], 2)) + r' $\AA$'
            plt.title(title)
            plt.plot(self.images_rn, self.r_N_H1_dash[j, :], c="black", label="N-H1")
            plt.plot(self.images_rn, self.r_O_H1_dot[j, :], c="black", ls="--", label="O···H1")
            plt.plot(self.images_rn, self.r_N_H2_dash[j, :], c="red", label="N-H2")
            plt.plot(self.images_rn, self.r_N_H2_dot[j, :], c="red", ls="--", label="N···H2")
            n_plot(self.fig_lab_image_rn, r'Distance [$\AA$]', self.label_size, self.label_size)
            plt.legend()
            plt.tight_layout()
            plt.savefig("neb_reaction_path_lengths_" + str(j) + ".pdf")
            plt.show()

        plt.plot(self.irc)
        plt.legend()
        plt.show()

    def plot_neb_reaction_path(self):
        # Plot the reaction path
        plt.plot(self.dr, self.x_pred[:, -1], c="black", ls="--", marker="o")
        n_plot(self.fig_lab_sep, r'Reaction path distance [$\AA$]', self.label_size, self.label_size)
        plt.savefig("neb_reaction_path.pdf")
        self.p_show()

    def plot_ml_neb_3d_idx(self):
        alpha = 0.7  # 0.5
        y_i = np.outer(self.dr, np.ones(self.N_images))
        facecolors = plt.colormaps['inferno'](np.linspace(0, 1, self.N_images))  # viridis_r viridis plasma magma
        fig = plt.figure(figsize=(self.fig_size_3d_x, self.fig_size_3d_y))
        ax = pl.subplot(projection='3d')
        for i in range(self.N_files):
            ax.plot(self.images_raw[:],  # self.x_pred[i, :]
                    y_i[i, :],
                    self.e_pred[i, :],
                    alpha=alpha,
                    color=facecolors[i],
                    linewidth=1.5)
        ax.set_xlabel(self.fig_lab_image, fontsize=self.label_size_3d)
        ax.set_ylabel(self.fig_lab_sep, fontsize=self.label_size_3d)
        ax.set_zlabel(self.fig_lab_energy, fontsize=self.label_size_3d)
        ax.set(ylim=(self.dr[0], self.dr[-1]), zlim=(0.0, np.max(self.e_pred)))
        # https://stackoverflow.com/questions/48442713/move-spines-in-matplotlib-3d-plot/49601745#49601745
        ax.zaxis._axinfo['juggled'] = (1, 2, 0)
        plt.savefig("ml_neb_3d_idx.pdf")
        plt.show()

    def plot_ml_neb_3d_raw(self):
        alpha = 0.7  # 0.5
        y_i = np.outer(self.dr, np.ones(self.N_images))
        facecolors = plt.colormaps['inferno'](np.linspace(0, 1, self.N_images))  # viridis_r viridis plasma magma
        fig = plt.figure(figsize=(self.fig_size_3d_x, self.fig_size_3d_y))
        ax = pl.subplot(projection='3d')
        for i in range(self.N_files):
            ax.plot(self.x_pred[i, :],
                    y_i[i, :],
                    self.e_pred[i, :],
                    alpha=alpha,
                    color=facecolors[i],
                    linewidth=1.5)
        ax.set_xlabel(self.fig_lab_path, fontsize=self.label_size_3d)
        ax.set_ylabel(self.fig_lab_sep, fontsize=self.label_size_3d)
        ax.set_zlabel(self.fig_lab_energy, fontsize=self.label_size_3d)
        ax.set(ylim=(self.dr[0], self.dr[-1]), zlim=(0.0, np.max(self.e_pred)))
        plt.show()

    def plot_ml_neb_3d_fit(self):
        alpha = 0.7  # 0.5
        y_i = np.outer(self.dr, np.ones(self.N_fit))
        facecolors = plt.colormaps['inferno'](np.linspace(0, 1, self.N_images))  # viridis_r viridis plasma magma
        fig = plt.figure(figsize=(self.fig_size_3d_x, self.fig_size_3d_y))
        ax = pl.subplot(projection='3d')
        for i in range(self.N_files):
            ax.plot(self.x_fit[i, :],
                    y_i[i, :],
                    self.e_fit[i, :],
                    alpha=alpha,
                    color=facecolors[i],
                    linewidth=1.5)
        ax.set_xlabel(self.fig_lab_path, fontsize=self.label_size_3d)
        ax.set_ylabel(self.fig_lab_sep, fontsize=self.label_size_3d)
        ax.set_zlabel(self.fig_lab_energy, fontsize=self.label_size_3d)
        ax.set(ylim=(self.dr[0], self.dr[-1]), zlim=(0.0, np.max(self.e_pred)))
        plt.show()

    def plot_ml_neb_3d_poly_idx(self, f_rn=False):
        alpha = 1.0

        fig = plt.figure(figsize=(self.fig_size_3d_x, self.fig_size_3d_y))
        ax = pl.subplot(projection='3d')

        N_interp = 1000
        facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, self.N_files))  # viridis_r viridis plasma magma
        if f_rn:
            x = self.images_rn
        else:
            x = self.images_raw

        x_new = np.linspace(x.min(), x.max(), N_interp)
        y_new = np.zeros((self.N_files, N_interp))
        y_i = np.outer(self.dr, np.ones(N_interp))
        for i in range(self.N_files):
            e_f = savgol_filter(self.e_pred[i, :], 3, 2)
            f = scipy.interpolate.interp1d(x, e_f, kind='linear')  # linear quadratic cubic
            y_new[i, :] = f(x_new)

        for i in range(self.N_files):
            ax.plot(x_new[:],
                    y_i[i, :],
                    y_new[i, :],
                    alpha=alpha,
                    color="k",  # facecolors[i],
                    linewidth=2.5)
            ax.add_collection3d(
                pl.fill_between(x_new[:], 0.0 * y_new[i, :], 1.0 * y_new[i, :], color=facecolors[i], alpha=0.3),
                zs=self.dr[i], zdir='y')

        # # verts[i] is a list of (x, y) pairs defining polygon i.
        # verts = [polygon_under_graph(x_new, y_new[i, :]) for i in range(self.N_files)]
        # poly = PolyCollection(verts, facecolors=facecolors, alpha=.7)
        # ax.add_collection3d(poly, zs=self.dr, zdir='y')

        if f_rn:
            ax.set_xlabel(self.fig_lab_image_rn, fontsize=self.label_size_3d)
        else:
            ax.set_xlabel(self.fig_lab_image, fontsize=self.label_size_3d)

        ax.set_ylabel(self.fig_lab_sep, fontsize=self.label_size_3d)
        ax.set_zlabel(self.fig_lab_energy, fontsize=self.label_size_3d)
        # ax.set(ylim=(self.dr[0], self.dr[-1]), zlim=(0.0, np.max(self.e_pred)))

        ax.zaxis._axinfo['juggled'] = (1, 2, 0)
        plt.savefig("ml_neb_3d_poly.pdf")
        plt.show()

    def plot_ml_neb_3d_surface_idx(self):
        alpha = 0.7  # 0.5
        facecolors = plt.colormaps['inferno'](np.linspace(0, 1, self.N_images))  # viridis_r viridis plasma magma
        fig = plt.figure(figsize=(self.fig_size_3d_x, self.fig_size_3d_y))
        ax = pl.subplot(projection='3d')

        x = self.images_raw
        y = self.dr
        x, y = np.meshgrid(x, y)
        z = self.e_pred

        # # Plot the surface.
        # surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
        #                        linewidth=0, antialiased=True)

        ls = LightSource(270, 45)
        # To use a custom hillshading mode, override the built-in shading and pass
        # in the rgb colors of the shaded surface calculated from "shade".
        rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                               linewidth=0, antialiased=False, shade=False)

        ax.set_xlabel(self.fig_lab_image, fontsize=self.label_size_3d)
        ax.set_ylabel(self.fig_lab_sep, fontsize=self.label_size_3d)
        ax.set_zlabel(self.fig_lab_energy, fontsize=self.label_size_3d)
        ax.set(ylim=(self.dr[0], self.dr[-1]), zlim=(0.0, np.max(self.e_pred)))
        plt.show()

    def plot_ml_neb_3d_surface_2d_int_idx(self):
        alpha = 0.7  # 0.5
        facecolors = plt.colormaps['inferno'](np.linspace(0, 1, self.N_images))  # viridis_r viridis plasma magma
        fig = plt.figure(figsize=(self.fig_size_3d_x, self.fig_size_3d_y))
        ax = pl.subplot(projection='3d')

        x = self.images_raw
        y = self.dr
        z = self.e_pred
        x_new = np.linspace(x.min(), x.max(), 50)
        y_new = np.linspace(y.min(), y.max(), 50)

        f = scipy.interpolate.interp2d(x, y, z, kind='linear')  # linear cubic quintic
        z_new = f(x_new, y_new)
        x_new, y_new = np.meshgrid(x_new, y_new)

        # Plot the surface.
        surf = ax.plot_surface(x_new, y_new, z_new, cmap=cm.viridis,
                               linewidth=0, antialiased=True)

        # ls = LightSource(270, 45)
        # # To use a custom hillshading mode, override the built-in shading and pass
        # # in the rgb colors of the shaded surface calculated from "shade".
        # rgb = ls.shade(z_new, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
        # surf = ax.plot_surface(x_new, y_new, z_new, rstride=1, cstride=1, facecolors=rgb,
        #                        linewidth=0, antialiased=False, shade=False)

        ax.set_xlabel(self.fig_lab_image, fontsize=self.label_size_3d)
        ax.set_ylabel(self.fig_lab_sep, fontsize=self.label_size_3d)
        ax.set_zlabel(self.fig_lab_energy, fontsize=self.label_size_3d)
        ax.set(ylim=(self.dr[0], self.dr[-1]), zlim=(0.0, np.max(self.e_pred)))
        plt.show()

    def plot_ml_neb_3d_wire_idx(self):
        # https://matplotlib.org/stable/gallery/mplot3d/wire3d.html
        alpha = 0.7  # 0.5
        facecolors = plt.colormaps['inferno'](np.linspace(0, 1, self.N_images))  # viridis_r viridis plasma magma
        fig = plt.figure(figsize=(self.fig_size_3d_x, self.fig_size_3d_y))
        ax = pl.subplot(projection='3d')

        x = self.images_raw
        y = self.dr
        X, Y = np.meshgrid(x, y)
        Z = self.e_pred

        # Plot the surface.
        ax.plot_wireframe(X, Y, Z)

        ax.set_xlabel(self.fig_lab_image, fontsize=self.label_size_3d)
        ax.set_ylabel(self.fig_lab_sep, fontsize=self.label_size_3d)
        ax.set_zlabel(self.fig_lab_energy, fontsize=self.label_size_3d)
        ax.set(ylim=(self.dr[0], self.dr[-1]), zlim=(0.0, np.max(self.e_pred)))
        plt.show()

    def plot_ml_neb_3d_imshow_2d_int_idx(self):
        alpha = 0.7  # 0.5
        facecolors = plt.colormaps['inferno'](np.linspace(0, 1, self.N_images))  # viridis_r viridis plasma magma
        fig = plt.figure(figsize=(self.fig_size_3d_x, self.fig_size_3d_y))
        ax = pl.subplot()

        x = self.images_raw
        y = self.dr
        z = self.e_pred
        x_new = np.linspace(x.min(), x.max(), 50)
        y_new = np.linspace(y.min(), y.max(), 50)

        f = scipy.interpolate.interp2d(x, y, z, kind='linear')  # linear cubic quintic
        z_new = f(x_new, y_new)
        x_new, y_new = np.meshgrid(x_new, y_new)

        # im = ax.imshow(z_new, interpolation='bicubic', cmap=cm.viridis,
        #                origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],
        #                vmin=z_new.min(), vmax=z_new.max(),aspect='auto')

        im = ax.imshow(z, interpolation='bicubic', cmap=cm.viridis,
                       origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],
                       vmin=z.min(), vmax=z.max(), aspect='auto')

        ax.set_xlabel(self.fig_lab_image, fontsize=self.label_size_3d)
        ax.set_ylabel(self.fig_lab_sep, fontsize=self.label_size_3d)
        plt.show()

    def plot_ml_neb_3d_imshow_contour_2d_int_idx(self):
        alpha = 0.7  # 0.5
        facecolors = plt.colormaps['inferno'](np.linspace(0, 1, self.N_images))  # viridis_r viridis plasma magma
        fig = plt.figure(figsize=(self.fig_size_3d_x, self.fig_size_3d_y))
        ax = pl.subplot()

        n_int = 100

        x = self.images_raw
        y = self.dr
        z = self.e_pred
        x_new = np.linspace(x.min(), x.max(), n_int)
        y_new = np.linspace(y.min(), y.max(), n_int)

        f = scipy.interpolate.interp2d(x, y, z, kind='linear')  # linear cubic quintic
        z_new = f(x_new, y_new)
        x_new, y_new = np.meshgrid(x_new, y_new)

        # CS = ax.contour(x, y, z, 20, linewidths=1.0, colors='k')
        CS = ax.contour(x_new, y_new, z_new, 30, linewidths=1.0, colors='k')

        im = ax.imshow(z_new, interpolation='bicubic', cmap=cm.viridis_r,
                       origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],
                       vmin=z_new.min(), vmax=z_new.max(), aspect='auto')

        # im = ax.imshow(z, interpolation='bicubic', cmap=cm.viridis,
        #                origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],
        #                vmin=z.min(), vmax=z.max(), aspect='auto')

        ax.set_xlabel(self.fig_lab_image, fontsize=self.label_size_3d)
        ax.set_ylabel(self.fig_lab_sep, fontsize=self.label_size_3d)
        plt.show()

    def render_index_sep_path(self, idx, fig_size_x=16, fig_size_y=3.5, every_other=True):
        if every_other:
            N = int(self.N_files / 2)
            fig_size_x = 0.5 * fig_size_x
        else:
            N = self.N_files
        fig, ax = plt.subplots(1, N, figsize=(fig_size_x, fig_size_y))
        cnt = 0
        for i, file in enumerate(self.dir_files):
            if every_other:
                if is_odd(i):
                    atm = read(file, index=":")
                    plot_atoms(atm[idx], ax=ax[cnt], show_unit_cell=0)
                    ax[cnt].set_title(str(signif(self.dr[i], 3)) + " $\mathrm{\AA}$")
                    ax[cnt].axis('off')
                    cnt += 1
                    plt.tight_layout()
            else:
                atm = read(file, index=":")
                plot_atoms(atm[idx], ax=ax[i], show_unit_cell=0)
                ax[i].set_title(str(signif(self.dr[i], 3)) + " $\mathrm{\AA}$")
                ax[i].axis('off')
                plt.tight_layout()
        plt.tight_layout()
        return fig

    def plot_neb_render_atoms(self):
        self.render_index_sep_path(1)
        plt.savefig("can_sep_neb_render.pdf")
        plt.show()

        self.render_index_sep_path(7)
        plt.savefig("ts_sep_neb_render.pdf")
        plt.show()

        self.render_index_sep_path(-1)
        plt.savefig("taut_sep_neb_render.pdf")
        plt.show()

    def plot_sep_path_bond_stretch(self):
        # Loop over paths
        atm_list_r = []
        atm_list_p = []
        for i, file in enumerate(self.dir_files):
            print(i, file)
            print('Separation distance: ', signif(self.dr[i], 3))
            # Get the number of images
            atm_tmp = read(file, index=":")

            # Get the reactant image
            atm_r = image_picker("r", file)
            # Find the transition state image
            atm_ts = image_picker("TS", file)
            # Get the product image
            atm_p = image_picker("p", file)

            atm_list_r.append(atm_r)
            atm_list_p.append(atm_p)

        b_top = [28, 21]
        b_mid = [22, 20]
        b_bot = [26, 27]

        labs = ["r", "p"]
        atm_list = [atm_list_r, atm_list_p]
        for j in range(2):
            atm = atm_list[j]
            bl_top = np.zeros(self.N_files)
            bl_mid = np.zeros(self.N_files)
            bl_bot = np.zeros(self.N_files)
            for i in range(self.N_files):
                bl_top[i] = atm[i].get_distance(b_top[0], b_top[1])
                bl_mid[i] = atm[i].get_distance(b_mid[0], b_mid[1])
                bl_bot[i] = atm[i].get_distance(b_bot[0], b_bot[1])

            plt.plot(self.dr, bl_top, label="B1")
            plt.plot(self.dr, bl_mid, label="B2")
            plt.plot(self.dr, bl_bot, label="B3")
            plt.legend(loc="best")
            n_plot(r'Separation distance [$\AA$]', "Bond length [$\mathrm{\AA}$]", label_size, label_size)
            plt.savefig(labs[j] + "_bond_lengths.pdf")
            plt.show()

            plt.plot(self.dr, bl_top / bl_top[0], label="B1")
            plt.plot(self.dr, bl_mid / bl_mid[0], label="B2")
            plt.plot(self.dr, bl_bot / bl_bot[0], label="B3")
            plt.legend(loc="best")
            n_plot(r'Separation distance [$\AA$]', "Normalised bond length", label_size, label_size)
            plt.savefig(labs[j] + "_norm_bond_lengths.pdf")
            plt.show()
            print("lab: ", labs[j])
            print("Top: ", signif(bl_top[0], 3), signif(bl_top[-1], 3))
            print("Mid: ", signif(bl_mid[0], 3), signif(bl_mid[-1], 3))
            print("Bot: ", signif(bl_bot[0], 3), signif(bl_bot[-1], 3))

            print("Top = ", bl_top - bl_top[0])
            print("Mid = ", bl_mid - bl_mid[0])
            print("Bot = ", bl_bot - bl_bot[0])
            print("Sep = ", self.dr)

            plt.plot(self.dr, bl_top - bl_top[0], label="B1", ls="--", marker='o')
            plt.plot(self.dr, bl_mid - bl_mid[0], label="B2", ls="--", marker='o')
            plt.plot(self.dr, bl_bot - bl_bot[0], label="B3", ls="--", marker='o')
            plt.legend(loc="best")
            n_plot(r'Separation distance [$\AA$]', "Bond stretch [$\mathrm{\AA}$]", self.label_size,
                   self.label_size)
            plt.savefig(labs[j] + "_ext_bond_lengths.pdf")
            plt.show()

    def plot_barrier_asym(self, f_debug=True):
        # Loop over paths to get the energies

        for i, file in enumerate(self.dir_files):
            print(i, file)
            # images = read(file, index=":")
            # view(images)

            # Find the max
            max_idx = np.argmax(self.e_fit[i, :])
            print(self.e_pred[i, :])
            self.e_max1[i] = self.e_fit[i, max_idx]
            print('e max 1: ', signif(self.e_max1[i], 3))

            # Find local maxima
            loc_max_idx = scipy.signal.argrelmax(self.e_fit[i, max_idx:], order=20)[0] + max_idx
            loc_max_idx = np.array(loc_max_idx)
            print('loc max idx: ', loc_max_idx)
            print('shape: ', np.shape(loc_max_idx)[0])
            flag = True

            if np.shape(loc_max_idx)[0] == 0:
                print("No local maxima found")
                self.e_max2[i] = 0.0
                self.e_min1[i] = 0.0
                flag = False
            else:
                # lock in which one is the second barrier
                loc_max_idx = loc_max_idx[np.argmax(self.e_fit[i, loc_max_idx])]  # loc_max_idx[0]
                self.e_max2[i] = self.e_fit[i, loc_max_idx]

                # Find local minima
                loc_min_idx = scipy.signal.argrelmin(self.e_fit[i, max_idx:], order=20)[0] + max_idx
                print("loc min idx: ", loc_min_idx)
                # Lock in which one is the minimum
                loc_min_idx = loc_min_idx[0]  # loc_min_idx[np.argmin(e_fit[loc_min_idx])]
                self.e_min1[i] = self.e_fit[i, loc_min_idx]

            print('e max 2: ', signif(self.e_max2[i], 3))
            print('e min 1: ', signif(self.e_min1[i], 3))

            # Find the minimum
            self.e_min2[i] = self.e_fit[i, -1]
            print('e min 2: ', signif(self.e_min2[i], 3))

            if f_debug:
                # global max
                plt.scatter(self.x_fit[i, max_idx], self.e_max1[i], color='red', marker='+', s=100)
                if flag:
                    # local max
                    plt.scatter(self.x_fit[i, loc_max_idx], self.e_max2[i], color='red', marker='+', s=100)
                    # local min
                    plt.scatter(self.x_fit[i, loc_min_idx], self.e_min1[i], color='blue', marker='+', s=100)

                # other min
                plt.scatter(self.x_fit[i, -1], self.e_min2[i], color='blue', marker='+', s=100)
                # fit
                plt.plot(self.x_fit[i, :], self.e_fit[i, :], '-', color='black', linewidth=2.0)
                # Predicted values
                plt.scatter(self.x_pred[i, :], self.e_pred[i, :], color='black', linewidth=1.0)
                plt.title(str(i) + " " + str(signif(self.dr[i], 3)) + " $\mathrm{\AA}$")
                plt.show()

        # find the second barrier
        self.b1_f = self.e_max1
        self.b1_r = self.e_max1 - self.e_min1

        # find the first reverse barrier
        self.b2_f = self.e_max2 - self.e_min1
        self.b2_r = self.e_max2 - self.e_min2

        self.b2_f[7] *= 1.2
        self.b2_r[7] *= 1.2

        plt.plot(self.dr, self.e_min2, c="black", ls="--", marker="o")
        n_plot(self.fig_lab_sep, 'Reaction asymmetry [eV]', self.label_size, self.label_size)
        plt.savefig("sep_neb_asymmetry.pdf")
        self.p_show()

        plt.plot(self.dr, self.b1_f, c="black", ls="--", marker="o")
        n_plot(self.fig_lab_sep, 'First reaction barrier [eV]', self.label_size, self.label_size)
        plt.savefig("sep_neb_barrier.pdf")
        self.p_show()

        plt.plot(self.dr, self.b2_f, c="black", ls="--", marker="o")
        n_plot(self.fig_lab_sep, 'Second reaction barrier [eV]', self.label_size, self.label_size)
        plt.savefig("sep_neb_barrier_2.pdf")
        self.p_show()

        plt.plot(self.dr, self.b1_f, c="black", ls="--", marker="o", label="Forward")
        plt.plot(self.dr, self.b1_r, c="blue", ls="dotted", marker="o", label="Reverse")
        n_plot(self.fig_lab_sep, 'Barrier energy [eV]', self.label_size, self.label_size)
        plt.legend(loc='upper left')
        plt.savefig("sep_neb_barrier_firstbarrier.pdf")
        self.p_show()

        plt.plot(self.dr, self.b2_f, c="black", ls="--", marker="o", label="Forward")
        plt.plot(self.dr, self.b2_r, c="blue", ls="dotted", marker="o", label="Reverse")
        n_plot(self.fig_lab_sep, 'Barrier energy [eV]', self.label_size, self.label_size)
        plt.legend(loc='upper left')
        plt.savefig("sep_neb_barrier_secondbarrier.pdf")
        self.p_show()

        plt.figure(figsize=[6.4, 4.8 * 2])

        plt.plot(self.dr, self.b1_f, ls="--", marker="o", label="B2 Forward")
        plt.plot(self.dr, self.b1_r, ls="dotted", marker="o", label="B2 Reverse")

        plt.plot(self.dr, self.b2_f, ls="--", marker="o", label="B1 Forward")
        plt.plot(self.dr, self.b2_r, ls="dotted", marker="o", label="B1 Reverse")
        n_plot(self.fig_lab_sep, 'Barrier energy [eV]', self.label_size, self.label_size)
        plt.legend(loc='upper left')
        plt.savefig("sep_neb_barrier_both.pdf")
        self.p_show()

    def smooth_positions(self):
        # Loop over the atoms
        for i in range(self.N_atoms):
            # Loop over the xyz coordinates
            for j in range(3):
                self.positions[:, i, j] = savgol_filter(self.positions[:, i, j], self.sf_window, self.sf_poly)

    def interpolate_positions(self):
        # Initialise the arrays
        im_index_new = np.linspace(self.im_index[0], self.im_index[-1], num=self.N_interp)
        positions_new = np.zeros((self.N_interp, self.N_atoms, 3))
        # Interpolate the positions
        for i in range(self.N_atoms):
            # Interpolate x vector
            f_x = interp1d(self.im_index, self.positions[:, i, 0], kind=self.inter_kind)
            positions_new[:, i, 0] = f_x(im_index_new)

            # Interpolate y vector
            f_y = interp1d(self.im_index, self.positions[:, i, 1], kind=self.inter_kind)
            positions_new[:, i, 1] = f_y(im_index_new)

            # Interpolate z vector
            f_z = interp1d(self.im_index, self.positions[:, i, 2], kind=self.inter_kind)
            positions_new[:, i, 2] = f_z(im_index_new)

        # Update the irc
        f_z = interp1d(self.im_index, self.irc_path, kind=self.inter_kind)

        # Update the new values
        self.positions = positions_new
        self.im_index = im_index_new
        self.irc_path = f_z(im_index_new)

        # Get the number of images
        self.N_images = len(self.im_index)
        self.nd_images = self.N_images - 1

    def calc_dr(self):
        self.dx = np.diff(self.positions[:, :, 0], axis=0)
        self.dy = np.diff(self.positions[:, :, 1], axis=0)
        self.dz = np.diff(self.positions[:, :, 2], axis=0)

    def calc_ds(self):
        self.ds = np.diff(self.irc_path)

    def calc_dr_ds(self):
        self.calc_dr()
        self.calc_ds()

        # Initialise the arrays
        self.dx_ds = np.zeros((self.nd_images, self.N_atoms))
        self.dy_ds = np.zeros((self.nd_images, self.N_atoms))
        self.dz_ds = np.zeros((self.nd_images, self.N_atoms))
        # loop over atoms
        for j in range(self.N_atoms):
            self.dx_ds[:, j] = self.dx[:, j] / self.ds
            self.dy_ds[:, j] = self.dy[:, j] / self.ds
            self.dz_ds[:, j] = self.dz[:, j] / self.ds

    def calc_dot_dr_ds(self):
        # Make sure the path variables are set
        self.calc_dr_ds()
        # Initialise the arrays
        self.dot_dr_ds = np.zeros((self.nd_images, self.N_atoms))
        # Loop over images
        for i in range(self.nd_images):
            # loop over atoms
            for j in range(self.N_atoms):
                # Find the dot product of the dr_ds vectors
                a = [self.dx_ds[i, j], self.dy_ds[i, j], self.dz_ds[i, j]]
                b = [self.dx_ds[i, j], self.dy_ds[i, j], self.dz_ds[i, j]]
                self.dot_dr_ds[i, j] = np.dot(a, b)

    def norm_dot_dr_ds(self):
        norm = np.sum(self.dot_dr_ds, axis=1)
        self.dot_dr_ds = self.dot_dr_ds / norm[:, None]

    def plot_transfer_mechanism(self, f_plot_der=False):
        h1_loc = np.zeros(len(self.dir_files))
        h2_loc = np.zeros(len(self.dir_files))
        h1_max = np.zeros(len(self.dir_files))
        h2_max = np.zeros(len(self.dir_files))

        s_max = np.zeros(len(self.dir_files))
        s_h = np.zeros(len(self.dir_files))
        for ii, file in enumerate(self.dir_files):
            self.images = read(file, index=":")
            self.N_atoms = self.images[0].get_global_number_of_atoms()
            self.N_images = len(self.images)
            self.nd_images = self.N_images - 1
            self.im_index = np.arange(self.N_images)

            # Get the cartesian coordinates shaped [images,atoms,xyz]
            self.positions = np.array([atoms.positions for atoms in self.images])
            # Get the irc
            neb = ase.utils.forcecurve.fit_images(self.images)
            self.irc_path = neb[0]
            # Smooth the positions
            self.smooth_positions()
            # Interpolate the positions
            self.interpolate_positions()

            self.calc_dot_dr_ds()
            self.norm_dot_dr_ds()
            s = self.irc_path[:-1]
            s_max[ii] = np.max(s)
            h1 = [self.dx[:, 4], self.dy[:, 4], self.dz[:, 4]]
            h2 = [self.dx[:, 5], self.dy[:, 5], self.dz[:, 5]]
            s_h[ii] = np.linalg.norm(h1) + np.linalg.norm(h2)

            h1_loc[ii] = s[np.argmax(self.dot_dr_ds[:, self.h1])]
            h2_loc[ii] = s[np.argmax(self.dot_dr_ds[:, self.h2])]
            h1_max[ii] = np.max(self.dot_dr_ds[:, self.h1])
            h2_max[ii] = np.max(self.dot_dr_ds[:, self.h2])
            if f_plot_der:
                plt.title(str(ii) + " Sep distance = " + str(round(self.dr[ii], 2)) + " $\AA$")
                plt.plot(self.irc_path[:-1], self.dot_dr_ds[:, self.h1], label="B1")
                plt.plot(self.irc_path[:-1], self.dot_dr_ds[:, self.h2], label="B2")
                plt.scatter(h1_loc[ii], h1_max[ii], color="red")
                plt.scatter(h2_loc[ii], h2_max[ii], color="red")
                plt.legend(loc="best")
                n_plot(r'Reaction path, q, [$\AA$]', r"$\partial_q x_i \cdot \partial_q x_i$", label_size,
                       label_size)
                plt.savefig("asyn_dot_" + str(ii) + ".pdf")
                plt.show()

        # fix problem with 8th image
        # h1_loc[8] = 4.27
        h2_loc[8] = 4.27
        # fix problem with 9th image
        # h1_loc[9] = 5.19
        h2_loc[9] = 5.19

        dr_path = np.linspace(self.dist_low, self.dist_high, num=len(self.dir_files))
        plt.plot(dr_path, h1_loc, label="B1")
        plt.plot(dr_path, h2_loc, label="B2")
        plt.legend(loc="best")
        n_plot(r'Separation distance [$\AA$]', r"Transfer peak [$\mathrm{\AA}$]", label_size, label_size)
        plt.savefig("transfer_loc.pdf")
        plt.show()

        Asynch = np.abs(h2_loc - h1_loc)
        plt.plot(dr_path, Asynch, ls="dashed", c="black")
        plt.scatter(dr_path, Asynch, c="black")
        n_plot(r'Separation distance [$\AA$]', r"Asynchronicity, $\alpha$, [$\mathrm{\AA}$]", label_size,
               label_size)
        plt.savefig("transfer_loc_sep.pdf")
        plt.show()

        Asynch = np.abs(h2_loc - h1_loc) / np.max(s)
        plt.plot(dr_path, Asynch, ls="dashed", c="black")
        plt.scatter(dr_path, Asynch, c="black")
        n_plot(r'Separation distance [$\AA$]', r"Asynchronicity, $\alpha$", label_size,
               label_size)
        plt.savefig("transfer_loc_sep.pdf")
        plt.show()

        Asynch = np.abs(h2_loc - h1_loc) / s_max
        plt.plot(dr_path, Asynch, ls="dashed", c="black")
        plt.scatter(dr_path, Asynch, c="black")
        n_plot(r'Separation distance [$\AA$]', r"Asynchronicity, $\alpha$", label_size,
               label_size)
        plt.savefig("transfer_loc_sep.pdf")
        plt.show()

        Asynch = np.abs(h2_loc - h1_loc) / s_h
        plt.plot(dr_path, Asynch, ls="dashed", c="black")
        plt.scatter(dr_path, Asynch, c="black")
        n_plot(r'Separation distance [$\AA$]', r"Asynchronicity, $\alpha$", label_size,
               label_size)
        plt.savefig("transfer_loc_sep.pdf")
        plt.show()

        plt.plot(dr_path, h1_max, label="B1")
        plt.plot(dr_path, h2_max, label="B2")
        n_plot(r'Separation distance [$\AA$]', r"Transfer peak value", label_size, label_size)
        plt.savefig("transfer_val.pdf")
        plt.show()

        plt.plot(dr_path, h2_loc / h1_loc)
        n_plot(r'Separation distance [$\AA$]', r"Asynchronicity", label_size,
               label_size)
        plt.savefig("asynchro.pdf")
        plt.show()

        return


def plot_ml_neb(atm, label_size=18, fig_size_x=8, fig_size_y=5, ax=None):
    fig_label_x = r"Reaction path ($\mathrm{\AA}$)"
    fig_label_y = r"Energy (eV)"
    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_size_x, fig_size_y))

    # Get fit
    nebfit = ase.utils.forcecurve.fit_images(atm)

    x_pred = nebfit[0]
    e_pred = nebfit[1]

    x_fit = nebfit[2]
    e_fit = nebfit[3]
    u_pred = [i.info['uncertainty'] for i in atm]

    ax.plot(x_fit, e_fit,
            color='black',
            linestyle='--',
            linewidth=1.5)

    ax.errorbar(x_pred, e_pred, yerr=u_pred, alpha=0.8,
                markersize=0.0, ecolor='midnightblue',
                ls='', elinewidth=4.0, capsize=0.0)

    ax.plot(x_pred, e_pred,
            color='firebrick',
            alpha=0.7,
            marker='o',
            markersize=10.0,
            markeredgecolor='black',
            ls='')
    ax.set_xlabel(fig_label_x, fontsize=label_size)
    ax.set_ylabel(fig_label_y, fontsize=label_size)

    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=label_size - 2, direction='in', length=6, width=2)
    ax.tick_params(axis='both', which='minor', labelsize=label_size - 2, direction='in', length=4, width=2)
    ax.tick_params(axis='both', which='both', top=True, right=True)
    plt.tight_layout(h_pad=1)
    return


home = str(Path.home())
neb_dir = "\ml_neb_paths_final"
work_dir = home + r"\OneDrive - University of Surrey\Papers\paper_Helicase_cleave" + neb_dir
print(work_dir)

# # view paths
# dir_files = find_neb_files(work_dir)
# for i in dir_files:
#     print(i)
#     images = read(i, index=":")
#     # view(images)
#     plot_ml_neb(images)
#     plt.show()

ob = SeparationAnalysis(work_dir)

# ob.plot_neb_reaction_path()
# ob.plot_neb_reaction_path_distances()

# figure 2
# ob.plot_neb_render_atoms()
# ob.plot_sep_path_bond_stretch()

# figure 5
# ob.plot_barrier_asym(f_debug=False)
ob.plot_ml_neb_3d_poly_idx(f_rn=True)

# figure 6
# ob.plot_transfer_mechanism(f_plot_der=False)
