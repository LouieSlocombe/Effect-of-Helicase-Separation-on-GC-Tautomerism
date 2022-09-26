#!/usr/bin/env python3
import copy
import os
import sys
import time
from pathlib import Path

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.optimize import BFGS
from ase.visualize import view
from ase.visualize.plot import plot_atoms
import ase
import scipy
from matplotlib import cm
from matplotlib.colors import LightSource
from matplotlib.collections import PolyCollection

import ase_common_utils_v03 as acom
import common_utils_v01 as common

plt.rcParams['axes.linewidth'] = 2.0
label_size = 18


def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]


def find_neb_files(work_dir, f_name1="last_predicted_path", f_name2="_interpolated"):
    # Find the first set of files
    l1 = [os.path.join(work_dir, i) for i in common.sub_file_list(work_dir, f_name1)]
    # Find the second set of files
    l2 = [os.path.join(work_dir, i) for i in common.sub_file_list(work_dir, f_name2)]
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

        self.f_correct = True
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

    def p_show(self):
        if self.f_show:
            plt.show()
        plt.close()
        return None

    def get_files(self):
        # self.files = common.sub_file_list(self.work_dir, self.name_prototype)
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
                line = tmp1[0] + 'energy=' + str(energy) + ' ' + str_stitch(rest)
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
            common.n_plot(self.fig_lab_image_rn, r'Distance [$\AA$]', self.label_size, self.label_size)
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
        common.n_plot(self.fig_lab_sep, r'Reaction path distance [$\AA$]', self.label_size, self.label_size)
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
            f = scipy.interpolate.interp1d(x, self.e_pred[i, :], kind='linear')
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
                if common.is_odd(i):
                    atm = read(file, index=":")
                    plot_atoms(atm[idx], ax=ax[cnt], show_unit_cell=0)
                    ax[cnt].set_title(str(common.signif(self.dr[i], 3)) + " $\mathrm{\AA}$")
                    ax[cnt].axis('off')
                    cnt += 1
                    plt.tight_layout()
            else:
                atm = read(file, index=":")
                plot_atoms(atm[idx], ax=ax[i], show_unit_cell=0)
                ax[i].set_title(str(common.signif(self.dr[i], 3)) + " $\mathrm{\AA}$")
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
            print('Separation distance: ', common.signif(self.dr[i], 3))
            # Get the number of images
            atm_tmp = read(file, index=":")

            # Get the reactant image
            atm_r = acom.image_picker("r", file)
            # Find the transition state image
            atm_ts = acom.image_picker("TS", file)
            # Get the product image
            atm_p = acom.image_picker("p", file)

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
            common.n_plot(r'Separation distance [$\AA$]', "Bond length [$\mathrm{\AA}$]", label_size, label_size)
            plt.savefig(labs[j] + "_bond_lengths.pdf")
            plt.show()

            plt.plot(self.dr, bl_top / bl_top[0], label="B1")
            plt.plot(self.dr, bl_mid / bl_mid[0], label="B2")
            plt.plot(self.dr, bl_bot / bl_bot[0], label="B3")
            plt.legend(loc="best")
            common.n_plot(r'Separation distance [$\AA$]', "Normalised bond length", label_size, label_size)
            plt.savefig(labs[j] + "_norm_bond_lengths.pdf")
            plt.show()
            print("lab: ", labs[j])
            print("Top: ", common.signif(bl_top[0], 3), common.signif(bl_top[-1], 3))
            print("Mid: ", common.signif(bl_mid[0], 3), common.signif(bl_mid[-1], 3))
            print("Bot: ", common.signif(bl_bot[0], 3), common.signif(bl_bot[-1], 3))

            print("Top = ", bl_top - bl_top[0])
            print("Mid = ", bl_mid - bl_mid[0])
            print("Bot = ", bl_bot - bl_bot[0])
            print("Sep = ", self.dr)

            plt.plot(self.dr, bl_top - bl_top[0], label="B1", ls="--", marker='o')
            plt.plot(self.dr, bl_mid - bl_mid[0], label="B2", ls="--", marker='o')
            plt.plot(self.dr, bl_bot - bl_bot[0], label="B3", ls="--", marker='o')
            plt.legend(loc="best")
            common.n_plot(r'Separation distance [$\AA$]', "Bond stretch [$\mathrm{\AA}$]", self.label_size,
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
            print('e max 1: ', common.signif(self.e_max1[i], 3))

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

            print('e max 2: ', common.signif(self.e_max2[i], 3))
            print('e min 1: ', common.signif(self.e_min1[i], 3))

            # Find the minimum
            self.e_min2[i] = self.e_fit[i, -1]
            print('e min 2: ', common.signif(self.e_min2[i], 3))

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
                plt.title(str(i) + " " + str(common.signif(self.dr[i], 3)) + " $\mathrm{\AA}$")
                plt.show()

        self.e_max2[6] = 0.619
        self.e_max2[7] = 0.64  # 0.552
        self.e_max2[9] = 0.71

        # find the second barrier
        self.b1_f = self.e_max1
        self.b1_r = self.e_max1 - self.e_min1
        # find the first reverse barrier
        self.b2_f = self.e_max2 - self.e_min1
        self.b2_r = self.e_max2 - self.e_min2

        plt.plot(self.dr, self.e_min2, c="black", ls="--", marker="o")
        common.n_plot(self.fig_lab_sep, 'Reaction asymmetry [eV]', self.label_size, self.label_size)
        plt.savefig("sep_neb_asymmetry.pdf")
        self.p_show()

        plt.plot(self.dr, self.b1_f, c="black", ls="--", marker="o")
        common.n_plot(self.fig_lab_sep, 'First reaction barrier [eV]', self.label_size, self.label_size)
        plt.savefig("sep_neb_barrier.pdf")
        self.p_show()

        plt.plot(self.dr, self.b2_f, c="black", ls="--", marker="o")
        common.n_plot(self.fig_lab_sep, 'Second reaction barrier [eV]', self.label_size, self.label_size)
        plt.savefig("sep_neb_barrier_2.pdf")
        self.p_show()

        plt.plot(self.dr, self.b1_f, c="black", ls="--", marker="o", label="Forward")
        plt.plot(self.dr, self.b1_r, c="blue", ls="dotted", marker="o", label="Reverse")
        common.n_plot(self.fig_lab_sep, 'Barrier energy [eV]', self.label_size, self.label_size)
        plt.legend(loc='upper left')
        plt.savefig("sep_neb_barrier_firstbarrier.pdf")
        self.p_show()

        plt.plot(self.dr, self.b2_f, c="black", ls="--", marker="o", label="Forward")
        plt.plot(self.dr, self.b2_r, c="blue", ls="dotted", marker="o", label="Reverse")
        common.n_plot(self.fig_lab_sep, 'Barrier energy [eV]', self.label_size, self.label_size)
        plt.legend(loc='upper left')
        plt.savefig("sep_neb_barrier_secondbarrier.pdf")
        self.p_show()

        plt.figure(figsize=[6.4, 4.8 * 2])

        plt.plot(self.dr, self.b1_f, ls="--", marker="o", label="B2 Forward")
        plt.plot(self.dr, self.b1_r, ls="dotted", marker="o", label="B2 Reverse")

        plt.plot(self.dr, self.b2_f, ls="--", marker="o", label="B1 Forward")
        plt.plot(self.dr, self.b2_r, ls="dotted", marker="o", label="B1 Reverse")
        common.n_plot(self.fig_lab_sep, 'Barrier energy [eV]', self.label_size, self.label_size)
        plt.legend(loc='upper left')
        plt.savefig("sep_neb_barrier_both.pdf")
        self.p_show()


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


def plot_stacked_3d(x, y, z, x_label, y_label, z_label, title=None, filename='plot.pdf'):
    alpha = 0.5
    n_plots = len(y)

    y_i = np.outer(y, np.ones(x.size))
    # Init figure
    pl.figure()
    ax = pl.subplot(projection='3d')
    # Loop over the plots
    for i in range(n_plots):
        ax.plot(x, y_i[i, :], z, alpha=alpha)

    # Set the axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    # Set the title
    if title is not None:
        ax.set_title(title)
    # Save the figure
    if filename is not None:
        plt.savefig(filename)

    plt.show()


def plot_ml_neb_3d(dir_files, dr, label_size=16, fig_size_x=8, fig_size_y=8, f_plot_index=True):
    # https://stackoverflow.com/questions/34099518/plotting-a-series-of-2d-plots-projected-in-3d-in-a-perspectival-way

    if f_plot_index:
        fig_label_x = r"Image number"
    else:
        fig_label_x = r"Reaction path ($\mathrm{\AA}$)"
    fig_label_y = r'Separation distance [$\AA$]'
    fig_label_z = r'Energy [eV]'
    alpha = 0.7  # 0.5

    # Get the size of things
    atm = read(dir_files[0], index=":")
    nebfit = ase.utils.forcecurve.fit_images(atm)
    n_images = len(nebfit[2])
    n_images = len(atm)
    images_raw = np.arange(len(atm), dtype=int)
    images = np.arange(n_images, dtype=int)

    y_i = np.outer(dr, np.ones(n_images))

    facecolors = plt.colormaps['inferno'](np.linspace(0, 1, n_images))  # viridis_r viridis plasma magma

    # Init figure
    # pl.figure(facecolor='black')
    fig = plt.figure(figsize=(fig_size_x, fig_size_y))
    ax = pl.subplot(projection='3d')
    for i, file in enumerate(dir_files):
        print(i, file)
        # Read the file
        atm = read(file, index=":")
        # Get fit
        nebfit = ase.utils.forcecurve.fit_images(atm)

        x_pred = nebfit[0]
        e_pred = nebfit[1]

        x_fit = nebfit[2]
        e_fit = nebfit[3]
        u_pred = [i.info['uncertainty'] for i in atm]

        e_pred = np.add(e_pred, u_pred)

        if f_plot_index:
            x = images_raw
        else:
            x = x_pred

        # Plot the fit
        ax.plot(x, y_i[i, :], e_pred,
                alpha=alpha,
                color=facecolors[i],
                linewidth=1.5)

        ax.errorbar(x, y_i[i, :], e_pred,
                    zerr=u_pred,
                    alpha=0.8,
                    markersize=0.0,
                    ecolor='red',  # midnightblue
                    ls='',
                    elinewidth=2.0,
                    capsize=0.0)

    # Set the axis labels
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # print(labels)
    # ax.*axis.set_ticklabels(images)
    # plt.xticks(images, images, )  # rotation='vertical')
    ax.set_xlabel(fig_label_x, fontsize=label_size)
    ax.set_ylabel(fig_label_y, fontsize=label_size)
    ax.set_zlabel(fig_label_z, fontsize=label_size)
    # ax.set(xlim=(images[0], images[-1]), ylim=(dr[0], dr[-1]), zlim=(0, max(e_pred)))
    ax.set(ylim=(dr[0], dr[-1]), zlim=(0, max(e_pred)))
    # ax.minorticks_on()
    # ax.tick_params(axis='both', which='major', labelsize=label_size - 2, direction='in', length=6, width=2)
    # ax.tick_params(axis='both', which='minor', labelsize=label_size - 2, direction='in', length=4, width=2)
    # ax.tick_params(axis='both', which='both', top=True, right=True)
    # plt.tight_layout(h_pad=1)
    # ax.view_init(azim=-80.0, elev=35.0)
    plt.show()

    return


def render_sep_path(file, fig_size_x=8, fig_size_y=3):
    atm = read(file, index=":")
    N_images = len(atm)

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(fig_size_x, fig_size_y))

    # Get the reactant image
    atm_r1 = acom.image_picker("r", file)
    # Find the transition state image
    atm_ts, ts_idx = acom.image_picker("TS", file, rtn_idx=True)
    # Get the product image
    atm_p1 = acom.image_picker("p", file)

    r2_idx = int((0 + ts_idx) / 2)
    atm_r2 = acom.image_picker(r2_idx, file)

    p2_idx = int((N_images + ts_idx) / 2)
    atm_p2 = acom.image_picker(p2_idx, file)

    # Plot the reactant image 1
    plot_atoms(atm_r1, ax=ax1, show_unit_cell=0)
    ax1.set_title("1 (R)")
    ax1.axis('off')

    # Plot the reactant image 2
    plot_atoms(atm_r2, ax=ax2, show_unit_cell=0)
    ax2.set_title(str(r2_idx))
    ax2.axis('off')

    # Plot the TS image
    plot_atoms(atm_ts, ax=ax3, show_unit_cell=0)
    ax3.set_title(str(ts_idx) + " (TS)")
    ax3.axis('off')

    # Plot the product image 2
    plot_atoms(atm_p2, ax=ax4, show_unit_cell=0)
    ax4.set_title(str(p2_idx))
    ax4.axis('off')

    # Plot the product image 1
    plot_atoms(atm_p1, ax=ax5, show_unit_cell=0)
    ax5.set_title(str(N_images) + " (P)")
    ax5.axis('off')
    plt.tight_layout()
    return fig


def plot_stack():
    home = str(Path.home())
    # plot the stack
    file = "ds_GGG_chop_split.traj"
    work_dir = home + r"\OneDrive - University of Surrey\Work\paper_Helicase_cleave"
    file_path = os.path.join(work_dir, file)
    print(file_path)
    atm = read(file_path, index=":")
    N_atm = len(atm)
    # view(atm)
    f_index = False

    # Plot the index vs the COM separation distance
    l1 = [75, 76, 93, 77, 78, 79, 80, 81, 94, 82, 83, 96, 95, 84, 85]
    l2 = [107, 114, 122, 113, 123, 111, 112, 125, 124, 110, 108, 109]
    # Just the R groups
    l1 = [75]
    l2 = [107]
    dr = np.zeros(N_atm)
    for i in range(N_atm):
        # Find the unit vector of the canonical base
        com_a = atm[i][l1].get_center_of_mass()
        com_b = atm[i][l2].get_center_of_mass()
        # Subtract the two COM vectors
        sub = np.subtract(com_a, com_b)
        # Find the unit vector in spherical coords
        ad_uv = common.spherical_coords(sub)
        dr[i] = ad_uv[0]
    # Zero the distance
    dr = dr - dr[0]
    plt.plot(range(N_atm), dr, c="black")
    common.n_plot("Index", r'Separation distance [$\AA$]', label_size, label_size)
    plt.savefig("stack_separation_distance.pdf")
    plt.show()

    N = np.arange(0, N_atm, 20)
    N = [0, 50, 104]
    fig, ax = plt.subplots(1, len(N), figsize=(10, 3))

    for i, val in enumerate(N):
        print(i, val)
        # plot_atoms(atm[val], ax=ax[i], show_unit_cell=0, rotation=('30x,10y,0z'))
        plot_atoms(atm[val], ax=ax[i], show_unit_cell=0)
        ax[i].axis('off')
        if f_index:
            title = str(val + 1)
        else:
            title = str(common.signif(dr[val], 3)) + " $\mathrm{\AA}$"
        ax[i].set_title(title)
        plt.tight_layout()
    plt.tight_layout()
    plt.savefig("stack_view.pdf")
    plt.show()

    # plot the bond lengths
    b_top = [112, 80]
    b_mid = [110, 81]
    b_bot = [109, 83]

    bl_top = np.zeros(N_atm)
    bl_mid = np.zeros(N_atm)
    bl_bot = np.zeros(N_atm)
    for i in range(N_atm):
        bl_top[i] = atm[i].get_distance(b_top[0], b_top[1])
        bl_mid[i] = atm[i].get_distance(b_mid[0], b_mid[1])
        bl_bot[i] = atm[i].get_distance(b_bot[0], b_bot[1])

    plt.plot(range(N_atm), bl_top, label="Top")
    plt.plot(range(N_atm), bl_mid, label="Mid")
    plt.plot(range(N_atm), bl_bot, label="Bot")
    plt.legend(loc="best")
    common.n_plot("Index", "Bond length [$\mathrm{\AA}$]", label_size, label_size)
    plt.savefig("stack_bond_lengths.pdf")
    plt.show()

    plt.plot(range(N_atm), bl_top / bl_top[0], label="Top")
    plt.plot(range(N_atm), bl_mid / bl_mid[0], label="Mid")
    plt.plot(range(N_atm), bl_bot / bl_bot[0], label="Bot")
    plt.legend(loc="best")
    common.n_plot("Index", "Normalised bond length", label_size, label_size)
    plt.savefig("stack_norm_bond_lengths.pdf")
    plt.show()

    print("Top: ", common.signif(bl_top[0], 3), common.signif(bl_top[-1], 3))
    print("Mid: ", common.signif(bl_mid[0], 3), common.signif(bl_mid[-1], 3))
    print("Bot: ", common.signif(bl_bot[0], 3), common.signif(bl_bot[-1], 3))

    plt.plot(dr, bl_top - bl_top[0], label="Top", ls="--")
    plt.plot(dr, bl_mid - bl_mid[0], label="Mid", ls="--")
    plt.plot(dr, bl_bot - bl_bot[0], label="Bot", ls="--")
    plt.legend(loc="best")
    common.n_plot(r'Separation distance [$\AA$]', "Bond stretch [$\mathrm{\AA}$]", label_size, label_size)
    plt.savefig("stack_ext_bond_lengths.pdf")
    plt.show()


def plot_long():
    print("Plotting long")
    home = str(Path.home())
    work_dir = home + r"\\OneDrive - University of Surrey\\Work\\paper_Helicase_cleave\\long\\"
    print(work_dir)
    # Separation details
    dist_low = 0.0  # 0.0
    dist_high = 5.0  # 5.0 10
    dist_num = 3  # 3 10
    dr = np.linspace(dist_low, dist_high, num=dist_num)
    print(dr)

    file = "long_sep_neb_2_5.traj"
    file = os.path.join(work_dir, file)
    render_sep_path(file)
    plt.savefig("2_5_sep_neb_render.pdf")
    plt.show()

    atm = read(file, index=":")
    view(atm)
    plot_ml_neb(atm)
    plt.savefig("2_5_sep_neb.pdf")
    plt.show()

    file = "long_sep_neb_5_0.traj"
    file = os.path.join(work_dir, file)
    render_sep_path(file)
    plt.savefig("5_0_sep_neb_render.pdf")
    plt.show()

    atm = read(file, index=":")
    view(atm)
    plot_ml_neb(atm)
    plt.savefig("5_0_sep_neb.pdf")
    plt.show()


def plot_ml_neb_all(dir_files, label_size=18, fig_size_x=8, fig_size_y=5, ):
    fig, ax = plt.subplots(figsize=(fig_size_x, fig_size_y))
    for i, file in enumerate(dir_files):
        print(i, file)
        if common.is_odd(i):
            continue
        print('Separation distance: ', common.signif(dr[i], 3))
        atm = read(file, index=":")
        plot_ml_neb(atm, ax=ax)
    plt.show()
    return


def plot_spe():
    data = np.genfromtxt("spe_asym.txt", delimiter="\n", dtype=str)

    dr = [i.replace(" ", "") for i in data if "sep:" in i]
    dr = np.asarray([i.split(":")[1] for i in dr], dtype=float)

    asym = [i.replace(" ", "") for i in data if "energy diff:" in i]
    asym = np.asarray([i.split(":")[1] for i in asym], dtype=float)

    energy = [i for i in data if "energy:" in i]
    energy = [i.split(":")[1] for i in energy]
    energy = np.asarray(
        [np.fromstring(i.replace("               ", "").replace("[", "").replace(" ]", ""), count=2, sep=' ') for i in
         energy])
    print(energy)
    print(np.shape(energy))

    plt.plot(dr, energy[:, 0] - energy[0, 0], c="black", ls="--", marker="o")
    common.n_plot(r'Separation distance [$\AA$]', 'Canonical energy [eV]', label_size, label_size)
    plt.savefig("sep_spe_can_energy.pdf")
    plt.show()

    plt.plot(dr, energy[:, 1] - energy[0, 1], c="black", ls="--", marker="o")
    common.n_plot(r'Separation distance [$\AA$]', 'Tautomeric energy [eV]', label_size, label_size)
    plt.savefig("sep_spe_taut_energy.pdf")
    plt.show()

    plt.plot(dr, np.abs(asym), c="black", ls="--", marker="o")
    common.n_plot(r'Separation distance [$\AA$]', 'Reaction asymmetry [eV]', label_size, label_size)
    plt.savefig("sep_spe_asymmetry.pdf")
    plt.show()
    return None


home = str(Path.home())
# neb_dir = "\ml_neb_paths_20220321"  # \ml_neb_paths
# neb_dir = "\ml_neb_paths_20220406"  # \ml_neb_paths
neb_dir = "\ml_neb_paths_20220516"
neb_dir = "\ml_neb_paths_20220525"
neb_dir = "\ml_neb_paths_20220530"
neb_dir = "\ml_neb_paths_20220823"
neb_dir = "\ml_neb_paths_20220906"
work_dir = home + r"\OneDrive - University of Surrey\Papers\paper_Helicase_cleave" + neb_dir

# neb_dir = "\ml_neb_paths_20220518"
# work_dir = home + r"\OneDrive - University of Surrey\Papers\paper_Helicase_cleave_AT" + neb_dir

print(work_dir)

ob = SeparationAnalysis(work_dir)
# ob.plot_neb_reaction_path()
# ob.plot_neb_reaction_path_distances()

# ob.plot_neb_render_atoms()
# ob.plot_sep_path_bond_stretch()
ob.plot_barrier_asym()

# ob.plot_ml_neb_3d_idx()
# ob.plot_ml_neb_3d_raw()
# ob.plot_ml_neb_3d_fit()
# ob.plot_ml_neb_3d_poly_idx(f_rn=True)
# ob.plot_ml_neb_3d_surface_idx()
# ob.plot_ml_neb_3d_surface_2d_int_idx()
# ob.plot_ml_neb_3d_wire_idx()
# ob.plot_ml_neb_3d_imshow_2d_int_idx()
exit()

f_flush = True
f_show = True

f_plot_spe = False
f_plot_stack = False
f_plot_path = True
f_plot_long = False

f_plot_ml_neb_all = False
f_plot_ml_neb_3d = False

f_load_custom_path = False
f_read_custom = False
f_render_custom = False

# Separation details
dist_low = 0.0  # 0.0
dist_high = 2.0  # 2.0 5.0 10
dist_num = 10  # 3 10
dr = np.linspace(dist_low, dist_high, num=dist_num)
# Init the energy vector
energies = np.zeros([2, dist_num])

dir_files = find_neb_files(work_dir)
N_files = len(dir_files)
asym = np.zeros(N_files)
barrier = np.zeros(N_files)
barrier_2 = np.zeros(N_files)

if f_plot_path:
    # Loop over paths
    f_view = True
    for i, file in enumerate(dir_files):
        print(i, file)
        print('Separation distance: ', common.signif(dr[i], 3))
        atm = read(file, index=":")
        view(atm)
        N_images = len(atm)
        energies = [ii.get_potential_energy() for ii in atm]
        energies = np.subtract(energies, energies[0])
        asym[i] = energies[-1]
        barrier[i] = np.max(energies)
        image_num = range(N_images)

        # Plot NEB
        plot_ml_neb(atm, label_size=label_size)
        plt.savefig(str(i) + "_sep_neb.pdf")
        plt.show()

        # Render the NEB path
        render_sep_path(file)
        plt.savefig(str(i) + "_sep_neb_render.pdf")
        plt.show()

if f_plot_stack:
    plot_stack()

if f_plot_long:
    plot_long()

if f_plot_ml_neb_all:
    plot_ml_neb_all(dir_files)

if f_plot_ml_neb_3d:
    plot_ml_neb_3d(dir_files, dr)

if f_read_custom:
    base = r"C:\\Users\\ls00338\\OneDrive - University of Surrey\\Work\\paper_Helicase_cleave\\"
    atm_path = os.path.join(base, "last_predicted_path_0.traj")
    atm = read(atm_path, index="0")
    print(atm.constraints)
    delattr(atm, "constraints")
    print(atm.constraints)
    view(atm)
    name = "G_C_sep_gold.traj"  # G_C_sep_gold G_enol_C_imino_sep_gold
    atm_path = os.path.join(base, name)
    write(atm_path, atm)

if f_render_custom:
    fig_size_x = 4
    fig_size_y = 3.5
    fig, ax = plt.subplots(1, 1, figsize=(fig_size_x, fig_size_y))
    atm = read(dir_files[0], index=":")
    plot_atoms(atm[0], ax=ax, show_unit_cell=0)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("custom_render.pdf")
    plt.show()

if f_plot_spe:
    plot_spe()

if f_load_custom_path:
    atm = read(dir_files[0], index=":")
    plot_ml_neb(atm)
    plt.show()

    # Get the path parameters
    # Get fit
    nebfit = ase.utils.forcecurve.fit_images(atm)

    # Get the actual values
    x_pred = nebfit[0]
    e_pred = nebfit[1]

    print("x_pred: ", x_pred)
    print("e_pred: ", e_pred)

    # Get the fit
    x_fit = nebfit[2]
    e_fit = nebfit[3]
    u_pred = [i.info['uncertainty'] for i in atm]

    x_new = np.linspace(0, )

    # depth, width, displacement
    # args =
    # common.v_d_morse(0, x, args)

#
# fileobj = read(dir_files[-1], index=":")
# #view(fileobj)
# images = 7
# ase.io.write("toc.pdb",fileobj[images],format="proteindatabank")
