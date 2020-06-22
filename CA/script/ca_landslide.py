#!/usr/bin/env python3
# Author: Paulo Quilao
# MS Geomatics Engineering Thesis
# Simulation of landslides using Cellular Automata


import os
import glob
import gdal
import imageio
import numpy as np


def read_raster(file):
    """Read image file and returns its data source and converted numpy array."""
    data_source = gdal.Open(file)
    band = data_source.GetRasterBand(1).ReadAsArray()
    return (data_source, band)


def create_png(path):
    """Create PNG images based on GeoTiff files."""
    folder = "png"
    op = os.path.join(path, folder)
    if not os.path.exists(op):
        os.makedirs(op)

    search = os.path.join(path, "*.tif")
    tif_paths = glob.glob(search)
    options_list = ["-ot Byte", "-of PNG", "-b 1", "-scale"]
    options_string = " ".join(options_list)
    counter = 0
    for tif in tif_paths:
        out_fn = f"ls_pred{counter}.png"
        gdal.Translate(os.path.join(op, out_fn), tif, options=options_string)
        counter += 1


def create_gif(path, out_fn):
    image_paths = glob.glob(os.path.join(path, "*.png"))
    output_gif_path = os.path.join(path, out_fn)
    # Save the animation to disk with 48 ms durations
    imageio.mimsave(output_gif_path, [imageio.imread(fp) for fp in image_paths], duration=0.48, subrectangles=True)
    if os.path.isfile(output_gif_path):
        print("\nGIF successfully exported.")
    else:
        print("\nFailed to create GIF.")


class Factors:
    def __init__(self, **params):   # keys are flow_dir, and lsi
        self.flow_dir_ds, self.flow_dir_arr = read_raster(params["flow_dir"])
        self.lsi_ds, self.lsi_arr = read_raster(params["lsi"])
        self.check_shape()

    def check_shape(self):
        print("Checking shapes of input growth factors...")
        if self.flow_dir_arr.shape == self.lsi_arr.shape:
            self.row = self.flow_dir_arr.shape[0]
            self.col = self.flow_dir_arr.shape[1]
            print("All factors have the same shape.")
        else:
            print("Shape of factors does not match.")


class Cellular_Automata:
    def __init__(self, factors, neigh_size):
        self.factors = factors
        self.neigh_size = tuple([neigh_size] * 2)  # kernel dimension

    def set_threshold(self, **params):
        """
        Computes the flow direction and lsi thresholds to be considered in the moore neighborhood.
        Params: actual landslide and initial cell (keys: "actual_ls" and "initial_cell")
        """
        self.initial_cell = read_raster(params["initial_cell"])[1].astype(int)
        self.actual_ls = read_raster(params["actual_ls"])[1].astype(int)
        assert self.initial_cell.shape == self.actual_ls.shape, "Shape of the input thresholds should match."
        # Get the center pixel of the kernel
        cent = int((self.neigh_size[0] - 2) - (self.neigh_size[0] - 3) / 2)
        self.center = tuple([cent] * 2)
        self.index = np.where(self.initial_cell == 1)
        kernel = np.full(self.neigh_size, self.factors.flow_dir_arr[self.index])
        kernel[self.center] = 0

        # mask
        m = np.isin(kernel, [0])
        # Flow direction threshold
        self.fd_threshold_low = np.where(m, kernel, kernel - 1).mean()  # low thresh flow_dir
        self.fd_threshold_hi = np.where(m, kernel, kernel + 1).mean()   # high thresh flow_Dir

        # LSI threshold
        kernel_items = np.prod(self.neigh_size)
        self.lsi_threshold_hi = (4 * (kernel_items - 1)) / kernel_items  # high lsi, less 1 for the center pixel
        self.lsi_threshold_vhi = (5 * (kernel_items - 1)) / kernel_items  # very high lsi, less 1 for the center pixel

    def simulate(self, export=False):
        """Implement CA using the growth factors and set thresholds."""
        print("\nSimulating landslide...")
        # landslide base
        self.landslide_predicted = np.zeros_like(self.factors.lsi_arr)

        col = self.neigh_size[0] - 1
        for i in range(1, self.factors.row - 1):
            # skip no data values
            if self.actual_ls[i].mean() == 0:
                continue
            for j in range(1, len(self.actual_ls[i]) - 1):
                try:
                    # flow dir kernel
                    kernel_fd = self.factors.flow_dir_arr[i-1:i+col, j-1:j+col]
                    # LSI kernel
                    kernel_LSI = self.factors.lsi_arr[i-1:i+col, j-1:j+col]

                    # kernel where means will be computed
                    kernel_no_center_fd = kernel_fd.copy()
                    kernel_no_center_LSI = kernel_LSI.copy()
                    kernel_no_center_fd[self.center] = 0
                    kernel_no_center_LSI[self.center] = 0

                    mean_fd = kernel_no_center_fd.mean()
                    mean_LSI = kernel_no_center_LSI.mean()
                    # Rule for landslide transition
                    if self.fd_threshold_low <= mean_fd <= self.fd_threshold_hi and self.lsi_threshold_hi <= mean_LSI <= self.lsi_threshold_vhi:
                        self.landslide_predicted[i][j] = 1

                # if raster dimension is not divisible by the kernel size
                except IndexError:
                    continue

            if export:
                # Export image for each row
                fn = f"ls_pred{i + 1 - self.index[0][0]}.tif"
                folder = "runs"
                op = os.path.join(os.getcwd(), folder)
                if not os.path.exists(op):
                    os.makedirs(op)
                self.export_predicted(os.path.join(op, fn))

        print("Finished implementing CA.")
        return self.landslide_predicted

    def export_predicted(self, out_fn):
        print("\nExporting array to GTiff...")
        driver = gdal.GetDriverByName("GTiff")
        out_data = driver.Create(out_fn, self.factors.col, self.factors.row, 1, gdal.GDT_UInt16)
        out_data.SetGeoTransform(self.factors.lsi_ds.GetGeoTransform())
        out_data.SetProjection(self.factors.lsi_ds.GetProjection())
        out_data.GetRasterBand(1).WriteArray(self.landslide_predicted)
        out_data.GetRasterBand(1).SetNoDataValue(-99999)
        out_data.FlushCache()
        out_data = None

        # check if exported
        if os.path.isfile(out_fn):
            print("Raster exported.")
        else:
            print("Failed to export raster.")


if __name__ == "__main__":
    flow_dir_ls1 = r"D:\ms gme\thesis\final parameters\ca\data\flow_dir_ls1.tif"
    ls1_raster = r"D:\ms gme\thesis\final parameters\ca\data\ls1_raster.tif"
    lsi_ls1 = r"D:\ms gme\thesis\final parameters\ca\data\MGD_lsi2_ls1_clip.tif"
    initial_ls1 = r"D:\ms gme\thesis\final parameters\ca\data\ls1_initial_same_extent.tif"

    # Initialize factors
    growth_factors = Factors(flow_dir=flow_dir_ls1, lsi=lsi_ls1)

    # Initialize CA
    kernel = 3  # 3x3 moore neighborhood
    ca_model = Cellular_Automata(growth_factors, kernel)
    ca_model.set_threshold(actual_ls=ls1_raster, initial_cell=initial_ls1)
    ca_model.simulate()

    # Export
#     out_fn = f"D:/ms gme/thesis/final parameters/ca/runs/ls1_{kernel}x{kernel}.tif"
#     ca_model.export_predicted(out_fn)

#     GIF
#     tif_path = r"D:\ms gme\thesis\final parameters\ca\runs"
#     create_png(tif_path)

    png_path = r"D:\ms gme\thesis\final parameters\ca\runs\png"
    fn = "ls1.gif"
    create_gif(png_path, fn)
