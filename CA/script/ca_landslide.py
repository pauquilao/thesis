#!/usr/bin/env python3
# Author: Paulo Quilao
# MS Geomatics Engineering Thesis
# Simulation of landslides using Cellular Automata


import os
import gdal
import numpy as np


# Helper function
def read_raster(file):
    dataSource = gdal.Open(file)
    band = dataSource.GetRasterBand(1).ReadAsArray()
    return (dataSource, band)

# end of helper function


class Factors:
    def __init__(self, **params):   # keys are flow_dir, lsi, and initial_cell
        self.flow_dir_ds, self.flow_dir_arr = read_raster(params["flow_dir"])
        self.lsi_ds, self.lsi_arr = read_raster(params["lsi"])
        self.initial_ls = read_raster(params["initial_cell"])[1]  # get only the array
        self.check_shape()

    def check_shape(self):
        print("Checking shapes of input growth factors...")
        if self.flow_dir_arr.shape == self.lsi_arr.shape == self.initial_ls.shape:
            self.row = self.flow_dir_arr.shape[0]
            self.col = self.flow_dir_arr.shape[1]
            print("All factors have the same shape.")
        else:
            print("Shape of factors does not match.")


class Cellular_Automata:
    def __init__(self, factors, neigh_size):
        self.factors = factors
        self.neigh_size = neigh_size

    def simulate(self):
        print("\nSimulating landslide...")
        # landslide base
        self.landslide_predicted = np.zeros_like(self.factors.initial_ls)

        col = self.neigh_size - 1
        center = int((self.neigh_size - 2) - (self.neigh_size - 3) / 2)

        for i in range(1, self.factors.row - 1):
            # skip no data values
            if self.factors.initial_ls[i].mean() == 0:
                continue
            for j in range(1, len(self.factors.initial_ls[i]) - 1):
                try:
                    # flow dir kernel
                    kernel_fd = flow_dir[i-1:i+col, j-1:j+col]
                    # LSI kernel
                    kernel_LSI = LSI[i-1:i+col, j-1:j+col]

                    # kernel where means will be computed
                    kernel_no_center_fd = kernel_fd.copy()
                    kernel_no_center_LSI = kernel_LSI.copy()
                    kernel_no_center_fd[center, center] = 0
                    kernel_no_center_LSI[center, center] = 0

                    mean_fd = kernel_no_center_fd.mean()
                    mean_LSI = kernel_no_center_LSI.mean()
                    # Rule for landslide transition
                    if mean_fd <= 5 and mean_fd >= 3.5 and mean_LSI <= 5 and mean_LSI >= 3.5:
                        self.landslide_predicted[i][j] = 1

                # if raster dimension is not divisible by the kernel size
                except IndexError:
                    continue

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

    # Initialize factors
    growth_factors = Factors(flow_dir=flow_dir_ls1, lsi=lsi_ls1, initial_cell=ls1_raster)

    # Initialize CA
    kernel = 3
    ca_model = Cellular_Automata(growth_factors, kernel)

    ca_model.simulate()

    # Export
    out_fn = f"D:/ms gme/thesis/final parameters/ca/data/output/ls1_3x3_new.tif"
    ca_model.export_predicted(out_fn)


