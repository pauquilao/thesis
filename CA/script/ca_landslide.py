#!/usr/bin/env python3
# Author: Paulo Quilao
# MS Geomatics Engineering Thesis
# Simulation of landslides using Cellular Automata


import os
import re
import glob
import time
import imageio
import gdal
import fiona
import rasterio
import rasterio.mask
import numpy as np


def read_raster(file):
    """Read image file and returns its data source and converted numpy array."""
    data_source = gdal.Open(file)
    band = data_source.GetRasterBand(1).ReadAsArray()
    return (data_source, band)


def create_png(path):
    """Creates PNG images based on GeoTiff files."""
    folder = "png"
    op = os.path.join(path, folder)
    if not os.path.exists(op):
        os.makedirs(op)

    search = os.path.join(path, "*mask.tif")
    tif_paths = glob.glob(search)
    # Arrange the paths per row
    regex = re.compile(r"\d+")  # this will match the suffix row
    sorted_tif_paths = sorted(tif_paths, key=lambda p: int(regex.search(p).group(0)))
    options_list = ["-ot Byte", "-of PNG", "-b 1", "-scale", "-outsize 1000% 1000%"]
    options_string = " ".join(options_list)
    for tif in sorted_tif_paths:
        out_fn = f"ls_pred_{regex.findall(tif)[1]}_mask.png"
        gdal.Translate(os.path.join(op, out_fn), tif, options=options_string)

    if os.path.isdir(op):
        print("\nDone converting tif image to png file.")
    else:
        print("\nFailed to convert tif images to png.")


def create_gif(path, out_fn):
    image_paths = glob.glob(os.path.join(path, r"*mask.png"))
    output_gif_path = os.path.join(path, out_fn)
    # Arrange the paths per row
    regex = re.compile(r"\d+")  # this will match the suffix row
    sorted_path = sorted(image_paths, key=lambda p: int(regex.findall(p)[1]))  # findall was used since sub folder is named run_{number}
    # Save the animation to disk with 48 ms durations
    imageio.mimsave(output_gif_path, [imageio.imread(fp) for fp in sorted_path], duration=0.48, subrectangles=True)
    if os.path.isfile(output_gif_path):
        print("\nGIF successfully exported.")
    else:
        print("\nFailed to create GIF.")


def clip_raster(polygon, raster, out_path, crop=False):
    with fiona.open(polygon, "r") as shapefile, rasterio.open(raster) as src:
        mask = [feature["geometry"] for feature in shapefile]
        out_image, out_transform = rasterio.mask.mask(src, mask, crop=crop)  # out_image -> numpy array
        out_meta = src.meta
        if crop:
            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})

    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(out_image)

    # Check if exported successfully
    if os.path.isfile(out_path):
        print("\nRaster clipped successfully.")

    else:
        print("\nFailed to clip raster.")


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
            print("Input factors have the same shape.")
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

        # Mask to exclude center pixel when computing thresholds
        m = np.isin(kernel, [0])

        # Flow direction threshold
        self.fd_threshold_low = np.where(m, kernel, kernel - 1).mean()  # low thresh flow_dir
        self.fd_threshold_hi = np.where(m, kernel, kernel + 1).mean()   # high thresh flow_Dir

        # LSI threshold
        kernel_items = np.prod(self.neigh_size)
        self.lsi_threshold_hi = (4 * (kernel_items - 1)) / kernel_items  # high lsi, less 1 for the center pixel
        self.lsi_threshold_vhi = (5 * (kernel_items - 1)) / kernel_items  # very high lsi, less 1 for the center pixel

    def simulate(self, export=False, run=None):
        start_time = time.time()
        """Implement CA using the growth factors and set thresholds."""
        print("\nSimulating landslide...")
        # landslide base
        self.landslide_predicted = np.zeros_like(self.factors.lsi_arr)

        col = self.neigh_size[0] - 1
        first_ite = 0  # counter for the creation of sub folder for the 1st row with no data val
        for i in range(1, self.factors.row - 1):
            # skip no data values
            test_elem = range(1, 8)
            if np.isin(self.actual_ls[i], test_elem).sum() == 0 and i != (self.index[0][0] - 1):  # to prevent the row of center pixel - 1 to be skipped
                continue
            for j in range(1, self.factors.col - 1):
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
                    if self.fd_threshold_low < mean_fd <= self.fd_threshold_hi and self.lsi_threshold_hi <= mean_LSI <= self.lsi_threshold_vhi:
                        self.landslide_predicted[i][j] = 1

                # If raster dimension is not divisible by the kernel size
                except IndexError:
                    continue

            # Export image for each row
            if export:
                main_folder = "runs"
                op = os.path.join(os.getcwd(), main_folder)
                if not os.path.exists(op):
                    os.makedirs(op)

                if first_ite == 0:
                    # Create sub folder everytime export is invoked
                    sub_folder = f"run_{run}"
                    rel_path = os.path.join(os.getcwd(), main_folder, sub_folder)
                    if os.path.isdir(rel_path):
                        regex = re.compile(r"\d+$")
                        folder_num = regex.search(rel_path).group(0)
                        new_folder = rel_path.strip(folder_num) + str(int(folder_num) + 1)  # increment the folder num by 1
                        rel_path = new_folder
                        os.makedirs(rel_path)
                        first_ite += 1
                    else:
                        os.makedirs(os.path.join(rel_path))
                        first_ite += 1

                self.export_predicted(os.path.join(rel_path, "ls_pred" + str(i) + ".tif"))

        end_time = time.time() - start_time
        display = f"\nFinished implementing CA after {end_time:.2f} seconds." if export else f"Finished implementing CA after {end_time:.2f} seconds."
        print(display)
        return self.landslide_predicted

    def check_accuracy(self):
        # Actual ls
        actual_ls_pixels = np.count_nonzero(self.actual_ls == 1)
        # Correct predictions
        intersection = self.actual_ls * self.landslide_predicted
        correct_pixels = np.count_nonzero(intersection == 1)
        percentage = (correct_pixels / actual_ls_pixels) * 100
        print("-" * 57)  # divider
        print(f"Actual landslide pixels: {actual_ls_pixels} | Correctly identified: {correct_pixels}")
        print(f"Accuracy of simulated landslide: {percentage:.2f}%")

    def export_predicted(self, out_fn):
        print("\nExporting array to GTiff...")
        driver = gdal.GetDriverByName("GTiff")
        out_data = driver.Create(out_fn, self.factors.col, self.factors.row, 1, gdal.GDT_Int16)
        out_data.SetGeoTransform(self.factors.lsi_ds.GetGeoTransform())
        out_data.SetProjection(self.factors.lsi_ds.GetProjection())
        out_data.GetRasterBand(1).WriteArray(self.landslide_predicted)
        out_data.GetRasterBand(1).SetNoDataValue(-999)
        out_data.FlushCache()
        out_data = None

        # check if exported
        if os.path.isfile(out_fn):
            print("Simulated landslide exported.")
        else:
            print("Failed to export raster.")


if __name__ == "__main__":
    ls_id = 1
    flow_dir = f"D:\\ms gme\\thesis\\final parameters\\ca\\data\\ls{ls_id}_flow_dir.tif"
    ls_raster = f"D:\\ms gme\\thesis\\final parameters\\ca\\data\\ls{ls_id}_node_raster.tif"
    lsi_ls = f"D:\\ms gme\\thesis\\final parameters\\ca\\data\\ls{ls_id}_lsi.tif"
    initial_cell = f"D:\\ms gme\\thesis\\final parameters\\ca\\data\\ls{ls_id}_initial_cell.tif"

    # Initialize factors
    factors = {"flow_dir": flow_dir, "lsi": lsi_ls}
    growth_factors = Factors(**factors)

    # Initialize CA and kernel size
    kernel = 3  # 3x3 moore neighborhood
    ca_model = Cellular_Automata(growth_factors, kernel)

    # Set thresholds
    thresholds = {"actual_ls": ls_raster, "initial_cell": initial_cell}
    ca_model.set_threshold(**thresholds)

    # Perform simulation
    run_num = 5
    ca_model.simulate()
    ca_model.check_accuracy()

    # Export predicted landslide
    out_fn = f"D:/ms gme/thesis/final parameters/ca/runs/exported/ls{ls_id}_{kernel}x{kernel}.tif"
    ca_model.export_predicted(out_fn)

    # Mask simulated landslide
    poly = f"D:/MS Gme/Thesis/Final Parameters/CA/data/landslide/ls{ls_id}.shp"
    image = f"D:/MS Gme/Thesis/Final Parameters/CA/runs/exported/ls{ls_id}_{kernel}x{kernel}.tif"
    op = f"D:/MS Gme/Thesis/Final Parameters/CA/runs/exported/masked/ls{ls_id}_{kernel}x{kernel}_mask.tif"

    clip_raster(poly, image, op)

    # Optional: for the creation of gif
    # Mask resulting raster of each iteration with the given landslide polygon
    # path = f"D:/ms gme/thesis/final parameters/ca/runs/run_{run_num}"
    # tif_path = glob.glob(os.path.join(path, "*.tif"))

    # for image in tif_path:
    #     op = image.strip(".tif") + "_mask" + ".tif"
    #     clip_raster(poly, image, op)

    # # Create png of masked images
    # create_png(path)

    # # Create a GIF from png images
    # png_path = f"D:/ms gme/thesis/final parameters/ca/runs/run_{run_num}/png"
    # fn = f"ls{ls_id}_mask.gif"
    # create_gif(png_path, fn)
