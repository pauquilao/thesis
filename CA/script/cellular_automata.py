#!/usr/bin/env python3
# Author: Paulo Quilao
# MS Geomatics Engineering Thesis
# Simulation of landslides using Cellular Automata


import os
import re
import glob
import time
import datetime
import imageio
import gdal
import fiona
import rasterio
import rasterio.mask
import numpy as np


# Helper functions
def read_raster(file):
    """Reads image file and returns its data source and converted numpy array."""
    data_source = gdal.Open(file)
    band = data_source.GetRasterBand(1).ReadAsArray()
    return (data_source, band)


def create_png(path):
    """Creates PNG images based on GeoTiff files."""
    folder = "png"
    op = os.path.join(path, folder)
    if not os.path.exists(op):
        os.makedirs(op)

    search = os.path.join(path, "*.tif")
    tif_paths = glob.glob(search)
    # Arrange the paths per row
    regex = re.compile(r"\d+")  # this will match the suffix row
    sorted_tif_paths = sorted(tif_paths, key=lambda p: int(regex.search(p).group(0)))
    options_list = ["-ot Byte", "-of PNG", "-b 1", "-scale", "-outsize 1000% 1000%"]
    options_string = " ".join(options_list)
    for tif in sorted_tif_paths:
        out_fn = f"ls_pred_{regex.findall(tif)[-1]}.png"
        gdal.Translate(os.path.join(op, out_fn), tif, options=options_string)

    if os.path.isdir(op):
        print("\nDone converting tif image to png file.")
    else:
        print("\nFailed to convert tif images to png.")


def create_gif(path, out_fn, reverse=False):
    """Generates GIF based on series of passed images."""
    print("\nCreating GIF...")
    image_paths = glob.glob(os.path.join(path, r"*.png"))
    output_gif_path = os.path.join(path, out_fn)
    # Arrange the paths per row
    regex = re.compile(r"\d+")  # this will match the suffix row
    sorted_path = sorted(image_paths, key=lambda p: int(regex.findall(p)[1]), reverse=reverse)  # findall was used since sub folder is named run_{number}
    # Save the animation to disk with 48 ms durations
    imageio.mimsave(output_gif_path, [imageio.imread(fp) for fp in sorted_path], duration=0.48, subrectangles=True)
    if os.path.isfile(output_gif_path):
        print("GIF successfully exported.")
    else:
        print("\nFailed to create GIF.")


def clip_raster(polygon, raster):
    """Masks raster using polygon coverage."""
    with fiona.open(polygon, "r") as shapefile, rasterio.open(raster) as src:
        mask = [feature["geometry"] for feature in shapefile]
        out_image, out_transform = rasterio.mask.mask(src, mask)  # out_image -> numpy array

    return out_image

# end of helper functions


class Factors:
    def __init__(self, **params):   # keys are flow_dir, and lsi
        self.lsi_arr = clip_raster(params["extent"], params["lsi"])[0]
        self.flow_dir_arr = clip_raster(params["extent"], params["flow_dir"])[0]
        self.check_shape()

    def check_shape(self):
        """Validates shape of input factors."""
        print("Checking shapes of input growth factors...")
        if self.flow_dir_arr.shape == self.lsi_arr.shape:
            self.row = self.flow_dir_arr.shape[0]
            self.col = self.flow_dir_arr.shape[1]
            print("Input factors have the same shape.")
        else:
            raise AttributeError("Shape of factors does not match.")


class Cellular_Automata:
    def __init__(self, factors, neigh_size):
        self.factors = factors
        self.neigh_size = tuple([neigh_size] * 2)  # kernel dimension
        self.dt = datetime.datetime.now().strftime("%x %X")

    def set_threshold(self, **params):
        """
        Computes the flow direction and lsi thresholds to be considered in the moore neighborhood.
        Params: actual landslide, initial cell, and limit (keys: "actual_ls", "initial_cell", limit")
        """
        self.data_source, self.initial_cell = read_raster(params["initial_cell"])
        self.initial_cell = self.initial_cell.astype(int)

        assert self.initial_cell.shape == self.factors.lsi_arr.shape, "Shape of initial cell should match shape of factors."

        # Get the center pixel of the kernel
        cent = int((self.neigh_size[0] - 2) - (self.neigh_size[0] - 3) / 2)
        self.center = tuple([cent] * 2)
        self.index = np.where(self.initial_cell == 1)

        # Flow dir path
        self.flow_path = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1), 4: (0, -1), 5: (-1, -1), 6: (-1, 0), 7: (-1, 1)}

        # LSI threshold
        self.lsi_threshold = params["lsi_thresh"]

    def simulate(self, export=False, run=None):
        """Implement CA using the growth factors and set thresholds."""
        start_time = time.time()
        print("\nSimulating landslide...")
        # landslide base
        self.landslide_predicted = np.full_like(self.factors.lsi_arr, 0)
        self.landslide_predicted[self.index] = 1
        row = col = self.neigh_size[0] - 1  # row and col increment every iteration
        first_ite = 0  # counter for the creation of sub folder for the 1st row with data val
        iteration = 0  # time step counter

        # Orientation of landslide
        if self.index[0][0] < (self.factors.row) // 2:  # failure originates from North
            orientation = range(self.index[0][0] + 1, self.factors.row - 1)
        else:
            orientation = range(self.index[0][0] + 1, 0, -1)  # failure originates from South

        for i in orientation:
            if np.isin(self.landslide_predicted[i-1], [1]).sum():  # -1 to start in the next row after a row with landslide pixel
                for j in range(1, self.factors.col - 1):
                    try:
                        # Flow dir kernel
                        kernel_fd = self.factors.flow_dir_arr[i-1:i+row, j-1:j+col]
                        kernel_fd[self.center] = -1
                        # LSI kernel
                        kernel_LSI = self.factors.lsi_arr[i-1:i+row, j-1:j+col]
                        # Predicted landslide
                        kernel_pred_ls = self.landslide_predicted[i-1:i+row, j-1:j+col]

                        # Mean thresholds
                        mean_LSI = np.delete(kernel_LSI.flatten(), (np.prod(self.neigh_size) // self.neigh_size[0] + 1), 0).mean()
                        mean_rf = np.delete(kernel_rf.flatten(), (np.prod(self.neigh_size) // self.neigh_size[0] + 1), 0).mean()

                        # Check each element in the flow dir kernel
                        # for suitable flow
                        for val in np.ndenumerate(kernel_fd):
                            row_ = val[0][0]
                            col_ = val[0][1]
                            if val[1] in self.flow_path:
                                k = self.flow_path[val[1]][0]
                                l = self.flow_path[val[1]][1]
                                try:
                                    #  If a possible flow is seen
                                    # stop the loop
                                    while kernel_fd[row_, col_] != -1:
                                        flow = kernel_fd[row, col]
                                        k = self.flow_path[flow][0]
                                        l = self.flow_path[flow][1]
                                        row_ += k
                                        col_ += l
                                        if np.isin(kernel_pred_ls, [1]).sum() >= 1:
                                            if kernel_fd[row_, col_] == -1:
                                                if mean_LSI >= 0.75:
                                                    self.landslide_predicted[i, j] = 1
                                                else:
                                                    self.landslide_predicted[i, j] = 0

                                except (IndexError, KeyError):
                                    continue
                                break  # if a suitable flow is already seen
                    # If raster dimension is not divisible by the kernel size
                    except IndexError:
                        continue

                iteration += 1

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

                fn = f"ls_{row+1}x{col+1}_pred_" + str(i) + ".tif"
                self.export_predicted(os.path.join(rel_path, fn))

        end_time = time.time() - start_time
        display = f"\nFinished implementing CA after {end_time:.2f} seconds." if export else f"Finished implementing CA after {end_time:.2f} seconds."
        print(display)
        print(f"Number of iterations: {iteration}")
        return self.landslide_predicted

    def check_accuracy(self):
        """Returns spatial accuracy of simulated landslide."""
        # Actual ls
        actual_ls_pixels = np.count_nonzero(self.actual_ls == 1)
        # Simulated ls
        sim_ls_pixels = np.isin(self.landslide_predicted, [1, 0]).sum()
        # Correct predictions
        intersection = self.actual_ls * self.landslide_predicted
        correct_pixels = np.count_nonzero(intersection == 1)
        percentage = (correct_pixels / (actual_ls_pixels + (abs(sim_ls_pixels - correct_pixels)))) * 100
        print("-" * 57)  # divider
        self.summary = f"Actual: {actual_ls_pixels} | Simulated: {sim_ls_pixels} | TP: {correct_pixels}"
        self.accuracy = f"Accuracy of simulated landslide: {percentage:.2f}%"
        print(self.summary)
        print(self.accuracy)

    def save_summary(self, landslide_id, filename):
        """Exports information of simulated landslide to a text file."""
        current_dir = os.getcwd()
        folders = r"runs\summary"
        out_path = os.path.join(current_dir, folders)
        if not os.path.isdir(folders):
            os.makedirs(out_path)
        with open(os.path.join(out_path, filename), "a") as out_file:
            try:
                out_file.write(self.dt + "\n")
                out_file.write(f"Landslide ID: {str(landslide_id)}\n")
                out_file.write(f"Neighborhood size: {self.neigh_size} Threshold: +-{self.limit}\n")
                out_file.write(self.summary + "\n")
                out_file.write(self.accuracy + "\n\n")

            except Exception:
                raise AttributeError("Cannot export summary unless save_summary method was invoked.")

    def export_predicted(self, out_fn):
        """Exports simulated landslide to a GeoTiff raster."""
        print("\nExporting array to GTiff...")
        driver = gdal.GetDriverByName("GTiff")
        out_data = driver.Create(out_fn, self.factors.col, self.factors.row, 1, gdal.GDT_Int16)
        out_data.SetGeoTransform(self.data_source.GetGeoTransform())
        out_data.SetProjection(self.data_source.GetProjection())
        out_data.GetRasterBand(1).WriteArray(self.landslide_predicted)
        out_data.GetRasterBand(1).SetNoDataValue(-999)
        out_data.FlushCache()
        out_data = None

        # check if exported
        if os.path.isfile(out_fn):
            print("Simulated landslide exported.")
        else:
            print("Failed to export raster.")

    def get_shape(self):
        print("factors", self.factors.row, self.factors.col)
        print("initial cell", self.initial_cell.shape)


if __name__ == "__main__":
    ls_id = "kating"
    flow_dir = "D:\\ms gme\\thesis\\final parameters\\ca\\Hypothetical_LS\\itogon_flow_dir.tif"
    lsi = "D:\\ms gme\\thesis\\final parameters\\ca\\Hypothetical_LS\\itogon_lsi.tif"
    initial_cell = f"D:\\ms gme\\thesis\\final parameters\\ca\\Hypothetical_LS\\{ls_id}_initial_cell.tif"
    bounds = f"D:\\ms gme\\thesis\\final parameters\\ca\\Hypothetical_LS\\shp\\{ls_id}_hypo_extent.shp"

    # Initialize factors
    factors = {"flow_dir": flow_dir, "lsi": lsi, "extent": bounds}
    growth_factors = Factors(**factors)

    # Initialize CA and kernel size
    kernel = 3  # 3x3 moore neighborhood
    ca_model = Cellular_Automata(growth_factors, kernel)

    # Set thresholds
    lsi_threshold = 0.75  # user-defined lsi treshold for to be considered in the kernel
    thresholds = {"initial_cell": initial_cell, "lsi_thresh": lsi_threshold}
    ca_model.set_threshold(**thresholds)

    # Perform simulation
#     runs = "D:/ms gme/thesis/final parameters/ca/runs"
#     r_path = [folder for folder in os.listdir(runs) if not os.path.isfile(folder) and folder.startswith("run")]
#     run_num = int(r_path[-1][-1]) + 1
#     exports = {"export": True, "run": run_num}  # optional, if export is set to True
    ca_model.simulate()

    # Export predicted landslide
    image = f"D:\\ms gme\\thesis\\final parameters\\ca\\Hypothetical_LS\\outputs\\{ls_id}_{kernel}x{kernel}.tif"
    ca_model.export_predicted(image)
