#!usr/bin/env python3
# Author: Paulo Quilao
# MS Geomatics Engineering Thesis
# Module for prepartion of datasets for neural network


# Import necessary packages
import os
import glob
import gdal
import numpy as np


# Helper functions
def convert_layer(filepath) -> np.ndarray:
    """Converts image file to an array."""
    try:
        # Read params using gdal and convert band 1 to np array
        image = gdal.Open(filepath)
        if image is None:
            print("Cannot locate image.")
            return
        # Convert raster to numpy array
        layer = image.GetRasterBand(1).ReadAsArray()

    except Exception as e:
        print(repr(e))

    finally:
        # Close dataset
        image = None

    return layer


def get_path_ds(filepath, extension):
    """Returns path of tif image."""
    ds_path = glob.glob(os.path.join(filepath, extension))
    # assert len(ds_path) == 9, "Path should contain 9 files"
    return ds_path


def get_relevant_val(array):
    """Returns an array with nontrivial scalars."""
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    array = array[array > -99]
    return array

# end of helper functions


# Class for preparation of datasets
class Dataset:
    def __init__(self, prone, notprone):
        self.prone = prone
        self.notprone = notprone
        self.file_ext = "*.tif"

    def set_file_ext(self, extension):
        """Alters the default image file extension (i.e. .tif)."""
        self.file_ext = extension

    def load_layers(self):
        """Prepare layers for neural network model."""

        # Read all landslide drivers
        print("Loading layers for the creation of input vector...")
        print()

        # Path to datasets
        ds_prone = get_path_ds(self.prone, self.file_ext)
        ds_np = get_path_ds(self.notprone, self.file_ext)

        # Prone datasets
        X_p = []
        for image in ds_prone:
            X_p.append(convert_layer(image))

        # Not prone datasets
        X_np = []
        for image in ds_np:
            X_np.append(convert_layer(image))

        # Check shape of each array if read correctly
        # Expected output:(6334, 3877)
        # If the list is not empty
        if X_p and X_np:
            print("Shape of converted layers to numpy array:")
            flag = True
            for factor1, factor2 in zip(X_p, X_np):
                assert factor1.shape == factor2.shape, "Vectorized images should have the same shape."
                flag = True
        if flag:
            print("prone factor: {}, not prone factor: {}".format(X_p[0].shape, X_np[0].shape))
            print()

        else:
            print("Error in input images.")

        # Convert arrays to row vector
        # Landslide prone samples
        X_p_flat = []
        for factor in X_p:
            X_p_flat.append(factor.flatten().astype("float32"))

        # Not prone to landslide samples
        X_np_flat = []
        for factor in X_np:
            X_np_flat.append(factor.flatten().astype("float32"))

        # Get flattened shape and dtype
        if X_p_flat and X_np_flat:
            print("Converting arrays to 1D arrays...")
            flag = True
            for vector1, vector2 in zip(X_p_flat, X_np_flat):
                assert vector1.shape == vector2.shape, "Row vectors should have the same shape."
                flag = True

        if flag:
            print("Shape of 1D arrays:")
            print(f"prone factor: {X_p_flat[0].shape}, dtype: {X_p_flat[0].dtype} | not prone factor: {X_np_flat[0].shape}, dtype: {X_np_flat[0].dtype}")
            print()

        # Boolean filtering
        # Get "non-nodata" values for each factor
        print("Applying boolean filtering to 1D arrays...")
        for i in range(len(X_p_flat)):
            X_p_flat[i] = get_relevant_val(X_p_flat[i])
            X_np_flat[i] = get_relevant_val(X_np_flat[i])

        # Add columns to those factors with less than columns of relevant dataset
        for i in range(len(X_np_flat)):
            ref = X_p_flat[0].shape[0]
            col = X_np_flat[i].shape[0]
            if col != ref:
                X_np_flat[i] = np.concatenate((X_np_flat[i], [np.mean(X_np_flat[i])] * (ref - col)))

        # Get filtered shape and dtype
        if X_p_flat and X_np_flat:
            flag = True
            for vector1, vector2 in zip(X_p_flat, X_np_flat):
                assert vector1.shape == vector2.shape, "Row vectors should have the same shape."
                flag = True

        if flag:
            print("Shape of filtered 1D arrays:")
            print(f"prone factor: {X_p_flat[0].shape}, dtype: {X_p_flat[0].dtype} | not prone factor: {X_np_flat[0].shape}, dtype: {X_np_flat[0].dtype}")
            print()

        # Scale values of each row vector
        print("Scaling all input vectors using min-max scaling...")
        X_p_norm = []
        X_np_norm = []
        for i in range(len(X_p)):
            norm_p = (X_p_flat[i] - min(X_p_flat[i])) / (max(X_p_flat[i]) - min(X_p_flat[i]))
            norm_np = (X_np_flat[i] - min(X_np_flat[i])) / (max(X_np_flat[i]) - min(X_np_flat[i]))

            X_p_norm.append(norm_p)
            X_np_norm.append(norm_np)
        print("All vectors are scaled in the interval [0, 1].")
        print()

        # Create input vector X from all parameters including bias
        print("Creating X vector with bias...")
        bias_p = np.ones(X_p_norm[0].shape[0])
        bias_np = np.ones(X_np_norm[0].shape[0])

        X_p = np.vstack([X_p_norm, bias_p]).T.astype("float32")
        X_np = np.vstack([X_np_norm, bias_np]).T.astype("float32")

        print(f"Landslide prone input vector: {X_p.shape} and data type: {X_p.dtype} \nNot prone to landslide input vector: {X_np.shape} and data type {X_np.dtype}")
        print()

        # Split dataset to training set and testing set
        print("Splitting samples to training, validation, and testing sets...")
        train_split = int(0.8 * X_p.shape[0])  # 80% of total dataset

        X_p_split = np.split(X_p, [train_split])
        X_p_train = X_p_split[0]  # get the 80% of the dataset
        X_p_test = X_p_split[1]  # get the remaining 80% of the dataset

        X_np_split = np.split(X_np, [train_split])
        X_np_train = X_np_split[0]
        X_np_test = X_np_split[1]

        X_train = np.vstack([X_p_train, X_np_train])
        X_test = np.vstack([X_p_test, X_np_test])

        # These target values resulted to continuous lsi (probability between 0.1 to 0.9)
        # For one output node with 0.1 and 0.9 target values
        # 0.1 -> absence of landslide; 0.9 -> presence of landslide
        y_p_train = np.full(X_p_train.shape[0], 0.9, dtype="float32")
        y_p_test = np.full(X_p_test.shape[0], 0.9, dtype="float32")

        y_np_train = np.full(X_np_train.shape[0], 0.1, dtype="float32")
        y_np_test = np.full(X_np_test.shape[0], 0.1, dtype="float32")

        y_train = np.concatenate([y_p_train, y_np_train])
        y_train = np.vstack(y_train)
        y_test = np.concatenate([y_p_test, y_np_test])
        y_test = np.vstack(y_test)

        # Check if the total number of dataset matched with training and testing sets
        assert (X_train.shape[0] + X_test.shape[0]) == (X_p.shape[0] + X_np.shape[0]), "Number of rows should match the total number of datasets."
        assert (X_train.shape[1] == X_p.shape[1]) and (X_test.shape[1] == X_p.shape[1]), "Number of columns should match the total number of datasets."

        print("Finalizing samples...")
        # Create a union of X and y vectors
        dataset_train = np.concatenate((X_train, y_train), axis=1)
        dataset_test = np.concatenate((X_test, y_test), axis=1)

        # Shuffle datasets
        np.random.shuffle(dataset_train)
        np.random.shuffle(dataset_test)

        # For 1 output node
        X_train_1o = dataset_train[..., :10]
        y_train_1o = dataset_train[..., [-1]]

        # Split train into train and validation sets
        # 80:20 ratio
        train_val_split = int(0.8 * X_train_1o.shape[0])
        X_train_split = np.split(X_train_1o, [train_val_split])
        y_train_split = np.split(y_train_1o, [train_val_split])

        # X vector
        X_train_1o = X_train_split[0]
        X_val_1o = X_train_split[1]

        # y vector
        y_train_1o = y_train_split[0]
        y_val_1o = y_train_split[1]

        # X and y test sets
        X_test_1o = dataset_test[..., :10]
        y_test_1o = dataset_test[..., [-1]]

        if X_train_1o.shape[0] == y_train_1o.shape[0] and X_test_1o.shape[0] == y_test_1o.shape[0] and X_val_1o.shape[0] == y_val_1o.shape[0]:
            print("Datasets created with shape: train={}, val={}, test={}".format(X_train_1o.shape, X_val_1o.shape, y_test_1o.shape))
            print()

        else:
            print("Error in dataset shapes.")

        # For one output node
        # Merge final train, test, and validation datasets (excluding validation set)
        # For shuffling in training regimen
        train_xy = np.concatenate([X_train_1o, y_train_1o], axis=1)
        val_xy = np.concatenate([X_val_1o, y_val_1o], axis=1)
        test_xy = np.concatenate([X_test_1o, y_test_1o], axis=1)

        return [train_xy, val_xy, test_xy]

    def load_fuzzified_layers(self, path):
        """Prepare arrays for lsi generation"""
        lsi_path = get_path_ds(path, self.file_ext)

        # Read all landslide drivers
        X_raw = []
        for factor in lsi_path:
            X_raw.append(convert_layer(factor))

        # Convert to 1D array
        X_lsi_flat = []
        for array in X_raw:
            X_lsi_flat.append(array.flatten())

        # Create input vector X from all contributing factors
        x_lsi = np.vstack(X_lsi_flat).T.astype("float32")
        # Finalize datasets
        y_lsi = x_lsi[..., [-1]]
        x_lsi = x_lsi[..., :-1]

        return [x_lsi, y_lsi]

# end of Dataset class


def main():
    # Filepath for landslide prone and nonprone samples
    fp_prone = r"D:\ms gme\thesis\final parameters\samples\Final\landslide"
    fp_notprone = r"D:\ms gme\thesis\final parameters\Samples\Final\no_landslide"

    my_thesis = Dataset(fp_prone, fp_notprone)
    datasets = my_thesis.load_layers()

    return datasets


if __name__ == "__main__":
    pass
    # main()
