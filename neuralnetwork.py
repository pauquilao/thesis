#!/usr/bin/env python3
# Author: Paulo Quilao
# MS Geomatics Engineering Thesis
# Simulation of landslides using a two-layer neural network


# Import necessary packages
import os
import sys
import datetime
import time
import copy
import gdal
import numpy as np
import matplotlib.pyplot as plt
from dataset import *  # user-defined module


# Neural network parent class
class Neural_Network(object):
    def __init__(self, *args, **kwargs):
        # Layers
        self.layers = args

        self.input_size = self.layers[0] + 1  # + 1 for bias
        self.hidden_size = self.layers[1]  # number of hidden neurons
        self.output_size = self.layers[2]  # number of output neurons

        # Parameters
        self.W1 = np.random.uniform(0.1, 0.3, size=(self.input_size, self.hidden_size))
        self.W2 = np.random.uniform(0.1, 0.3, size=(self.hidden_size, self.output_size))

        # Hyperparameters
        if len(kwargs.values()) == 1:
            if "lr" in kwargs:
                self.lrate = kwargs["lr"]
            else:
                # Default value for lrate
                self.lrate = 0.01
                self.momentum = kwargs["mu"]

        elif len(kwargs.values()) > 1:
            self.lrate = kwargs["lr"]
            self.momentum = kwargs["mu"]

        self.cache_W1 = np.zeros(self.W1.shape)  # holds input-to-hidden weights for momentum
        self.cache_W2 = np.zeros(self.W2.shape)  # holds hidden-to-output weights for momentum

        # Date/time information for exporting
        dt = datetime.datetime.now()
        self.date = dt.strftime("%x")  # local date
        self.time = dt.strftime("%X")  # local time

    # Training methods
    # Activation function in output layer
    def sigmoid(self, s):
        """Logistic activation function."""
        return 1 / (1 + np.exp(-s))

    def sigmoid_prime(self, s):
        """Derivative of logistic function."""
        return self.sigmoid(s) * (1 - self.sigmoid(s))

    # Activation function in hidden layer
    def lrelu(self, x):
        """Leaky rectified linear unit."""
        return np.where(x > 0, x, x * 0.01)

    def lrelu_prime(self, x):
        """Derivative of leaky rectified linear unit."""
        dx = np.ones_like(x)
        dx[x < 0] = 0.01
        return dx

    def forward(self, X):
        """Forward propagation through the network."""
        self.z = np.dot(X, self.W1)
        self.z2 = self.lrelu(self.z)  # activation on hidden layer
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)  # final activation on output layer
        return o

    def backward(self, X, y, o):
        """Backward propagation to opmtimize weights."""
        self.o_error = y - o  # loss function
        self.o_delta = self.o_error * self.sigmoid_prime(o)  # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.lrelu_prime(self.z2)  # applying derivative of sigmoid to z2 error

        # Convert to 2d array to handle 1x1 vectors
        X = np.atleast_2d(X)
        self.z2_delta = np.atleast_2d(self.z2_delta)

        # Momentum update to input-to-hidden weights
        dw1 = (X.T.dot(self.z2_delta) * self.lrate) + (self.momentum * self.cache_W1)  # delta input to hidden weights for momentum
        self.W1 += dw1

        # Convert to 2d array to handle 1x1 vectors
        self.z2 = np.atleast_2d(self.z2)
        self.o_delta = np.atleast_2d(self.o_delta)

        # Momentum update to hidden-to-output weights
        dw2 = (self.z2.T.dot(self.o_delta) * self.lrate) + (self.momentum * self.cache_W2)  # delta hidden to output weights for momentum
        self.W2 += dw2

        # Store previous weights
        self.cache_W1 = dw1
        self.cache_W2 = dw2

    def train(self, X, y):
        """Main learning method of the model."""
        o = self.forward(X)
        self.backward(X, y, o)

    def compute_error(self, X, y):
        """Computes RMSE of feeded dataset after a full pass."""
        feed = self.forward(X)
        error = np.sqrt(np.mean((y - feed)**2))
        return error

    def predict(self, test_x, test_y):
        """Simulates y based on feeded testing data."""
        print("Predicted data based on trained weights:")
        print("Predicted values: {}\n".format(self.forward(test_x)))
        print("Target values: {}\n".format(test_y))
        rmse = self.compute_error(test_x, test_y)
        print("Testing loss: {}\n".format(rmse))

    # end of training methods

    # Plot method
    def plot_loss_curve(self, xlabel, ylabel, name, save=True):
        """Creates a loss curve plot."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_train, label="Training Loss")
        plt.plot(self.loss_val, label="Validation Loss")
        plt.title(f"h={self.hidden_size}, lrate={self.lrate}, momentum={self.momentum}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc=1)

        if save:
            # Save plot
            folder = "plots"
            op = os.path.join(os.getcwd(), folder)

            if not os.path.exists(op):
                os.makedirs(folder)

            suffix = f"{str(self.hidden_size)}_{str(self.lrate)}_{str(self.momentum)}"

            if not os.path.isfile(os.path.join(op, name)):
                plt.savefig(os.path.join(op, f"{name}_{suffix}.jpg"))
            else:
                print("Name already exists.")

        plt.show()

    # Write methods
    def save_predict(self, test_x, test_y):
        """Saves a summary of results based on simulated testing data."""
        folder = "runs"
        op = os.path.join(os.getcwd(), folder)

        if not os.path.exists(op):
            os.makedirs(folder)

        # Predicted
        with open(os.path.join(op, "predict.txt"), "a") as pred:
            pred.write("Date and time: {}, {}\n".format(self.date, self.time))
            pred.write("Optimizer: {}\n".format(type(self).__name__))
            pred.write("NN structure: {}\n".format(self.layers))
            pred.write("lrate={}, ".format(self.lrate))
            pred.write("momentum={}, ".format(self.momentum))
            rmse = self.compute_error(test_x, test_y)
            pred.write("error: {}\n\n".format(rmse))

    def save_summary(self, filename, layers: list):
        """Exports the training summary."""
        folder = "runs"
        op = os.path.join(os.getcwd(), folder)

        if not os.path.exists(op):
            os.makedirs(folder)

        # Compute weight of each factor
        m_ave = []
        m_max = []
        weights = self.W1

        for w in weights:
            m_ave.append(np.mean(w))
            m_max.append(np.amax(w))

        # Map weights to factors
        f = layers
        d_ave = dict(zip(f, m_ave[:10]))    # exclude bias, average weight
        d_max = dict(zip(f, m_max[:10]))    # exclude bias, max weight

        # Save weights if error in testing data is reasonable
        with open(os.path.join(op, filename), "a") as data:
            data.write("Date and time: {}, {}\n".format(self.date, self.time))
            data.write("NN structure: {}\n".format(self.layers[:-1]))
            data.write("Hyperparameters: ")
            data.write("lrate={}, ".format(self.lrate))
            data.write("mu={}, ".format(self.momentum))

            try:
                # For MGD
                data.write("batch size={}".format(self.batch_size))
                data.write("\nCycle: {} for 1 epoch (max of {}), ".format(self.total_batch, epoch))
                data.write("terminated at epoch={}, iteration={}".format(self.term_epoch, self.term_iter))

            except AttributeError:
                # For SGD and BGD
                data.write("\nEpoch={}, terminated at:{}".format(self.epoch, self.term_epoch))

            data.write("\nSet and actual errors: ")
            data.write("RMSE: {}, ".format(self.rmse))
            data.write("val error: {}\n".format(np.amin(self.loss_val)))
            data.write("Input to hidden weights:\n {}\n".format(self.W1))
            data.write("Hidden to output weights:\n {}\n".format(self.W2))
            data.write("Weight of each layer (ave):\n {}\n".format(d_ave))
            data.write("Weight of each layer (max):\n {}\n\n".format(d_max))

        # Check if the file exists
        assert os.path.isfile(os.path.join(op, filename)), "The file was not created."
        print("\nSummary of training was exported.")

    def export_to_image(self, ref_path, out_path, array):
        """Exports array to image (default GeoTiff)."""
        print("\nExporting array to image...")

        # Reference data
        data0 = gdal.Open(ref_path)

        if data0 is None:
            print("No reference data.")
            sys.exit(1)

        # get rows and columns of array
        row = array.shape[0]  # number of pixels in y
        col = array.shape[1]  # number of pixels in x

        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(out_path, col, row, 1, gdal.GDT_Float32)

        if dataset is None:
            print("Could not create lsi.tif")
            sys.exit(1)

        dataset.GetRasterBand(1).WriteArray(lsi)
        dataset.GetRasterBand(1).SetNodataValue(-9999)

        # Add GeoTranform and Projection
        geotrans = data0.GetGeoTransform()  # get GeoTranform from ref 'data0'
        proj = data0.GetProjection()  # get GetProjection from ref 'data0'
        dataset.SetGeoTransform(geotrans)
        dataset.SetProjection(proj)
        dataset.FlushCache()
        dataset = None

        # Check if exported correctly
        if os.path.isfile(out_path):
            print("\nLandslide susceptility index was exported.")

    # Getters
    def get_structure(self):
        """Returns the initialized NN structure."""
        return self.layers

    def get_weights(self):
        """Returns the weights of the model."""
        return [self.W1, self.W2]

    def get_lrate(self):
        """Returns the set learning rate."""
        return self.lrate

    def get_momentum(self):
        """Returns the set momentum factor."""
        return self.momentum

    # Setters
    def set_weights(self, weight):
        """Set the weights for final forward pass (excluding bias weights)."""
        self.W1 = weight[0][0:-1]  # exclude the bias for prediction
        self.W2 = weight[1]

# end of Neural_Network class


# Class optimizers
class BGD(Neural_Network):
    """Batch Gradient Descent."""

    def train_BGD(self, *args):
        """Performs simulation using batch gradien descent."""

        print(f"\nStarting BGD Training with NN structure:{self.layers}, lrate={self.lrate}, mu={self.momentum}:")
        training = args[0][0]
        validation = args[0][1]

        self.rmse = args[1]
        self.epoch = args[2]

        # X and y training sets
        X_train_1o = training[..., :10]
        y_train_1o = training[..., [-1]]

        # X and y validation sets
        X_val_1o = validation[..., :10]
        y_val_1o = validation[..., [-1]]

        self.loss_train = []
        self.loss_val = []

        for i in range(epoch):
            # Shuffle training set every epoch
            np.random.shuffle(training)

            # Start training
            self.train(X_train_1o, y_train_1o)

            # Training loss
            rmse_train = self.compute_error(X_train_1o, y_train_1o)

            # Validation loss
            rmse_val = self.compute_error(X_val_1o, y_val_1o)

            # Store errors
            self.loss_val.append(rmse_val)
            self.loss_train.append(rmse_train)

            # Print result per epoch
            print("epoch: {}".format(i))
            print("Training Loss: {}".format(rmse_train))
            print("Validation Loss: {}".format(rmse_val))
            print()

            # Threshold error
            if rmse_val <= self.rmse:
                break

        # Termination information
        self.term_epoch = i

        # Training statistics
        min_loss_train = np.amin(self.loss_train)
        max_loss_train = np.amax(self.loss_train)
        mean_loss_train = np.mean(self.loss_train)
        min_val_train = np.amin(self.loss_val)
        max_val_train = np.amax(self.loss_val)
        mean_val_train = np.mean(self.loss_val)

        print("\nTraining statistics:")
        print("Minimum train error:", min_loss_train)
        print("Maximum train error:", max_loss_train)
        print("Mean train error:", mean_loss_train)
        print()
        print("Minimum val error:", min_val_train)
        print("Maximum val error:", max_val_train)
        print("Mean val error:", mean_val_train)


class SGD(Neural_Network):
    """Stochastic Gradient Descent."""

    def train_SGD(self, *args):
        """Performs simulation using stochastic gradien descent."""
        print(f"\nStarting SGD Training with NN structure:{self.layers}, lrate={self.lrate}, mu={self.momentum}:")
        training = args[0][0]
        validation = args[0][1]

        self.rmse = args[1]
        self.epoch = args[2]

        # X and y training sets
        X_train_1o = training[..., :10]
        y_train_1o = training[..., [-1]]

        # X and y validation sets
        X_val_1o = validation[..., :10]
        y_val_1o = validation[..., [-1]]

        self.loss_train = []
        self.loss_val = []

        for i in range(epoch):
            # Stochastic sampling
            n = np.random.choice(X_train_1o.shape[0])

            # Start training
            self.train(X_train_1o[n], y_train_1o[n])

            # Training loss
            rmse_train = self.compute_error(X_train_1o[n], y_train_1o[n])

            # Validation loss
            rmse_val = self.compute_error(X_val_1o, y_val_1o)

            # Store errors
            self.loss_val.append(rmse_val)
            self.loss_train.append(rmse_train)

            # Print result per epoch
            print("epoch: {}".format(i))
            print("Training Loss: {}".format(rmse_train))
            print("Validation Loss: {}".format(rmse_val))
            print()

            # Threshold error
            if rmse_val <= self.rmse:
                break

        # Termination information
        self.term_epoch = i

        # Training statistics
        min_loss_train = np.amin(self.loss_train)
        max_loss_train = np.amax(self.loss_train)
        mean_loss_train = np.mean(self.loss_train)
        min_val_train = np.amin(self.loss_val)
        max_val_train = np.amax(self.loss_val)
        mean_val_train = np.mean(self.loss_val)

        print("\nTraining statistics:")
        print("Minimum train error:", min_loss_train)
        print("Maximum train error:", max_loss_train)
        print("Mean train error:", mean_loss_train)
        print()
        print("Minimum val error:", min_val_train)
        print("Maximum val error:", max_val_train)
        print("Mean val error:", mean_val_train)


class MGD(Neural_Network):
    """Mini-batch Gradient Descent."""

    def __init__(self, *args, **kwargs):
        Neural_Network.__init__(self, *args, **kwargs)
        self.batch_size = args[-1]

    def train_MGD(self, *args):
        """Performs simulation using mini-batch gradien descent"""
        print(f"\nStarting MGD Training with NN structure:{self.layers}, lrate={self.lrate}, mu={self.momentum}:")

        training = args[0][0]
        validation = args[0][1]

        self.rmse = args[1]
        self.epoch = args[2]

        # X and y training sets
        X_train_1o = training[..., :10]
        y_train_1o = training[..., [-1]]

        # X and y validation sets
        X_val_1o = validation[..., :10]
        y_val_1o = validation[..., [-1]]

        # Initialize number of iterations
        self.total_batch = X_train_1o.shape[0] // (self.batch_size - 1)  # exlude the last index
        val_batch = X_val_1o.shape[0] // (self.batch_size - 1)

        self.loss_train = []
        self.loss_val = []
        for i in range(epoch):

            # Shuffle datasets each epoch (after completely seeing the entire dataset)
            np.random.shuffle(training)

            # Reassign X and y training vectors
            X_train_1o = training[..., :10]
            y_train_1o = training[..., [-1]]

            X_batches = np.array_split(X_train_1o, self.total_batch)
            y_batches = np.array_split(y_train_1o, self.total_batch)

            # Initialize rmse
            # Will be reset after completing 1 epoch
            ave_cost_train = 0

            # Train the neural network per mini-batch
            for j in range(self.total_batch):
                minibatch_x = X_batches[j]
                minibatch_y = y_batches[j]

                # Train per mini-batch
                self.train(minibatch_x, minibatch_y)

                # Train loss for each mini-batch
                rmse_train = self.compute_error(minibatch_x, minibatch_y)

                # Get average training cost for each epoch
                ave_cost_train += rmse_train / self.total_batch

            self.loss_train.append(ave_cost_train)

#             # Validation using the whole validation set
#             rmse_val = self.compute_error(X_val_1o, y_val_1o)
#             self.loss_val.append(rmse_val)

#             # Print result per epoch
#             print("epoch: {}".format(i))
#             print("Training loss: {}".format(ave_cost_train))
#             print("Validation loss: {}".format(rmse_val))
#             print()

#             if rmse_val <= RMSE:
#                 break

            # Per mini-batch validation
            ave_cost_val = 0
            x_val = np.array_split(X_val_1o, val_batch)
            y_val = np.array_split(y_val_1o, val_batch)
            for v in range(val_batch):
                mb_val_x = x_val[v]
                mb_val_y = y_val[v]

                # Validation loss
                rmse_val = self.compute_error(mb_val_x, mb_val_y)

                # Get average val cost for each epoch
                ave_cost_val += rmse_val / val_batch

            self.loss_val.append(ave_cost_val)

            # Print result per epoch
            print("epoch: {}".format(i))
            print("Training loss: {}".format(ave_cost_train))
            print("Validation loss: {}".format(ave_cost_val))
            print()

            # Stoping criterion using validation set
            if ave_cost_val <= self.rmse:
                print("Training done.")
                break

        # Termination information
        self.term_epoch = i
        self.term_iter = j

        # Training statistics
        min_loss_train = np.amin(self.loss_train)
        max_loss_train = np.amax(self.loss_train)
        mean_loss_train = np.mean(self.loss_train)
        min_val_train = np.amin(self.loss_val)
        max_val_train = np.amax(self.loss_val)
        mean_val_train = np.mean(self.loss_val)

        print("\nTraining statistics:")
        print("Minimum train error:", min_loss_train)
        print("Maximum train error:", max_loss_train)
        print("Mean train error:", mean_loss_train)
        print()
        print("Minimum val error:", min_val_train)
        print("Maximum val error:", max_val_train)
        print("Mean val error:", mean_val_train)

# end of class optimizers


if __name__ == "__main__":
    # Start time
    start_time = time.process_time()

    # Initiate seed for easier debugging
    seed = 100
    np.random.seed(seed)

    # Filepath for landslide prone and nonprone samples
    fp_prone = r"D:\ms gme\thesis\final parameters\samples\Final\landslide"
    fp_notprone = r"D:\ms gme\thesis\final parameters\Samples\Final\no_landslide"

    my_thesis = Dataset(fp_prone, fp_notprone)

    # Load landslide factors
    datasets = my_thesis.load_layers()
    train_val_sets = datasets[0:2]
    test_set = datasets[2]

    # Start training
    # Instantiate NN class
    in_size = 9  # input layer
    h_size = 25  # hidden layer
    out_size = 1  # output layer

    # Hyperparameters
    lrate = 0.01
    momentum = 0.9
    batch_size = 32
    mgd = MGD(in_size, h_size, out_size, batch_size, lr=lrate, mu=momentum)

    # Initialize Root Mean Square Error
    RMSE = 0.01
    # Initialize number of epochs
    epoch = 3000

    mgd.train_MGD(train_val_sets, RMSE, epoch)
    # end of training

    # Plot loss curve
    xlabel = "Epoch"
    ylabel = "Average Cost"
    name = "raw_MGD_leaky_1o"
    mgd.plot_loss_curve(xlabel, ylabel, name, save=True)

    # Predict using testing samples
    # Forward pass using the whole dataset
    X_test_1o = test_set[..., :10]
    y_test_1o = test_set[..., [-1]]

    mgd.predict(X_test_1o, y_test_1o)

    # Write results to disk
    # Predicted
    mgd.save_predict(X_test_1o, y_test_1o)

    # Summary of results
    # Get layers name with the same index as input vector
    layers = [file for file in os.listdir(fp_prone) if os.path.isfile(os.path.join(fp_prone, file))]
    filename = "runs_MGD_leaky.txt"
    mgd.save_summary(filename, layers)

    # Plot target and predicted values
    plt.figure(figsize=(10, 5))
    plt.xlabel("Test samples")
    plt.ylabel("Cell Value")

    x, y = copy.deepcopy(X_test_1o), copy.deepcopy(y_test_1o)

    # Draw target values
    plt.plot(y, "o", color="b", label="Target value")

    # Draw network output values
    for i in range(X_test_1o.shape[0]):
        y[i] = mgd.forward(X_test_1o[i])
    plt.plot(y, '.', color='r', alpha=0.6, label="Predicted value")

    plt.legend(loc=2)

    # Save plot
    op = r"D:\MS Gme\Thesis\Final Parameters\ANN\sublime_run\plots"
    fn = f"{h_size}_{mgd.get_lrate()}_{mgd.get_momentum()}_{epoch}"

    if not os.path.isfile(os.path.join(op, fn)):
        plt.savefig(os.path.join(op, f"raw_leaky_test_1o_{fn}.jpg"))

    plt.show()

    #---------------------------------------------------------------#
    # # Generate LSI using the optimized weights
    # print("\nGenerating lsi using the best fit model.")
    # fuzzy_path = r"D:\MS Gme\Thesis\Final Parameters\Samples\for_lsi\Fuzzy\fuzzy2"
    # tiff = "*.tif"

    # lsi_ds = my_thesis.load_fuzzified_layers(fuzzy_path, tiff)
    # X_lsi = lsi_ds[0]
    # y_lsi = lsi_ds[1]

    # mgd.set_weights(mgd.get_weights())  # exxlude the bias weights

    # # Execute forward pass to the whole area per sample
    # for i in range(X_lsi.shape[0]):
    #     y_lsi[i] = mgd.forward(X_lsi[i])

    # # Reshape computed lsi to 2D array
    # lsi = y_lsi.reshape(6334, 3877)

    # # Export generated lsi
    # folder = "lsi"
    # out_path = os.getcwd()
    # op = os.path.join(out_path, folder)
    # if not os.path.exists(op):
    #     os.mkdir(folder)

    # fn = "lsi_MGD_leaky"
    # ref_data = os.path.join(fuzzy_path, "itogon_grid.tif")
    # mgd.export_to_image(ref_data, os.path.join(op, fn), lsi)

    print("-------------------------------------------------------")
    print("\nThe script finished its execution after %.2f seconds" % (time.process_time() - start_time))
