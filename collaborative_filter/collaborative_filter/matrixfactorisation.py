import numpy as np
from collaborative_filter.hkvmodel import HKVModel


# This class takes a matrix, and performs a factorisation into two smaller matrices, using a model class specified
# when calling the factorisation (defaults to HKV)
# TODO: for a given dataset, add a class to calibrate the model parameters by cross-validation
class MatrixFactorisation:
    def __init__(self, matrix, model=HKVModel()):
        self.M = matrix
        self.model = model

        (self.X, self.Y) = model.calibrate(matrix)

        self.prediction_matrix = self.X * self.Y
        self.prediction_error = self.model.cost(self.M, self.X, self.Y)

        self.print_model_output()

    # Print useful diagnostics around the model's performance, and any outcomes that are very different to the input
    # matrix (these are the product/client matches that haven't happened yet but we believe quite likely)
    def print_model_output(self):
        print("Calibrated the following matrix factors:")
        print(np.around(self.X, decimals=1))
        print(np.around(self.Y, decimals=1))

        print("Calibrated cost function is: %2.0f" % self.prediction_error)

        print("The calculated product matrix is:")
        print(np.around(self.prediction_matrix, decimals=1))

    def get_calibrated_matrices(self):
        return self.X, self.Y

    def get_predicted_matrix(self):
        return self.prediction_matrix

    def get_prediction_error(self):
        return self.prediction_error
