import numpy as np


# This class uses the procedure described in the paper by Hu, Koren and Volinsky to factorise two matrices.
# Our model differs from theirs in that we break the regularisation term lambda into lambda_1 for the L2 norm of the
# x matrix, and lambda_2 for the L2 norm of the y matrix
class HKVModel:
    # In the initialiser of the method, we specify the model constants dimension (number of latent variables), alpha
    # (additional weight given to non-zero entries in M), and the regularisation terms lambda_1 and lambda_2
    def __init__(self, dimension=10, alpha=40, lambda_1=1, lambda_2=1):
        self.dimension = dimension
        self.alpha = alpha
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    # We calibrate two smaller matrixes by taking two initial guesses and performing the iterative triangulation
    # procedure described in the paper, fixing Y and minimising the cost on X, then repeating for Y, and continuing
    # until the cost is below a certain value
    def calibrate(self, matrix, steps=20):
        (height, width) = matrix.shape

        # Define initial guesses for matrices x and y that have every element 1. TODO: improve initial guesses
        # The entered matrix is dimension n*m, x will be n*k and y will be k*m
        x = np.matrix([[1] * self.dimension] * height)
        y = np.matrix([[1] * width] * self.dimension)

        for i in range(steps):
            # Hold y constant, and optimise for x using analytic expression
            x = self.x_optimise(matrix, y)

            # Hold x constant, and optimise for y using analytic expression
            y = self.y_optimise(matrix, x)

        return x, y

    # Given the initial matrix and the y matrix current guess, we can optimise x analytically
    def x_optimise(self, matrix, y):
        (height, width) = matrix.shape
        lambda_i = np.diag([self.lambda_1] * self.dimension)

        x_new = []
        for u in range(height):
            p_u = matrix[u, :].reshape((width, 1))
            C_u = np.diag([(1 if matrix[u, j] == 0 else self.alpha * matrix[u, j]) for j in range(width)])

            # This is Eq. (4) in the paper
            improved = np.linalg.inv(y * C_u * np.transpose(y) + lambda_i) * y * C_u * p_u

            x_new.append(np.asarray(improved).reshape(-1))

        return np.matrix(np.concatenate(x_new).reshape(height, self.dimension))

    def y_optimise(self, matrix, x):
        (height, width) = matrix.shape
        lambda_i = np.diag([self.lambda_2] * self.dimension)

        y_new = []

        for i in range(width):
            p_i = matrix[:, i].reshape((height, 1))
            C_i = np.diag([(1 if matrix[j, i] == 0 else self.alpha * matrix[j, i]) for j in range(height)])

            # This is Eq. (5) in the paper
            improved = np.linalg.inv(np.transpose(x) * C_i * x + lambda_i) * np.transpose(x) * C_i * p_i

            y_new.append(np.asarray(improved).reshape(-1))

        return np.transpose(np.matrix(np.concatenate(y_new).reshape(width, self.dimension)))

    # Calculate the cost function for a given matrix M, and two trial matrices X and Y, as given in Eq.(3) in the paper
    # Currently this is done quite inefficiently, but we aren't calling it in any loops as we have an analytical
    # solution - it's purely here for informational purposes
    def cost(self, matrix, x, y):
        (height, width) = matrix.shape

        error_terms = matrix - (x * y)
        weighted_error_terms = 0

        # The weight factor is 1 if the initial matrix has a 0 in the space, or alpha if it is non-zero (reduced weight
        # on the zeros helps to prevent over-fitting on zeros). Most easily achieved by nested loops, O(mn)
        for i in range(height):
            for j in range(width):
                weighted_error_terms += pow(error_terms[i, j], 2) \
                                      * (self.alpha * matrix[i, j] if abs(matrix[i, j]) > 0 else 1)

        # The regularisation terms are the trace of the product of x with its transpose, and similarly for y. It's
        # more efficient to calculate this as the sum of the element-wise squares though, which is O( d*max(m,n) )
        x_regulariser = np.sum(np.square(x))
        y_regulariser = np.sum(np.square(y))

        return weighted_error_terms + self.lambda_1 * x_regulariser + self.lambda_2 * y_regulariser
