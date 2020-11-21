import unittest
import numpy as np
import collaborative_filter as cf


class TestHLVModelImplementation(unittest.TestCase):
    def setUp(self):
        pass

    # Test a matrix that is the direct factor of two other matrices. While the factorisation is not unique
    # so we cannot check the result for equality, the procedure should find two extremely good approximations to
    # true factors and we expect the errors to be tiny (as long as lambda_1, lambda_2 are small)
    def test_precomputed_matrix_epsilon_matrix(self):
        precomputed_matrix = np.matrix([[7, 10, 17, 23, 29], [15, 22, 39, 53, 67],
                                        [23, 34, 61, 83, 105], [31, 46, 83, 113, 143]]);

        lambda_1 = 1e-2
        lambda_2 = 1e-2

        model = cf.HKVModel(dimension=2, alpha=5, lambda_1=lambda_1, lambda_2=lambda_2)
        factored = cf.MatrixFactorisation(precomputed_matrix, model)

        result = factored.get_predicted_matrix()
        error = np.sum(np.square(result - precomputed_matrix))

        self.assertLess(error, 0.01)

    # Also test the cost function coming from the model, but be aware that this will be dominated by the sum of
    # squares of the factor matrices that appear in the regulisation term, so subtract these from the error term
    def test_precomputed_matrix_cost_function(self):
        precomputed_matrix = np.matrix([[7, 10, 17, 23, 29], [15, 22, 39, 53, 67],
                                        [23, 34, 61, 83, 105], [31, 46, 83, 113, 143]]);

        lambda_1 = 1e-2
        lambda_2 = 1e-2

        model = cf.HKVModel(dimension=2, alpha=5, lambda_1=lambda_1, lambda_2=lambda_2)
        factored = cf.MatrixFactorisation(precomputed_matrix, model)

        error = factored.get_prediction_error()
        x,y = factored.get_calibrated_matrices()

        modified_error = error - lambda_1 * np.sum(np.square(x)) - lambda_2 * np.sum(np.square(y))

        self.assertLess(modified_error, 1)


if __name__ == '__main__':
    unittest.main()
