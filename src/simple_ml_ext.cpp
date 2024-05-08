#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void matmul(const float* A, const float* B, float* C, int n, int m, int p) {
    /**
     * Args:
     * A: n x m, B: m x p
     * C: n x p = A x B
     */
     for (int i = 0; i < n; i++) {
        for (int k = 0; k < p; k++) {
            C[i * p + k] = 0
            for (int j = 0; j < m; j++) {
                C[i * p + k] += A[i * m + j] * B[j * p + k];
            }
        }
     }
}

void softmax(float* X, int n, int m) {
    for (int i = 0; i < n; i++) {
        float sum = 0;
        for (int j = 0; j < m; j++) {
            X[i * m + j] = exp(X[i * m + j]);
            sum += X[i * m + j];
        }
        for (int j = 0; j < m; j++) {
            X[i * m + j] = X[i * m + j] / sum;
        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    auto Z = new float[m * k];

    for (int i = 0; i < m; i += batch) {
        auto X_batch = X + i * n
        auto y_batch = y + i
        m_batch = min(m - i, batch);
        matmul(X_batch, y_batch, Z, m_batch, n, k);
        softmax(Z, m_batch, k);
        // grad = X_batch.T @ (Z - np.eye(Z.shape[1])[y_batch]) / y_batch.shape[0]
        for (int r_Z = 0; r_Z < m_batch; r_Z++) {
            Z[r_Z * k + y_batch[r_Z]] -= 1;
        }
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < k; col++) {
                float grad = 0;
                for (int mid = 0; mid < m_batch; mid++) {
                    grad += X_batch[mid * n + row] * Z[mid * k + col];
                }
                theta[row * k + col] -= lr * grad / m_batch;
            }
        }
    }

    delete Z;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
