use crate::base::DMatrix;
use crate::storage::Storage;
use crate::{Dim, Dynamic, Matrix, Scalar};
use num::Zero;
use std::ops::{AddAssign, Mul};

impl<N, R1, C1, SA> Matrix<N, R1, C1, SA>
where
    N: Scalar + Zero + AddAssign + Mul<Output = N> + Copy,
    R1: Dim,
    C1: Dim,
    SA: Storage<N, R1, C1>,
{
    /// Returns the convolution of the target matrix and a kernel.
    ///
    /// # Arguments
    ///
    /// * `kernel` - A Matrix with size > 0
    ///
    /// # Errors
    /// Inputs must satisfy `matrix.len() >= matrix.len() > 0`.
    ///
    pub fn filter_2d<R2, C2, SB>(&self, kernel: Matrix<N, R2, C2, SB>) -> DMatrix<N>
    where
        R2: Dim,
        C2: Dim,
        SB: Storage<N, R2, C2>,
    {
        let mat_shape = self.shape();
        let ker_shape = kernel.shape();

        if ker_shape == (0, 0) || ker_shape > mat_shape {
            panic!("filter_2d expects `self.shape() >= kernel.shape() > 0`, received {:?} and {:?} respectively.", mat_shape, ker_shape);
        }

        let result_shape = (mat_shape.0 - ker_shape.0 + 1, mat_shape.1 - ker_shape.1 + 1);
        let mut conv = DMatrix::zeros_generic(
            Dynamic::from_usize(result_shape.0),
            Dynamic::from_usize(result_shape.1),
        );

        // TODO: optimize
        for i in 0..(result_shape.0) {
            for j in 0..(result_shape.1) {
                for k in 0..(ker_shape.0) {
                    for l in 0..(ker_shape.1) {
                        conv[(i, j)] += self[(i + k, j + l)] * kernel[(k, l)]
                    }
                }
            }
        }
        conv
    }
}
