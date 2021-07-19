//! This module provides the matrix exponential (pow) function to square matrices.

use std::ops::DivAssign;

use crate::{
    allocator::Allocator,
    storage::{Storage, StorageMut},
    DefaultAllocator, DimMin, Matrix, OMatrix,
};
use num::PrimInt;
use simba::scalar::ComplexField;

impl<T: ComplexField, D, S> Matrix<T, D, D, S>
where
    D: DimMin<D, Output = D>,
    S: StorageMut<T, D, D>,
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>,
{
    /// Attempts to raise this matrix to an integral power `e` in-place. If this
    /// matrix is non-invertible and `e` is negative, it leaves this matrix
    /// untouched and returns `false`. Otherwise, it returns `true` and
    /// overwrites this matrix with the result.
    pub fn pow_mut<I: PrimInt + DivAssign>(&mut self, mut e: I) -> bool {
        let zero = I::zero();

        // A matrix raised to the zeroth power is just the identity.
        if e == zero {
            self.fill_with_identity();
            return true;
        }

        // If e is negative, we compute the inverse matrix, then raise it to the
        // power of -e.
        if e < zero && !self.try_inverse_mut() {
            return false;
        }

        let one = I::one();
        let two = I::from(2u8).unwrap();

        // We use the buffer to hold the result of multiplier ^ 2, thus avoiding
        // extra allocations.
        let (nrows, ncols) = self.data.shape();
        let mut multiplier = self.clone_owned();
        let mut buf = Matrix::new_uninitialized_generic(nrows, ncols);

        // Exponentiation by squares.
        loop {
            if e % two == one {
                let init_buf = self.mul_to(&multiplier, &mut buf);
                self.copy_from(&init_buf);

                // Safety: `mul_to` leaves `buf` completely initialized.
                unsafe {
                    buf.reinitialize();
                }
            }

            e /= two;

            let init_buf = multiplier.mul_to(&multiplier, &mut buf);
            multiplier.copy_from(&init_buf);

            // Safety: `mul_to` leaves `buf` completely initialized.
            unsafe {
                buf.reinitialize();
            }

            if e == zero {
                return true;
            }
        }
    }
}

impl<T: ComplexField, D, S: Storage<T, D, D>> Matrix<T, D, D, S>
where
    D: DimMin<D, Output = D>,
    S: StorageMut<T, D, D>,
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>,
{
    /// Attempts to raise this matrix to an integral power `e`. If this matrix
    /// is non-invertible and `e` is negative, it returns `None`. Otherwise, it
    /// returns the result as a new matrix. Uses exponentiation by squares.
    #[must_use]
    pub fn pow<I: PrimInt + DivAssign>(&self, e: I) -> Option<OMatrix<T, D, D>> {
        let mut clone = self.clone_owned();

        if clone.pow_mut(e) {
            Some(clone)
        } else {
            None
        }
    }
}
