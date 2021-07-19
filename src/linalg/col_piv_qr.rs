use num::Zero;
#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

use crate::allocator::{Allocator, Reallocator};
use crate::base::{Const, DefaultAllocator, Matrix, OMatrix, OVector, Unit};
use crate::constraint::{SameNumberOfRows, ShapeConstraint};
use crate::dimension::{Dim, DimMin, DimMinimum};
use crate::storage::{Storage, StorageMut};
use crate::ComplexField;

use crate::geometry::Reflection;
use crate::linalg::{householder, PermutationSequence};

/// The QR decomposition (with column pivoting) of a general matrix.
#[cfg_attr(feature = "serde-serialize-no-std", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(serialize = "DefaultAllocator: Allocator<T, R, C> +
                           Allocator<T, DimMinimum<R, C>>,
         OMatrix<T, R, C>: Serialize,
         PermutationSequence<DimMinimum<R, C>>: Serialize,
         OVector<T, DimMinimum<R, C>>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(deserialize = "DefaultAllocator: Allocator<T, R, C> +
                           Allocator<T, DimMinimum<R, C>>,
         OMatrix<T, R, C>: Deserialize<'de>,
         PermutationSequence<DimMinimum<R, C>>: Deserialize<'de>,
         OVector<T, DimMinimum<R, C>>: Deserialize<'de>"))
)]
pub struct ColPivQR<T: ComplexField, R: DimMin<C>, C: Dim>
where
    DefaultAllocator: Allocator<T, R, C>
        + Allocator<T, DimMinimum<R, C>>
        + Allocator<(usize, usize), DimMinimum<R, C>>,
{
    col_piv_qr: OMatrix<T, R, C>,
    p: PermutationSequence<DimMinimum<R, C>>,
    diag: OVector<T, DimMinimum<R, C>>,
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> Copy for ColPivQR<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C>
        + Allocator<T, DimMinimum<R, C>>
        + Allocator<(usize, usize), DimMinimum<R, C>>,
    OMatrix<T, R, C>: Copy,
    PermutationSequence<DimMinimum<R, C>>: Copy,
    OVector<T, DimMinimum<R, C>>: Copy,
{
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> Clone for ColPivQR<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C>
        + Allocator<T, DimMinimum<R, C>>
        + Allocator<(usize, usize), DimMinimum<R, C>>,
    OMatrix<T, R, C>: Clone,
    PermutationSequence<DimMinimum<R, C>>: Clone,
    OVector<T, DimMinimum<R, C>>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            col_piv_qr: self.col_piv_qr.clone(),
            p: self.p.clone(),
            diag: self.diag.clone(),
        }
    }
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> ColPivQR<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C>
        + Allocator<T, R>
        + Allocator<T, DimMinimum<R, C>>
        + Allocator<(usize, usize), DimMinimum<R, C>>,
{
    /// Computes the ColPivQR decomposition using householder reflections.
    pub fn new(mut matrix: OMatrix<T, R, C>) -> Self {
        let (nrows, ncols) = matrix.data.shape();
        let min_nrows_ncols = nrows.min(ncols);
        let mut p = PermutationSequence::identity_generic(min_nrows_ncols);

        let mut diag = Matrix::new_uninitialized_generic(min_nrows_ncols, Const::<1>);

        if min_nrows_ncols.value() == 0 {
            // Safety: there's no (uninitialized) values.
            unsafe {
                return ColPivQR {
                    col_piv_qr: matrix,
                    p,
                    diag: diag.assume_init(),
                };
            };
        }

        for i in 0..min_nrows_ncols.value() {
            let piv = matrix.slice_range(i.., i..).icamax_full();
            let col_piv = piv.1 + i;
            matrix.swap_columns(i, col_piv);
            p.append_permutation(i, col_piv);

            // Safety: the pointer is valid for writes, aligned, and uninitialized.
            unsafe {
                householder::clear_column_unchecked(&mut matrix, diag[i].as_mut_ptr(), i, 0, None);
            }
        }

        // Safety: all values have been initialized.
        unsafe {
            ColPivQR {
                col_piv_qr: matrix,
                p,
                diag: diag.assume_init(),
            }
        }
    }

    /// Retrieves the upper trapezoidal submatrix `R` of this decomposition.
    #[inline]
    #[must_use]
    pub fn r(&self) -> OMatrix<T, DimMinimum<R, C>, C>
    where
        DefaultAllocator: Allocator<T, DimMinimum<R, C>, C>,
    {
        let (nrows, ncols) = self.col_piv_qr.data.shape();
        let mut res = self
            .col_piv_qr
            .rows_generic(0, nrows.min(ncols))
            .upper_triangle();
        res.set_partial_diagonal(self.diag.iter().map(|e| T::from_real(e.modulus())));
        res
    }

    /// Retrieves the upper trapezoidal submatrix `R` of this decomposition.
    ///
    /// This is usually faster than `r` but consumes `self`.
    #[inline]
    pub fn unpack_r(self) -> OMatrix<T, DimMinimum<R, C>, C>
    where
        DefaultAllocator: Reallocator<T, R, C, DimMinimum<R, C>, C>,
    {
        let (nrows, ncols) = self.col_piv_qr.data.shape();
        let mut res = self
            .col_piv_qr
            .resize_generic(nrows.min(ncols), ncols, T::zero());
        res.fill_lower_triangle(T::zero(), 1);
        res.set_partial_diagonal(self.diag.iter().map(|e| T::from_real(e.modulus())));
        res
    }

    /// Computes the orthogonal matrix `Q` of this decomposition.
    #[must_use]
    pub fn q(&self) -> OMatrix<T, R, DimMinimum<R, C>>
    where
        DefaultAllocator: Allocator<T, R, DimMinimum<R, C>>,
    {
        let (nrows, ncols) = self.col_piv_qr.data.shape();

        // NOTE: we could build the identity matrix and call q_mul on it.
        // Instead we don't so that we take in account the matrix sparseness.
        let mut res = Matrix::identity_generic(nrows, nrows.min(ncols));
        let dim = self.diag.len();

        for i in (0..dim).rev() {
            let axis = self.col_piv_qr.slice_range(i.., i);
            // TODO: sometimes, the axis might have a zero magnitude.
            let refl = Reflection::new(Unit::new_unchecked(axis), T::zero());

            let mut res_rows = res.slice_range_mut(i.., i..);
            refl.reflect_with_sign(&mut res_rows, self.diag[i].signum());
        }

        res
    }
    /// Retrieves the column permutation of this decomposition.
    #[inline]
    #[must_use]
    pub fn p(&self) -> &PermutationSequence<DimMinimum<R, C>> {
        &self.p
    }

    /// Unpacks this decomposition into its two matrix factors.
    pub fn unpack(
        self,
    ) -> (
        OMatrix<T, R, DimMinimum<R, C>>,
        OMatrix<T, DimMinimum<R, C>, C>,
        PermutationSequence<DimMinimum<R, C>>,
    )
    where
        DimMinimum<R, C>: DimMin<C, Output = DimMinimum<R, C>>,
        DefaultAllocator: Allocator<T, R, DimMinimum<R, C>>
            + Reallocator<T, R, C, DimMinimum<R, C>, C>
            + Allocator<(usize, usize), DimMinimum<R, C>>,
    {
        (self.q(), self.r(), self.p)
    }

    #[doc(hidden)]
    pub fn col_piv_qr_internal(&self) -> &OMatrix<T, R, C> {
        &self.col_piv_qr
    }

    /// Multiplies the provided matrix by the transpose of the `Q` matrix of this decomposition.
    pub fn q_tr_mul<R2: Dim, C2: Dim, S2>(&self, rhs: &mut Matrix<T, R2, C2, S2>)
    where
        S2: StorageMut<T, R2, C2>,
    {
        let dim = self.diag.len();

        for i in 0..dim {
            let axis = self.col_piv_qr.slice_range(i.., i);
            let refl = Reflection::new(Unit::new_unchecked(axis), T::zero());

            let mut rhs_rows = rhs.rows_range_mut(i..);
            refl.reflect_with_sign(&mut rhs_rows, self.diag[i].signum().conjugate());
        }
    }
}

impl<T: ComplexField, D: DimMin<D, Output = D>> ColPivQR<T, D, D>
where
    DefaultAllocator:
        Allocator<T, D, D> + Allocator<T, D> + Allocator<(usize, usize), DimMinimum<D, D>>,
{
    /// Solves the linear system `self * x = b`, where `x` is the unknown to be determined.
    ///
    /// Returns `None` if `self` is not invertible.
    #[must_use = "Did you mean to use solve_mut()?"]
    pub fn solve<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<T, R2, C2, S2>,
    ) -> Option<OMatrix<T, R2, C2>>
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
        DefaultAllocator: Allocator<T, R2, C2>,
    {
        let mut res = b.clone_owned();

        if self.solve_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Solves the linear system `self * x = b`, where `x` is the unknown to be determined.
    ///
    /// If the decomposed matrix is not invertible, this returns `false` and its input `b` is
    /// overwritten with garbage.
    pub fn solve_mut<R2: Dim, C2: Dim, S2>(&self, b: &mut Matrix<T, R2, C2, S2>) -> bool
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        assert_eq!(
            self.col_piv_qr.nrows(),
            b.nrows(),
            "ColPivQR solve matrix dimension mismatch."
        );
        assert!(
            self.col_piv_qr.is_square(),
            "ColPivQR solve: unable to solve a non-square system."
        );

        self.q_tr_mul(b);
        let solved = self.solve_upper_triangular_mut(b);
        self.p.inv_permute_rows(b);

        solved
    }

    // TODO: duplicate code from the `solve` module.
    fn solve_upper_triangular_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<T, R2, C2, S2>,
    ) -> bool
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let dim = self.col_piv_qr.nrows();

        for k in 0..b.ncols() {
            let mut b = b.column_mut(k);
            for i in (0..dim).rev() {
                let coeff;

                unsafe {
                    let diag = self.diag.vget_unchecked(i).modulus();

                    if diag.is_zero() {
                        return false;
                    }

                    coeff = b.vget_unchecked(i).unscale(diag);
                    *b.vget_unchecked_mut(i) = coeff;
                }

                b.rows_range_mut(..i)
                    .axpy(-coeff, &self.col_piv_qr.slice_range(..i, i), T::one());
            }
        }

        true
    }

    /// Computes the inverse of the decomposed matrix.
    ///
    /// Returns `None` if the decomposed matrix is not invertible.
    #[must_use]
    pub fn try_inverse(&self) -> Option<OMatrix<T, D, D>> {
        assert!(
            self.col_piv_qr.is_square(),
            "ColPivQR inverse: unable to compute the inverse of a non-square matrix."
        );

        // TODO: is there a less naive method ?
        let (nrows, ncols) = self.col_piv_qr.data.shape();
        let mut res = OMatrix::identity_generic(nrows, ncols);

        if self.solve_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Indicates if the decomposed matrix is invertible.
    #[must_use]
    pub fn is_invertible(&self) -> bool {
        assert!(
            self.col_piv_qr.is_square(),
            "ColPivQR: unable to test the invertibility of a non-square matrix."
        );

        for i in 0..self.diag.len() {
            if self.diag[i].is_zero() {
                return false;
            }
        }

        true
    }

    /// Computes the determinant of the decomposed matrix.
    #[must_use]
    pub fn determinant(&self) -> T {
        let dim = self.col_piv_qr.nrows();
        assert!(
            self.col_piv_qr.is_square(),
            "ColPivQR determinant: unable to compute the determinant of a non-square matrix."
        );

        let mut res = T::one();
        for i in 0..dim {
            res *= unsafe { *self.diag.vget_unchecked(i) };
        }

        res * self.p.determinant()
    }
}
