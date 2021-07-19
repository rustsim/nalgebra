use std::fmt;
use std::mem;

#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

use crate::allocator::{Allocator, Reallocator};
use crate::base::{DefaultAllocator, Matrix, OMatrix, Scalar};
use crate::constraint::{SameNumberOfRows, ShapeConstraint};
use crate::dimension::{Dim, DimMin, DimMinimum};
use crate::storage::{InnerOwned, Storage, StorageMut};
use simba::scalar::{ComplexField, Field};

use crate::linalg::PermutationSequence;

/// LU decomposition with partial (row) pivoting.
#[cfg_attr(feature = "serde-serialize-no-std", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(serialize = "DefaultAllocator: Allocator<T, R, C> +
                           Allocator<(usize, usize), DimMinimum<R, C>>,
         OMatrix<T, R, C>: Serialize,
         PermutationSequence<DimMinimum<R, C>>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(deserialize = "DefaultAllocator: Allocator<T, R, C> +
                           Allocator<(usize, usize), DimMinimum<R, C>>,
         OMatrix<T, R, C>: Deserialize<'de>,
         PermutationSequence<DimMinimum<R, C>>: Deserialize<'de>"))
)]
pub struct LU<T, R: DimMin<C>, C: Dim>
where
    DefaultAllocator: Allocator<T, R, C> + Allocator<(usize, usize), DimMinimum<R, C>>,
{
    lu: OMatrix<T, R, C>,
    p: PermutationSequence<DimMinimum<R, C>>,
}

/*
impl<T: Copy, R: DimMin<C>, C: Dim> Copy for LU<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C> + Allocator<(usize, usize), DimMinimum<R, C>>,
    PermutationSequence<DimMinimum<R, C>>: Copy,
    InnerOwned<T, R, C>: Copy,
{
}
*/

impl<T: Clone, R: DimMin<C>, C: Dim> Clone for LU<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C> + Allocator<(usize, usize), DimMinimum<R, C>>,
    PermutationSequence<DimMinimum<R, C>>: Clone,
    InnerOwned<T, R, C>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            lu: self.lu.clone(),
            p: self.p.clone(),
        }
    }
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> fmt::Debug for LU<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C> + Allocator<(usize, usize), DimMinimum<R, C>>,
    PermutationSequence<DimMinimum<R, C>>: fmt::Debug,
    InnerOwned<T, R, C>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("LU")
            .field("lu", &self.lu)
            .field("p", &self.p)
            .finish()
    }
}

/// Performs a LU decomposition to overwrite `out` with the inverse of `matrix`.
///
/// If `matrix` is not invertible, `false` is returned and `out` may contain invalid data.
pub fn try_invert_to<T: ComplexField, D: Dim, S>(
    mut matrix: OMatrix<T, D, D>,
    out: &mut Matrix<T, D, D, S>,
) -> bool
where
    S: StorageMut<T, D, D>,
    DefaultAllocator: Allocator<T, D, D>,
{
    assert!(
        matrix.is_square(),
        "LU inversion: unable to invert a rectangular matrix."
    );
    let dim = matrix.nrows();

    out.fill_with_identity();

    for i in 0..dim {
        let piv = matrix.slice_range(i.., i).icamax() + i;
        let diag = matrix[(piv, i)];

        if diag.is_zero() {
            return false;
        }

        if piv != i {
            out.swap_rows(i, piv);
            matrix.columns_range_mut(..i).swap_rows(i, piv);
            gauss_step_swap(&mut matrix, diag, i, piv);
        } else {
            gauss_step(&mut matrix, diag, i);
        }
    }

    let _ = matrix.solve_lower_triangular_with_diag_mut(out, T::one());
    matrix.solve_upper_triangular_mut(out)
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> LU<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C> + Allocator<(usize, usize), DimMinimum<R, C>>,
{
    /// Computes the LU decomposition with partial (row) pivoting of `matrix`.
    pub fn new(mut matrix: OMatrix<T, R, C>) -> Self {
        let (nrows, ncols) = matrix.data.shape();
        let min_nrows_ncols = nrows.min(ncols);

        let mut p = PermutationSequence::identity_generic(min_nrows_ncols);

        if min_nrows_ncols.value() == 0 {
            return LU { lu: matrix, p };
        }

        for i in 0..min_nrows_ncols.value() {
            let piv = matrix.slice_range(i.., i).icamax() + i;
            let diag = matrix[(piv, i)];

            if diag.is_zero() {
                // No non-zero entries on this column.
                continue;
            }

            if piv != i {
                p.append_permutation(i, piv);
                matrix.columns_range_mut(..i).swap_rows(i, piv);
                gauss_step_swap(&mut matrix, diag, i, piv);
            } else {
                gauss_step(&mut matrix, diag, i);
            }
        }

        LU { lu: matrix, p }
    }

    #[doc(hidden)]
    pub fn lu_internal(&self) -> &OMatrix<T, R, C> {
        &self.lu
    }

    /// The lower triangular matrix of this decomposition.
    #[inline]
    #[must_use]
    pub fn l(&self) -> OMatrix<T, R, DimMinimum<R, C>>
    where
        DefaultAllocator: Allocator<T, R, DimMinimum<R, C>>,
    {
        let (nrows, ncols) = self.lu.data.shape();
        let mut m = self.lu.columns_generic(0, nrows.min(ncols)).into_owned();
        m.fill_upper_triangle(T::zero(), 1);
        m.fill_diagonal(T::one());
        m
    }

    /// The lower triangular matrix of this decomposition.
    fn l_unpack_with_p(
        self,
    ) -> (
        OMatrix<T, R, DimMinimum<R, C>>,
        PermutationSequence<DimMinimum<R, C>>,
    )
    where
        DefaultAllocator: Reallocator<T, R, C, R, DimMinimum<R, C>>,
    {
        let (nrows, ncols) = self.lu.data.shape();
        let mut m = self.lu.resize_generic(nrows, nrows.min(ncols), T::zero());
        m.fill_upper_triangle(T::zero(), 1);
        m.fill_diagonal(T::one());
        (m, self.p)
    }

    /// The lower triangular matrix of this decomposition.
    #[inline]
    pub fn l_unpack(self) -> OMatrix<T, R, DimMinimum<R, C>>
    where
        DefaultAllocator: Reallocator<T, R, C, R, DimMinimum<R, C>>,
    {
        let (nrows, ncols) = self.lu.data.shape();
        let mut m = self.lu.resize_generic(nrows, nrows.min(ncols), T::zero());
        m.fill_upper_triangle(T::zero(), 1);
        m.fill_diagonal(T::one());
        m
    }

    /// The upper triangular matrix of this decomposition.
    #[inline]
    #[must_use]
    pub fn u(&self) -> OMatrix<T, DimMinimum<R, C>, C>
    where
        DefaultAllocator: Allocator<T, DimMinimum<R, C>, C>,
    {
        let (nrows, ncols) = self.lu.data.shape();
        self.lu.rows_generic(0, nrows.min(ncols)).upper_triangle()
    }

    /// The row permutations of this decomposition.
    #[inline]
    #[must_use]
    pub fn p(&self) -> &PermutationSequence<DimMinimum<R, C>> {
        &self.p
    }

    /// The row permutations and two triangular matrices of this decomposition: `(P, L, U)`.
    #[inline]
    pub fn unpack(
        self,
    ) -> (
        PermutationSequence<DimMinimum<R, C>>,
        OMatrix<T, R, DimMinimum<R, C>>,
        OMatrix<T, DimMinimum<R, C>, C>,
    )
    where
        DefaultAllocator: Allocator<T, R, DimMinimum<R, C>>
            + Allocator<T, DimMinimum<R, C>, C>
            + Reallocator<T, R, C, R, DimMinimum<R, C>>,
    {
        // Use reallocation for either l or u.
        let u = self.u();
        let (l, p) = self.l_unpack_with_p();

        (p, l, u)
    }
}

impl<T: ComplexField, D: DimMin<D, Output = D>> LU<T, D, D>
where
    DefaultAllocator: Allocator<T, D, D> + Allocator<(usize, usize), D>,
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
        S2: Storage<T, R2, C2>,
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
    /// If the decomposed matrix is not invertible, this returns `false` and its input `b` may
    /// be overwritten with garbage.
    pub fn solve_mut<R2: Dim, C2: Dim, S2>(&self, b: &mut Matrix<T, R2, C2, S2>) -> bool
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        assert_eq!(
            self.lu.nrows(),
            b.nrows(),
            "LU solve matrix dimension mismatch."
        );
        assert!(
            self.lu.is_square(),
            "LU solve: unable to solve a non-square system."
        );

        self.p.permute_rows(b);
        let _ = self.lu.solve_lower_triangular_with_diag_mut(b, T::one());
        self.lu.solve_upper_triangular_mut(b)
    }

    /// Computes the inverse of the decomposed matrix.
    ///
    /// Returns `None` if the matrix is not invertible.
    #[must_use]
    pub fn try_inverse(&self) -> Option<OMatrix<T, D, D>> {
        assert!(
            self.lu.is_square(),
            "LU inverse: unable to compute the inverse of a non-square matrix."
        );

        let (nrows, ncols) = self.lu.data.shape();
        let mut res = OMatrix::identity_generic(nrows, ncols);
        if self.try_inverse_to(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Computes the inverse of the decomposed matrix and outputs the result to `out`.
    ///
    /// If the decomposed matrix is not invertible, this returns `false` and `out` may be
    /// overwritten with garbage.
    pub fn try_inverse_to<S2: StorageMut<T, D, D>>(&self, out: &mut Matrix<T, D, D, S2>) -> bool {
        assert!(
            self.lu.is_square(),
            "LU inverse: unable to compute the inverse of a non-square matrix."
        );
        assert!(
            self.lu.shape() == out.shape(),
            "LU inverse: mismatched output shape."
        );

        out.fill_with_identity();
        self.solve_mut(out)
    }

    /// Computes the determinant of the decomposed matrix.
    #[must_use]
    pub fn determinant(&self) -> T {
        let dim = self.lu.nrows();
        assert!(
            self.lu.is_square(),
            "LU determinant: unable to compute the determinant of a non-square matrix."
        );

        let mut res = T::one();
        for i in 0..dim {
            res *= unsafe { *self.lu.get_unchecked((i, i)) };
        }

        res * self.p.determinant()
    }

    /// Indicates if the decomposed matrix is invertible.
    #[must_use]
    pub fn is_invertible(&self) -> bool {
        assert!(
            self.lu.is_square(),
            "QR: unable to test the invertibility of a non-square matrix."
        );

        for i in 0..self.lu.nrows() {
            if self.lu[(i, i)].is_zero() {
                return false;
            }
        }

        true
    }
}

#[doc(hidden)]
/// Executes one step of gaussian elimination on the i-th row and column of `matrix`. The diagonal
/// element `matrix[(i, i)]` is provided as argument.
pub fn gauss_step<T, R: Dim, C: Dim, S>(matrix: &mut Matrix<T, R, C, S>, diag: T, i: usize)
where
    T: Scalar + Field,
    S: StorageMut<T, R, C>,
{
    let mut submat = matrix.slice_range_mut(i.., i..);

    let inv_diag = T::one() / diag;

    let (mut coeffs, mut submat) = submat.columns_range_pair_mut(0, 1..);

    let mut coeffs = coeffs.rows_range_mut(1..);
    coeffs *= inv_diag;

    let (pivot_row, mut down) = submat.rows_range_pair_mut(0, 1..);

    for k in 0..pivot_row.ncols() {
        down.column_mut(k)
            .axpy(-pivot_row[k].inlined_clone(), &coeffs, T::one());
    }
}

#[doc(hidden)]
/// Swaps the rows `i` with the row `piv` and executes one step of gaussian elimination on the i-th
/// row and column of `matrix`. The diagonal element `matrix[(i, i)]` is provided as argument.
pub fn gauss_step_swap<T, R: Dim, C: Dim, S>(
    matrix: &mut Matrix<T, R, C, S>,
    diag: T,
    i: usize,
    piv: usize,
) where
    T: Scalar + Field,
    S: StorageMut<T, R, C>,
{
    let piv = piv - i;
    let mut submat = matrix.slice_range_mut(i.., i..);

    let inv_diag = T::one() / diag;

    let (mut coeffs, mut submat) = submat.columns_range_pair_mut(0, 1..);

    coeffs.swap((0, 0), (piv, 0));
    let mut coeffs = coeffs.rows_range_mut(1..);
    coeffs *= inv_diag;

    let (mut pivot_row, mut down) = submat.rows_range_pair_mut(0, 1..);

    for k in 0..pivot_row.ncols() {
        mem::swap(&mut pivot_row[k], &mut down[(piv - 1, k)]);
        down.column_mut(k)
            .axpy(-pivot_row[k].inlined_clone(), &coeffs, T::one());
    }
}
