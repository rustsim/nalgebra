use std::fmt;

#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, Matrix, OMatrix};
use crate::constraint::{SameNumberOfRows, ShapeConstraint};
use crate::dimension::{Dim, DimMin, DimMinimum};
use crate::storage::{Storage, StorageMut};
use simba::scalar::ComplexField;

use crate::linalg::lu;
use crate::linalg::PermutationSequence;

/// LU decomposition with full row and column pivoting.
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
pub struct FullPivLU<T, R: DimMin<C>, C: Dim>
where
    DefaultAllocator: Allocator<T, R, C> + Allocator<(usize, usize), DimMinimum<R, C>>,
{
    lu: OMatrix<T, R, C>,
    p: PermutationSequence<DimMinimum<R, C>>,
    q: PermutationSequence<DimMinimum<R, C>>,
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> Copy for FullPivLU<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C> + Allocator<(usize, usize), DimMinimum<R, C>>,
    PermutationSequence<DimMinimum<R, C>>: Copy,
    OMatrix<T, R, C>: Copy,
{
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> Clone for FullPivLU<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C> + Allocator<(usize, usize), DimMinimum<R, C>>,
    PermutationSequence<DimMinimum<R, C>>: Clone,
    OMatrix<T, R, C>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            lu: self.lu.clone(),
            p: self.p.clone(),
            q: self.q.clone(),
        }
    }
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> fmt::Debug for FullPivLU<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C> + Allocator<(usize, usize), DimMinimum<R, C>>,
    PermutationSequence<DimMinimum<R, C>>: fmt::Debug,
    OMatrix<T, R, C>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FullPivLU")
            .field("lu", &self.lu)
            .field("p", &self.p)
            .field("q", &self.q)
            .finish()
    }
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> FullPivLU<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C> + Allocator<(usize, usize), DimMinimum<R, C>>,
{
    /// Computes the LU decomposition with full pivoting of `matrix`.
    ///
    /// This effectively computes `P, L, U, Q` such that `P * matrix * Q = LU`.
    pub fn new(mut matrix: OMatrix<T, R, C>) -> Self {
        let (nrows, ncols) = matrix.data.shape();
        let min_nrows_ncols = nrows.min(ncols);

        let mut p = PermutationSequence::identity_generic(min_nrows_ncols);
        let mut q = PermutationSequence::identity_generic(min_nrows_ncols);

        if min_nrows_ncols.value() == 0 {
            return Self { lu: matrix, p, q };
        }

        for i in 0..min_nrows_ncols.value() {
            let piv = matrix.slice_range(i.., i..).icamax_full();
            let row_piv = piv.0 + i;
            let col_piv = piv.1 + i;
            let diag = matrix[(row_piv, col_piv)];

            if diag.is_zero() {
                // The remaining of the matrix is zero.
                break;
            }

            matrix.swap_columns(i, col_piv);
            q.append_permutation(i, col_piv);

            if row_piv != i {
                p.append_permutation(i, row_piv);
                matrix.columns_range_mut(..i).swap_rows(i, row_piv);
                lu::gauss_step_swap(&mut matrix, diag, i, row_piv);
            } else {
                lu::gauss_step(&mut matrix, diag, i);
            }
        }

        Self { lu: matrix, p, q }
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

    /// The column permutations of this decomposition.
    #[inline]
    #[must_use]
    pub fn q(&self) -> &PermutationSequence<DimMinimum<R, C>> {
        &self.q
    }

    /// The two matrices of this decomposition and the row and column permutations: `(P, L, U, Q)`.
    #[inline]
    pub fn unpack(
        self,
    ) -> (
        PermutationSequence<DimMinimum<R, C>>,
        OMatrix<T, R, DimMinimum<R, C>>,
        OMatrix<T, DimMinimum<R, C>, C>,
        PermutationSequence<DimMinimum<R, C>>,
    )
    where
        DefaultAllocator: Allocator<T, R, DimMinimum<R, C>> + Allocator<T, DimMinimum<R, C>, C>,
    {
        // Use reallocation for either l or u.
        let l = self.l();
        let u = self.u();
        let p = self.p;
        let q = self.q;

        (p, l, u, q)
    }
}

impl<T: ComplexField, D: DimMin<D, Output = D>> FullPivLU<T, D, D>
where
    DefaultAllocator: Allocator<T, D, D> + Allocator<(usize, usize), D>,
{
    /// Solves the linear system `self * x = b`, where `x` is the unknown to be determined.
    ///
    /// Returns `None` if the decomposed matrix is not invertible.
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
            "FullPivLU solve matrix dimension mismatch."
        );
        assert!(
            self.lu.is_square(),
            "FullPivLU solve: unable to solve a non-square system."
        );

        if self.is_invertible() {
            self.p.permute_rows(b);
            let _ = self.lu.solve_lower_triangular_with_diag_mut(b, T::one());
            let _ = self.lu.solve_upper_triangular_mut(b);
            self.q.inv_permute_rows(b);

            true
        } else {
            false
        }
    }

    /// Computes the inverse of the decomposed matrix.
    ///
    /// Returns `None` if the decomposed matrix is not invertible.
    #[must_use]
    pub fn try_inverse(&self) -> Option<OMatrix<T, D, D>> {
        assert!(
            self.lu.is_square(),
            "FullPivLU inverse: unable to compute the inverse of a non-square matrix."
        );

        let (nrows, ncols) = self.lu.data.shape();

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
            self.lu.is_square(),
            "FullPivLU: unable to test the invertibility of a non-square matrix."
        );

        let dim = self.lu.nrows();
        !self.lu[(dim - 1, dim - 1)].is_zero()
    }

    /// Computes the determinant of the decomposed matrix.
    #[must_use]
    pub fn determinant(&self) -> T {
        assert!(
            self.lu.is_square(),
            "FullPivLU determinant: unable to compute the determinant of a non-square matrix."
        );

        let dim = self.lu.nrows();
        let mut res = self.lu[(dim - 1, dim - 1)];
        if !res.is_zero() {
            for i in 0..dim - 1 {
                res *= unsafe { *self.lu.get_unchecked((i, i)) };
            }

            res * self.p.determinant() * self.q.determinant()
        } else {
            T::zero()
        }
    }
}
