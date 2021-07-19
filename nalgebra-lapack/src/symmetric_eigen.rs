#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use num::Zero;
use std::ops::MulAssign;

use simba::scalar::RealField;

use crate::ComplexHelper;
use na::allocator::Allocator;
use na::dimension::{Const, Dim};
use na::storage::Storage;
use na::{DefaultAllocator, Matrix, OMatrix, OVector, Scalar};

use lapack;

/// Eigendecomposition of a real square symmetric matrix with real eigenvalues.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(serialize = "DefaultAllocator: Allocator<T, D, D> +
                           Allocator<T, D>,
         OVector<T, D>: Serialize,
         OMatrix<T, D, D>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(deserialize = "DefaultAllocator: Allocator<T, D, D> +
                           Allocator<T, D>,
         OVector<T, D>: Deserialize<'de>,
         OMatrix<T, D, D>: Deserialize<'de>"))
)]
#[derive(Clone, Debug)]
pub struct SymmetricEigen<T: Scalar, D: Dim>
where
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    /// The eigenvectors of the decomposed matrix.
    pub eigenvectors: OMatrix<T, D, D>,

    /// The unsorted eigenvalues of the decomposed matrix.
    pub eigenvalues: OVector<T, D>,
}

impl<T: Scalar + Copy, D: Dim> Copy for SymmetricEigen<T, D>
where
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>,
    OMatrix<T, D, D>: Copy,
    OVector<T, D>: Copy,
{
}

impl<T: SymmetricEigenScalar + RealField, D: Dim> SymmetricEigen<T, D>
where
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>,
{
    /// Computes the eigenvalues and eigenvectors of the symmetric matrix `m`.
    ///
    /// Only the lower-triangular part of `m` is read. If `eigenvectors` is `false` then, the
    /// eigenvectors are not computed explicitly. Panics if the method did not converge.
    pub fn new(m: OMatrix<T, D, D>) -> Self {
        let (vals, vecs) =
            Self::do_decompose(m, true).expect("SymmetricEigen: convergence failure.");
        Self {
            eigenvalues: vals,
            eigenvectors: vecs.unwrap(),
        }
    }

    /// Computes the eigenvalues and eigenvectors of the symmetric matrix `m`.
    ///
    /// Only the lower-triangular part of `m` is read. If `eigenvectors` is `false` then, the
    /// eigenvectors are not computed explicitly. Returns `None` if the method did not converge.
    pub fn try_new(m: OMatrix<T, D, D>) -> Option<Self> {
        Self::do_decompose(m, true).map(|(vals, vecs)| SymmetricEigen {
            eigenvalues: vals,
            eigenvectors: vecs.unwrap(),
        })
    }

    fn do_decompose(
        mut m: OMatrix<T, D, D>,
        eigenvectors: bool,
    ) -> Option<(OVector<T, D>, Option<OMatrix<T, D, D>>)> {
        assert!(
            m.is_square(),
            "Unable to compute the eigenvalue decomposition of a non-square matrix."
        );

        let jobz = if eigenvectors { b'V' } else { b'T' };

        let nrows = m.data.shape().0;
        let n = nrows.value();

        let lda = n as i32;
        
  // IMPORTANT TODO: this is still UB.
        let mut values =
            unsafe { Matrix::new_uninitialized_generic(nrows, Const::<1>).assume_init() };
        let mut info = 0;

        let lwork = T::xsyev_work_size(jobz, b'L', n as i32, m.as_mut_slice(), lda, &mut info);
        lapack_check!(info);

        let mut work = unsafe { crate::uninitialized_vec(lwork as usize) };

        T::xsyev(
            jobz,
            b'L',
            n as i32,
            m.as_mut_slice(),
            lda,
            values.as_mut_slice(),
            &mut work,
            lwork,
            &mut info,
        );
        lapack_check!(info);

        let vectors = if eigenvectors { Some(m) } else { None };
        Some((values, vectors))
    }

    /// Computes only the eigenvalues of the input matrix.
    ///
    /// Panics if the method does not converge.
    pub fn eigenvalues(m: OMatrix<T, D, D>) -> OVector<T, D> {
        Self::do_decompose(m, false)
            .expect("SymmetricEigen eigenvalues: convergence failure.")
            .0
    }

    /// Computes only the eigenvalues of the input matrix.
    ///
    /// Returns `None` if the method does not converge.
    pub fn try_eigenvalues(m: OMatrix<T, D, D>) -> Option<OVector<T, D>> {
        Self::do_decompose(m, false).map(|res| res.0)
    }

    /// The determinant of the decomposed matrix.
    #[inline]
    #[must_use]
    pub fn determinant(&self) -> T {
        let mut det = T::one();
        for e in self.eigenvalues.iter() {
            det *= *e;
        }

        det
    }

    /// Rebuild the original matrix.
    ///
    /// This is useful if some of the eigenvalues have been manually modified.
    #[must_use]
    pub fn recompose(&self) -> OMatrix<T, D, D> {
        let mut u_t = self.eigenvectors.clone();
        for i in 0..self.eigenvalues.len() {
            let val = self.eigenvalues[i];
            u_t.column_mut(i).mul_assign(val);
        }
        u_t.transpose_mut();
        &self.eigenvectors * u_t
    }
}

/*
 *
 * Lapack functions dispatch.
 *
 */
/// Trait implemented by scalars for which Lapack implements the eigendecomposition of symmetric
/// real matrices.
pub trait SymmetricEigenScalar: Scalar {
    #[allow(missing_docs)]
    fn xsyev(
        jobz: u8,
        uplo: u8,
        n: i32,
        a: &mut [Self],
        lda: i32,
        w: &mut [Self],
        work: &mut [Self],
        lwork: i32,
        info: &mut i32,
    );
    #[allow(missing_docs)]
    fn xsyev_work_size(jobz: u8, uplo: u8, n: i32, a: &mut [Self], lda: i32, info: &mut i32)
        -> i32;
}

macro_rules! real_eigensystem_scalar_impl (
    ($N: ty, $xsyev: path) => (
        impl SymmetricEigenScalar for $N {
            #[inline]
            fn xsyev(jobz: u8, uplo: u8, n: i32, a: &mut [Self], lda: i32, w: &mut [Self], work: &mut [Self],
                     lwork: i32, info: &mut i32) {
                unsafe { $xsyev(jobz, uplo, n, a, lda, w, work, lwork, info) }
            }


            #[inline]
            fn xsyev_work_size(jobz: u8, uplo: u8, n: i32, a: &mut [Self], lda: i32, info: &mut i32) -> i32 {
                let mut work = [ Zero::zero() ];
                let mut w    = [ Zero::zero() ];
                let lwork    = -1 as i32;

                unsafe { $xsyev(jobz, uplo, n, a, lda, &mut w, &mut work, lwork, info); }
                ComplexHelper::real_part(work[0]) as i32
            }
        }
    )
);

real_eigensystem_scalar_impl!(f32, lapack::ssyev);
real_eigensystem_scalar_impl!(f64, lapack::dsyev);
