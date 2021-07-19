#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use num::Zero;
use num_complex::Complex;

use simba::scalar::RealField;

use crate::ComplexHelper;
use na::allocator::Allocator;
use na::dimension::{Const, Dim};
use na::storage::Storage;
use na::{DefaultAllocator, Matrix, OMatrix, OVector, Scalar};

use lapack;

/// Eigendecomposition of a real square matrix with real eigenvalues.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(
        bound(serialize = "DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>,
         OVector<T, D>: Serialize,
         OMatrix<T, D, D>: Serialize")
    )
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(
        bound(deserialize = "DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>,
         OVector<T, D>: Serialize,
         OMatrix<T, D, D>: Deserialize<'de>")
    )
)]
#[derive(Clone, Debug)]
pub struct Schur<T: Scalar, D: Dim>
where
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    re: OVector<T, D>,
    im: OVector<T, D>,
    t: OMatrix<T, D, D>,
    q: OMatrix<T, D, D>,
}

impl<T: Scalar + Copy, D: Dim> Copy for Schur<T, D>
where
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>,
    OMatrix<T, D, D>: Copy,
    OVector<T, D>: Copy,
{
}

impl<T: SchurScalar + RealField, D: Dim> Schur<T, D>
where
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, D>,
{
    /// Computes the eigenvalues and real Schur form of the matrix `m`.
    ///
    /// Panics if the method did not converge.
    pub fn new(m: OMatrix<T, D, D>) -> Self {
        Self::try_new(m).expect("Schur decomposition: convergence failed.")
    }

    /// Computes the eigenvalues and real Schur form of the matrix `m`.
    ///
    /// Returns `None` if the method did not converge.
    pub fn try_new(mut m: OMatrix<T, D, D>) -> Option<Self> {
        assert!(
            m.is_square(),
            "Unable to compute the eigenvalue decomposition of a non-square matrix."
        );

        let (nrows, ncols) = m.data.shape();
        let n = nrows.value();

        let lda = n as i32;

        let mut info = 0;

        let mut wr = unsafe { Matrix::new_uninitialized_generic(nrows, Const::<1>).assume_init() };
        let mut wi = unsafe { Matrix::new_uninitialized_generic(nrows, Const::<1>).assume_init() };
        let mut q = unsafe { Matrix::new_uninitialized_generic(nrows, ncols).assume_init() };
        // Placeholders:
        let mut bwork = [0i32];
        let mut unused = 0;

        let lwork = T::xgees_work_size(
            b'V',
            b'T',
            n as i32,
            m.as_mut_slice(),
            lda,
            &mut unused,
            wr.as_mut_slice(),
            wi.as_mut_slice(),
            q.as_mut_slice(),
            n as i32,
            &mut bwork,
            &mut info,
        );
        lapack_check!(info);

        let mut work = unsafe { crate::uninitialized_vec(lwork as usize) };

        T::xgees(
            b'V',
            b'T',
            n as i32,
            m.as_mut_slice(),
            lda,
            &mut unused,
            wr.as_mut_slice(),
            wi.as_mut_slice(),
            q.as_mut_slice(),
            n as i32,
            &mut work,
            lwork,
            &mut bwork,
            &mut info,
        );
        lapack_check!(info);

        Some(Schur {
            re: wr,
            im: wi,
            t: m,
            q: q,
        })
    }

    /// Retrieves the unitary matrix `Q` and the upper-quasitriangular matrix `T` such that the
    /// decomposed matrix equals `Q * T * Q.transpose()`.
    pub fn unpack(self) -> (OMatrix<T, D, D>, OMatrix<T, D, D>) {
        (self.q, self.t)
    }

    /// Computes the real eigenvalues of the decomposed matrix.
    ///
    /// Return `None` if some eigenvalues are complex.
    #[must_use]
    pub fn eigenvalues(&self) -> Option<OVector<T, D>> {
        if self.im.iter().all(|e| e.is_zero()) {
            Some(self.re.clone())
        } else {
            None
        }
    }

    /// Computes the complex eigenvalues of the decomposed matrix.
    #[must_use]
    pub fn complex_eigenvalues(&self) -> OVector<Complex<T>, D>
    where
        DefaultAllocator: Allocator<Complex<T>, D>,
    {
        let mut out =
            unsafe { OVector::new_uninitialized_generic(self.t.data.shape().0, Const::<1>) };

        for i in 0..out.len() {
            out[i] = MaybeUninit::new(Complex::new(self.re[i], self.im[i]));
        }

        // Safety: all entries have been initialized.
        unsafe { out.assume_init() }
    }
}

/*
 *
 * Lapack functions dispatch.
 *
 */
/// Trait implemented by scalars for which Lapack implements the RealField Schur decomposition.
pub trait SchurScalar: Scalar {
    #[allow(missing_docs)]
    fn xgees(
        jobvs: u8,
        sort: u8,
        // select: ???
        n: i32,
        a: &mut [Self],
        lda: i32,
        sdim: &mut i32,
        wr: &mut [Self],
        wi: &mut [Self],
        vs: &mut [Self],
        ldvs: i32,
        work: &mut [Self],
        lwork: i32,
        bwork: &mut [i32],
        info: &mut i32,
    );

    #[allow(missing_docs)]
    fn xgees_work_size(
        jobvs: u8,
        sort: u8,
        // select: ???
        n: i32,
        a: &mut [Self],
        lda: i32,
        sdim: &mut i32,
        wr: &mut [Self],
        wi: &mut [Self],
        vs: &mut [Self],
        ldvs: i32,
        bwork: &mut [i32],
        info: &mut i32,
    ) -> i32;
}

macro_rules! real_eigensystem_scalar_impl (
    ($N: ty, $xgees: path) => (
        impl SchurScalar for $N {
            #[inline]
            fn xgees(jobvs:  u8,
                     sort:   u8,
                     // select: ???
                     n:      i32,
                     a:      &mut [$N],
                     lda:    i32,
                     sdim:   &mut i32,
                     wr:     &mut [$N],
                     wi:     &mut [$N],
                     vs:     &mut [$N],
                     ldvs:   i32,
                     work:   &mut [$N],
                     lwork:  i32,
                     bwork:  &mut [i32],
                     info:   &mut i32) {
                unsafe { $xgees(jobvs, sort, None, n, a, lda, sdim, wr, wi, vs, ldvs, work, lwork, bwork, info); }
            }


            #[inline]
            fn xgees_work_size(jobvs:  u8,
                               sort:   u8,
                               // select: ???
                               n:      i32,
                               a:      &mut [$N],
                               lda:    i32,
                               sdim:   &mut i32,
                               wr:     &mut [$N],
                               wi:     &mut [$N],
                               vs:     &mut [$N],
                               ldvs:   i32,
                               bwork:  &mut [i32],
                               info:   &mut i32)
                               -> i32 {
                let mut work = [ Zero::zero() ];
                let lwork    = -1 as i32;

                unsafe { $xgees(jobvs, sort, None, n, a, lda, sdim, wr, wi, vs, ldvs, &mut work, lwork, bwork, info); }
                ComplexHelper::real_part(work[0]) as i32
            }
        }
    )
);

real_eigensystem_scalar_impl!(f32, lapack::sgees);
real_eigensystem_scalar_impl!(f64, lapack::dgees);
