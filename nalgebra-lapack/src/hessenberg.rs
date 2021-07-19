use num::Zero;
use num_complex::Complex;

use crate::ComplexHelper;
use na::allocator::Allocator;
use na::dimension::{Const, DimDiff, DimSub, U1};
use na::storage::Storage;
use na::{DefaultAllocator, Matrix, OMatrix, OVector, Scalar};

use lapack;

/// The Hessenberg decomposition of a general matrix.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(serialize = "DefaultAllocator: Allocator<T, D, D> +
                           Allocator<T, DimDiff<D, U1>>,
         OMatrix<T, D, D>: Serialize,
         OVector<T, DimDiff<D, U1>>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(deserialize = "DefaultAllocator: Allocator<T, D, D> +
                           Allocator<T, DimDiff<D, U1>>,
         OMatrix<T, D, D>: Deserialize<'de>,
         OVector<T, DimDiff<D, U1>>: Deserialize<'de>"))
)]
#[derive(Clone, Debug)]
pub struct Hessenberg<T: Scalar, D: DimSub<U1>>
where
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, DimDiff<D, U1>>,
{
    h: OMatrix<T, D, D>,
    tau: OVector<T, DimDiff<D, U1>>,
}

impl<T: Scalar + Copy, D: DimSub<U1>> Copy for Hessenberg<T, D>
where
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, DimDiff<D, U1>>,
    OMatrix<T, D, D>: Copy,
    OVector<T, DimDiff<D, U1>>: Copy,
{
}

impl<T: HessenbergScalar + Zero, D: DimSub<U1>> Hessenberg<T, D>
where
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, DimDiff<D, U1>>,
{
    /// Computes the hessenberg decomposition of the matrix `m`.
    pub fn new(mut m: OMatrix<T, D, D>) -> Self {
        let nrows = m.data.shape().0;
        let n = nrows.value() as i32;

        assert!(
            m.is_square(),
            "Unable to compute the hessenberg decomposition of a non-square matrix."
        );
        assert!(
            !m.is_empty(),
            "Unable to compute the hessenberg decomposition of an empty matrix."
        );

        // IMPORTANT TODO: this is still UB.
        let mut tau = unsafe {
            Matrix::new_uninitialized_generic(nrows.sub(Const::<1>), Const::<1>).assume_init()
        };

        let mut info = 0;
        let lwork =
            T::xgehrd_work_size(n, 1, n, m.as_mut_slice(), n, tau.as_mut_slice(), &mut info);
        let mut work = unsafe { crate::uninitialized_vec(lwork as usize) };

        lapack_panic!(info);

        T::xgehrd(
            n,
            1,
            n,
            m.as_mut_slice(),
            n,
            tau.as_mut_slice(),
            &mut work,
            lwork,
            &mut info,
        );
        lapack_panic!(info);

        Self { h: m, tau: tau }
    }

    /// Computes the hessenberg matrix of this decomposition.
    #[inline]
    #[must_use]
    pub fn h(&self) -> OMatrix<T, D, D> {
        let mut h = self.h.clone_owned();
        h.fill_lower_triangle(T::zero(), 2);

        h
    }
}

impl<T: HessenbergReal + Zero, D: DimSub<U1>> Hessenberg<T, D>
where
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, DimDiff<D, U1>>,
{
    /// Computes the matrices `(Q, H)` of this decomposition.
    #[inline]
    pub fn unpack(self) -> (OMatrix<T, D, D>, OMatrix<T, D, D>) {
        (self.q(), self.h())
    }

    /// Computes the unitary matrix `Q` of this decomposition.
    #[inline]
    #[must_use]
    pub fn q(&self) -> OMatrix<T, D, D> {
        let n = self.h.nrows() as i32;
        let mut q = self.h.clone_owned();
        let mut info = 0;

        let lwork =
            T::xorghr_work_size(n, 1, n, q.as_mut_slice(), n, self.tau.as_slice(), &mut info);
        let mut work = vec![T::zero(); lwork as usize];

        T::xorghr(
            n,
            1,
            n,
            q.as_mut_slice(),
            n,
            self.tau.as_slice(),
            &mut work,
            lwork,
            &mut info,
        );

        q
    }
}

/*
 *
 * Lapack functions dispatch.
 *
 */
pub trait HessenbergScalar: Scalar + Copy {
    fn xgehrd(
        n: i32,
        ilo: i32,
        ihi: i32,
        a: &mut [Self],
        lda: i32,
        tau: &mut [Self],
        work: &mut [Self],
        lwork: i32,
        info: &mut i32,
    );
    fn xgehrd_work_size(
        n: i32,
        ilo: i32,
        ihi: i32,
        a: &mut [Self],
        lda: i32,
        tau: &mut [Self],
        info: &mut i32,
    ) -> i32;
}

/// Trait implemented by scalars for which Lapack implements the hessenberg decomposition.
pub trait HessenbergReal: HessenbergScalar {
    #[allow(missing_docs)]
    fn xorghr(
        n: i32,
        ilo: i32,
        ihi: i32,
        a: &mut [Self],
        lda: i32,
        tau: &[Self],
        work: &mut [Self],
        lwork: i32,
        info: &mut i32,
    );
    #[allow(missing_docs)]
    fn xorghr_work_size(
        n: i32,
        ilo: i32,
        ihi: i32,
        a: &mut [Self],
        lda: i32,
        tau: &[Self],
        info: &mut i32,
    ) -> i32;
}

macro_rules! hessenberg_scalar_impl(
    ($N: ty, $xgehrd: path) => (
        impl HessenbergScalar for $N {
            #[inline]
            fn xgehrd(n: i32, ilo: i32, ihi: i32, a: &mut [Self], lda: i32,
                      tau: &mut [Self], work: &mut [Self], lwork: i32, info: &mut i32) {
                unsafe { $xgehrd(n, ilo, ihi, a, lda, tau, work, lwork, info) }
            }

            #[inline]
            fn xgehrd_work_size(n: i32, ilo: i32, ihi: i32, a: &mut [Self], lda: i32,
                                tau: &mut [Self], info: &mut i32) -> i32 {
                let mut work = [ Zero::zero() ];
                let lwork = -1 as i32;

                unsafe { $xgehrd(n, ilo, ihi, a, lda, tau, &mut work, lwork, info) };
                ComplexHelper::real_part(work[0]) as i32
            }
        }
    )
);

macro_rules! hessenberg_real_impl(
    ($N: ty, $xorghr: path) => (
        impl HessenbergReal for $N {
            #[inline]
            fn xorghr(n: i32, ilo: i32, ihi: i32, a: &mut [Self], lda: i32, tau: &[Self],
                      work: &mut [Self], lwork: i32, info: &mut i32) {
                unsafe { $xorghr(n, ilo, ihi, a, lda, tau, work, lwork, info) }
            }

            #[inline]
            fn xorghr_work_size(n: i32, ilo: i32, ihi: i32, a: &mut [Self], lda: i32,
                                tau: &[Self], info: &mut i32) -> i32 {
                let mut work = [ Zero::zero() ];
                let lwork = -1 as i32;

                unsafe { $xorghr(n, ilo, ihi, a, lda, tau, &mut work, lwork, info) };
                ComplexHelper::real_part(work[0]) as i32
            }
        }
    )
);

hessenberg_scalar_impl!(f32, lapack::sgehrd);
hessenberg_scalar_impl!(f64, lapack::dgehrd);
hessenberg_scalar_impl!(Complex<f32>, lapack::cgehrd);
hessenberg_scalar_impl!(Complex<f64>, lapack::zgehrd);

hessenberg_real_impl!(f32, lapack::sorghr);
hessenberg_real_impl!(f64, lapack::dorghr);
