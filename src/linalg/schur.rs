#![allow(clippy::suspicious_operation_groupings)]
use std::cmp;
use std::fmt;
use std::mem::MaybeUninit;

#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

use approx::AbsDiffEq;
use num_complex::Complex as NumComplex;
use simba::scalar::{ComplexField, RealField};

use crate::allocator::Allocator;
use crate::base::dimension::{Const, Dim, DimDiff, DimName, DimSub, Dynamic, U1, U2};
use crate::base::storage::{InnerOwned, Storage};
use crate::base::{
    DefaultAllocator, OMatrix, OVector, Owned, SquareMatrix, Unit, Vector2, Vector3,
};

use crate::geometry::Reflection;
use crate::linalg::givens::GivensRotation;
use crate::linalg::householder;
use crate::linalg::Hessenberg;

/// Schur decomposition of a square matrix.
///
/// If this is a real matrix, this will be a RealField Schur decomposition.
#[cfg_attr(feature = "serde-serialize-no-std", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(serialize = "DefaultAllocator: Allocator<T, D, D>,
         OMatrix<T, D, D>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(deserialize = "DefaultAllocator: Allocator<T, D, D>,
         OMatrix<T, D, D>: Deserialize<'de>"))
)]
pub struct Schur<T, D: Dim>
where
    DefaultAllocator: Allocator<T, D, D>,
{
    q: OMatrix<T, D, D>,
    t: OMatrix<T, D, D>,
}

impl<T: Copy, D: DimName> Copy for Schur<T, D>
where
    DefaultAllocator: Allocator<T, D, D>,
    Owned<T, D, D>: Copy,
{
}

impl<T: Clone, D: Dim> Clone for Schur<T, D>
where
    DefaultAllocator: Allocator<T, D, D>,
    InnerOwned<T, D, D>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            q: self.q.clone(),
            t: self.t.clone(),
        }
    }
}

impl<T: fmt::Debug, D: Dim> fmt::Debug for Schur<T, D>
where
    DefaultAllocator: Allocator<T, D, D>,
    InnerOwned<T, D, D>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Schur")
            .field("q", &self.q)
            .field("t", &self.t)
            .finish()
    }
}

impl<T: ComplexField, D: Dim> Schur<T, D>
where
    D: DimSub<U1>, // For Hessenberg.
    DefaultAllocator: Allocator<T, D, DimDiff<D, U1>>
        + Allocator<T, DimDiff<D, U1>>
        + Allocator<T, D, D>
        + Allocator<T, D>,
{
    /// Computes the Schur decomposition of a square matrix.
    pub fn new(m: OMatrix<T, D, D>) -> Self {
        Self::try_new(m, T::RealField::default_epsilon(), 0).unwrap()
    }

    /// Attempts to compute the Schur decomposition of a square matrix.
    ///
    /// If only eigenvalues are needed, it is more efficient to call the matrix method
    /// `.eigenvalues()` instead.
    ///
    /// # Arguments
    ///
    /// * `eps`       − tolerance used to determine when a value converged to 0.
    /// * `max_niter` − maximum total number of iterations performed by the algorithm. If this
    /// number of iteration is exceeded, `None` is returned. If `niter == 0`, then the algorithm
    /// continues indefinitely until convergence.
    pub fn try_new(m: OMatrix<T, D, D>, eps: T::RealField, max_niter: usize) -> Option<Self> {
        let mut work = OVector::new_uninitialized_generic(m.data.shape().0, Const::<1>);

        Self::do_decompose(m, &mut work, eps, max_niter, true)
            .map(|(q, t)| Schur { q: q.unwrap(), t })
    }

    fn do_decompose(
        mut m: OMatrix<T, D, D>,
        work: &mut OVector<MaybeUninit<T>, D>,
        eps: T::RealField,
        max_niter: usize,
        compute_q: bool,
    ) -> Option<(Option<OMatrix<T, D, D>>, OMatrix<T, D, D>)> {
        assert!(
            m.is_square(),
            "Unable to compute the eigenvectors and eigenvalues of a non-square matrix."
        );

        let dim = m.data.shape().0;

        // Specialization would make this easier.
        if dim.value() == 0 {
            let vecs = Some(OMatrix::from_element_generic(dim, dim, T::zero()));
            let vals = OMatrix::from_element_generic(dim, dim, T::zero());
            return Some((vecs, vals));
        } else if dim.value() == 1 {
            if compute_q {
                let q = OMatrix::from_element_generic(dim, dim, T::one());
                return Some((Some(q), m));
            } else {
                return Some((None, m));
            }
        } else if dim.value() == 2 {
            return decompose_2x2(m, compute_q);
        }

        let amax_m = m.camax();
        m.unscale_mut(amax_m);

        let hess = Hessenberg::new_with_workspace(m, work);
        let mut q;
        let mut t;

        if compute_q {
            // TODO: could we work without unpacking? Using only the internal representation of
            // hessenberg decomposition.
            let (vecs, vals) = hess.unpack();
            q = Some(vecs);
            t = vals;
        } else {
            q = None;
            t = hess.unpack_h()
        }

        // Implicit double-shift QR method.
        let mut niter = 0;
        let (mut start, mut end) = Self::delimit_subproblem(&mut t, eps, dim.value() - 1);

        while end != start {
            let subdim = end - start + 1;

            if subdim > 2 {
                let m = end - 1;
                let n = end;

                let h11 = t[(start, start)];
                let h12 = t[(start, start + 1)];
                let h21 = t[(start + 1, start)];
                let h22 = t[(start + 1, start + 1)];
                let h32 = t[(start + 2, start + 1)];

                let hnn = t[(n, n)];
                let hmm = t[(m, m)];
                let hnm = t[(n, m)];
                let hmn = t[(m, n)];

                let tra = hnn + hmm;
                let det = hnn * hmm - hnm * hmn;

                let mut axis = Vector3::new(
                    h11 * h11 + h12 * h21 - tra * h11 + det,
                    h21 * (h11 + h22 - tra),
                    h21 * h32,
                );

                for k in start..n - 1 {
                    let (norm, not_zero) = householder::reflection_axis_mut(&mut axis);

                    if not_zero {
                        if k > start {
                            t[(k, k - 1)] = norm;
                            t[(k + 1, k - 1)] = T::zero();
                            t[(k + 2, k - 1)] = T::zero();
                        }

                        let refl = Reflection::new(Unit::new_unchecked(axis), T::zero());

                        {
                            let krows = cmp::min(k + 4, end + 1);
                            let mut work = work.rows_mut(0, krows);
                            refl.reflect(&mut t.generic_slice_mut(
                                (k, k),
                                (Const::<3>, Dynamic::new(dim.value() - k)),
                            ));
                            refl.reflect_rows(
                                &mut t.generic_slice_mut((0, k), (Dynamic::new(krows), Const::<3>)),
                                &mut work,
                            );
                        }

                        if let Some(ref mut q) = q {
                            refl.reflect_rows(
                                &mut q.generic_slice_mut((0, k), (dim, Const::<3>)),
                                work,
                            );
                        }
                    }

                    axis.x = t[(k + 1, k)];
                    axis.y = t[(k + 2, k)];

                    if k < n - 2 {
                        axis.z = t[(k + 3, k)];
                    }
                }

                let mut axis = Vector2::new(axis.x, axis.y);
                let (norm, not_zero) = householder::reflection_axis_mut(&mut axis);

                if not_zero {
                    let refl = Reflection::new(Unit::new_unchecked(axis), T::zero());

                    t[(m, m - 1)] = norm;
                    t[(n, m - 1)] = T::zero();

                    {
                        let mut work = work.rows_mut(0, end + 1);
                        refl.reflect(&mut t.generic_slice_mut(
                            (m, m),
                            (Const::<2>, Dynamic::new(dim.value() - m)),
                        ));
                        refl.reflect_rows(
                            &mut t.generic_slice_mut((0, m), (Dynamic::new(end + 1), Const::<2>)),
                            &mut work,
                        );
                    }

                    if let Some(ref mut q) = q {
                        refl.reflect_rows(
                            &mut q.generic_slice_mut((0, m), (dim, Const::<2>)),
                            work,
                        );
                    }
                }
            } else {
                // Decouple the 2x2 block if it has real eigenvalues.
                if let Some(rot) = compute_2x2_basis(&t.fixed_slice::<2, 2>(start, start)) {
                    let inv_rot = rot.inverse();
                    inv_rot.rotate(&mut t.generic_slice_mut(
                        (start, start),
                        (Const::<2>, Dynamic::new(dim.value() - start)),
                    ));
                    rot.rotate_rows(
                        &mut t.generic_slice_mut((0, start), (Dynamic::new(end + 1), Const::<2>)),
                    );
                    t[(end, start)] = T::zero();

                    if let Some(ref mut q) = q {
                        rot.rotate_rows(&mut q.generic_slice_mut((0, start), (dim, Const::<2>)));
                    }
                }

                // Check if we reached the beginning of the matrix.
                if end > 2 {
                    end -= 2;
                } else {
                    break;
                }
            }

            let sub = Self::delimit_subproblem(&mut t, eps, end);

            start = sub.0;
            end = sub.1;

            niter += 1;
            if niter == max_niter {
                return None;
            }
        }

        t.scale_mut(amax_m);

        Some((q, t))
    }

    /// Computes the eigenvalues of the decomposed matrix.
    fn do_eigenvalues(t: &OMatrix<T, D, D>, out: &mut OVector<MaybeUninit<T>, D>) -> bool {
        // TODO: check dropping stuff.

        let dim = t.nrows();
        let mut m = 0;

        while m < dim - 1 {
            let n = m + 1;

            if t[(n, m)].is_zero() {
                out[m] = MaybeUninit::new(t[(m, m)]);
                m += 1;
            } else {
                // Complex eigenvalue.
                return false;
            }
        }

        if m == dim - 1 {
            out[m] = MaybeUninit::new(t[(m, m)]);
        }

        true
    }

    /// Computes the complex eigenvalues of the decomposed matrix.
    fn do_complex_eigenvalues(
        t: &OMatrix<T, D, D>,
        out: &mut OVector<MaybeUninit<NumComplex<T>>, D>,
    ) where
        T: RealField,
        DefaultAllocator: Allocator<NumComplex<T>, D>,
    {
        // TODO: check for dropping behavior.

        let dim = t.nrows();
        let mut m = 0;

        while m < dim - 1 {
            let n = m + 1;

            if t[(n, m)].is_zero() {
                out[m] = MaybeUninit::new(NumComplex::new(t[(m, m)], T::zero()));
                m += 1;
            } else {
                // Solve the 2x2 eigenvalue subproblem.
                let hmm = t[(m, m)];
                let hnm = t[(n, m)];
                let hmn = t[(m, n)];
                let hnn = t[(n, n)];

                // NOTE: use the same algorithm as in compute_2x2_eigvals.
                let val = (hmm - hnn) * crate::convert(0.5);
                let discr = hnm * hmn + val * val;

                // All 2x2 blocks have negative discriminant because we already decoupled those
                // with positive eigenvalues.
                let sqrt_discr = NumComplex::new(T::zero(), (-discr).sqrt());

                let half_tra = (hnn + hmm) * crate::convert(0.5);
                out[m] = MaybeUninit::new(NumComplex::new(half_tra, T::zero()) + sqrt_discr);
                out[m + 1] = MaybeUninit::new(NumComplex::new(half_tra, T::zero()) - sqrt_discr);

                m += 2;
            }
        }

        if m == dim - 1 {
            out[m] = MaybeUninit::new(NumComplex::new(t[(m, m)], T::zero()));
        }
    }

    fn delimit_subproblem(t: &mut OMatrix<T, D, D>, eps: T::RealField, end: usize) -> (usize, usize)
    where
        D: DimSub<U1>,
        DefaultAllocator: Allocator<T, DimDiff<D, U1>>,
    {
        let mut n = end;

        while n > 0 {
            let m = n - 1;

            if t[(n, m)].norm1() <= eps * (t[(n, n)].norm1() + t[(m, m)].norm1()) {
                t[(n, m)] = T::zero();
            } else {
                break;
            }

            n -= 1;
        }

        if n == 0 {
            return (0, 0);
        }

        let mut new_start = n - 1;
        while new_start > 0 {
            let m = new_start - 1;

            let off_diag = t[(new_start, m)];
            if off_diag.is_zero()
                || off_diag.norm1() <= eps * (t[(new_start, new_start)].norm1() + t[(m, m)].norm1())
            {
                t[(new_start, m)] = T::zero();
                break;
            }

            new_start -= 1;
        }

        (new_start, n)
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
        let mut out = OVector::new_uninitialized_generic(self.t.data.shape().0, Const::<1>);
        if Self::do_eigenvalues(&self.t, &mut out) {
            Some(unsafe { out.assume_init() })
        } else {
            None
        }
    }

    /// Computes the complex eigenvalues of the decomposed matrix.
    #[must_use]
    pub fn complex_eigenvalues(&self) -> OVector<NumComplex<T>, D>
    where
        T: RealField,
        DefaultAllocator: Allocator<NumComplex<T>, D>,
    {
        let mut out = OVector::new_uninitialized_generic(self.t.data.shape().0, Const::<1>);
        Self::do_complex_eigenvalues(&self.t, &mut out);
        unsafe { out.assume_init() }
    }
}

fn decompose_2x2<T: ComplexField, D: Dim>(
    mut m: OMatrix<T, D, D>,
    compute_q: bool,
) -> Option<(Option<OMatrix<T, D, D>>, OMatrix<T, D, D>)>
where
    DefaultAllocator: Allocator<T, D, D>,
{
    let dim = m.data.shape().0;
    let mut q = None;
    match compute_2x2_basis(&m.fixed_slice::<2, 2>(0, 0)) {
        Some(rot) => {
            let mut m = m.fixed_slice_mut::<2, 2>(0, 0);
            let inv_rot = rot.inverse();
            inv_rot.rotate(&mut m);
            rot.rotate_rows(&mut m);
            m[(1, 0)] = T::zero();

            if compute_q {
                // XXX: we have to build the matrix manually because
                // rot.to_rotation_matrix().unwrap() causes an ICE.
                let c = T::from_real(rot.c());
                q = Some(OMatrix::from_column_slice_generic(
                    dim,
                    dim,
                    &[c, rot.s(), -rot.s().conjugate(), c],
                ));
            }
        }
        None => {
            if compute_q {
                q = Some(OMatrix::identity_generic(dim, dim));
            }
        }
    };

    Some((q, m))
}

fn compute_2x2_eigvals<T: ComplexField, S: Storage<T, U2, U2>>(
    m: &SquareMatrix<T, U2, S>,
) -> Option<(T, T)> {
    // Solve the 2x2 eigenvalue subproblem.
    let h00 = m[(0, 0)];
    let h10 = m[(1, 0)];
    let h01 = m[(0, 1)];
    let h11 = m[(1, 1)];

    // NOTE: this discriminant computation is more stable than the
    // one based on the trace and determinant: 0.25 * tra * tra - det
    // because it ensures positiveness for symmetric matrices.
    let val = (h00 - h11) * crate::convert(0.5);
    let discr = h10 * h01 + val * val;

    discr.try_sqrt().map(|sqrt_discr| {
        let half_tra = (h00 + h11) * crate::convert(0.5);
        (half_tra + sqrt_discr, half_tra - sqrt_discr)
    })
}

// Computes the 2x2 transformation that upper-triangulates a 2x2 matrix with real eigenvalues.
/// Computes the singular vectors for a 2x2 matrix.
///
/// Returns `None` if the matrix has complex eigenvalues, or is upper-triangular. In both case,
/// the basis is the identity.
fn compute_2x2_basis<T: ComplexField, S: Storage<T, U2, U2>>(
    m: &SquareMatrix<T, U2, S>,
) -> Option<GivensRotation<T>> {
    let h10 = m[(1, 0)];

    if h10.is_zero() {
        return None;
    }

    if let Some((eigval1, eigval2)) = compute_2x2_eigvals(m) {
        let x1 = eigval1 - m[(1, 1)];
        let x2 = eigval2 - m[(1, 1)];

        // NOTE: Choose the one that yields a larger x component.
        // This is necessary for numerical stability of the normalization of the complex
        // number.
        if x1.norm1() > x2.norm1() {
            Some(GivensRotation::new(x1, h10).0)
        } else {
            Some(GivensRotation::new(x2, h10).0)
        }
    } else {
        None
    }
}

impl<T: ComplexField, D: Dim, S: Storage<T, D, D>> SquareMatrix<T, D, S>
where
    D: DimSub<U1>, // For Hessenberg.
    DefaultAllocator: Allocator<T, D, DimDiff<D, U1>>
        + Allocator<T, DimDiff<D, U1>>
        + Allocator<T, D, D>
        + Allocator<T, D>,
{
    /// Computes the eigenvalues of this matrix.
    #[must_use]
    pub fn eigenvalues(&self) -> Option<OVector<T, D>> {
        // TODO: check drop stuff.

        assert!(
            self.is_square(),
            "Unable to compute eigenvalues of a non-square matrix."
        );

        let mut work = OVector::new_uninitialized_generic(self.data.shape().0, Const::<1>);

        // Special case for 2x2 matrices.
        if self.nrows() == 2 {
            // TODO: can we avoid this slicing
            // (which is needed here just to transform D to U2)?
            let me = self.fixed_slice::<2, 2>(0, 0);
            return match compute_2x2_eigvals(&me) {
                Some((a, b)) => {
                    work[0] = MaybeUninit::new(a);
                    work[1] = MaybeUninit::new(b);
                    Some(unsafe { work.assume_init() })
                }
                None => None,
            };
        }

        // TODO: add balancing?
        let schur = Schur::do_decompose(
            self.clone_owned(),
            &mut work,
            T::RealField::default_epsilon(),
            0,
            false,
        )
        .unwrap();
        if Schur::do_eigenvalues(&schur.1, &mut work) {
            Some(unsafe { work.assume_init() })
        } else {
            None
        }
    }

    /// Computes the eigenvalues of this matrix.
    #[must_use]
    pub fn complex_eigenvalues(&self) -> OVector<NumComplex<T>, D>
    // TODO: add balancing?
    where
        T: RealField,
        DefaultAllocator: Allocator<NumComplex<T>, D>,
    {
        let dim = self.data.shape().0;
        let mut work = OVector::new_uninitialized_generic(dim, Const::<1>);

        let schur = Schur::do_decompose(
            self.clone_owned(),
            &mut work,
            T::default_epsilon(),
            0,
            false,
        )
        .unwrap();
        let mut eig = OVector::new_uninitialized_generic(dim, Const::<1>);
        Schur::do_complex_eigenvalues(&schur.1, &mut eig);
        unsafe { eig.assume_init() }
    }
}
