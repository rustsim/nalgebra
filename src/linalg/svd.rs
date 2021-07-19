use std::fmt;

#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

use approx::AbsDiffEq;
use num::{One, Zero};

use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, Matrix, Matrix2x3, OMatrix, OVector, Vector2};
use crate::constraint::{SameNumberOfRows, ShapeConstraint};
use crate::dimension::{Dim, DimDiff, DimMin, DimMinimum, DimName, DimSub, U1};
use crate::storage::{InnerOwned, Storage};
use simba::scalar::{ComplexField, RealField};

use crate::linalg::givens::GivensRotation;
use crate::linalg::symmetric_eigen;
use crate::linalg::Bidiagonal;

/// Singular Value Decomposition of a general matrix.
#[cfg_attr(feature = "serde-serialize-no-std", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(
        serialize = "DefaultAllocator: Allocator<T::RealField, DimMinimum<R, C>>    +
                           Allocator<T, DimMinimum<R, C>, C> +
                           Allocator<T, R, DimMinimum<R, C>>,
         OMatrix<T, R, DimMinimum<R, C>>: Serialize,
         OMatrix<T, DimMinimum<R, C>, C>: Serialize,
         OVector<T::RealField, DimMinimum<R, C>>: Serialize"
    ))
)]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(
        deserialize = "DefaultAllocator: Allocator<T::RealField, DimMinimum<R, C>>    +
                           Allocator<T, DimMinimum<R, C>, C> +
                           Allocator<T, R, DimMinimum<R, C>>,
         OMatrix<T, R, DimMinimum<R, C>>: Deserialize<'de>,
         OMatrix<T, DimMinimum<R, C>, C>: Deserialize<'de>,
         OVector<T::RealField, DimMinimum<R, C>>: Deserialize<'de>"
    ))
)]
pub struct SVD<T: ComplexField, R: DimMin<C>, C: Dim>
where
    DefaultAllocator: Allocator<T, DimMinimum<R, C>, C>
        + Allocator<T, R, DimMinimum<R, C>>
        + Allocator<T::RealField, DimMinimum<R, C>>,
{
    /// The left-singular vectors `U` of this SVD.
    pub u: Option<OMatrix<T, R, DimMinimum<R, C>>>,
    /// The right-singular vectors `V^t` of this SVD.
    pub v_t: Option<OMatrix<T, DimMinimum<R, C>, C>>,
    /// The singular values of this SVD.
    pub singular_values: OVector<T::RealField, DimMinimum<R, C>>,
}

impl<T: ComplexField + Copy, R: DimName + DimMin<C>, C: DimName> Copy for SVD<T, R, C>
where
    DefaultAllocator: Allocator<T, DimMinimum<R, C>, C>
        + Allocator<T, R, DimMinimum<R, C>>
        + Allocator<T::RealField, DimMinimum<R, C>>,
    InnerOwned<T, R, DimMinimum<R, C>>: Copy,
    InnerOwned<T, DimMinimum<R, C>, C>: Copy,
    InnerOwned<T::RealField, DimMinimum<R, C>>: Copy,
{
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> Clone for SVD<T, R, C>
where
    DefaultAllocator: Allocator<T, DimMinimum<R, C>, C>
        + Allocator<T, R, DimMinimum<R, C>>
        + Allocator<T::RealField, DimMinimum<R, C>>,
    InnerOwned<T, R, DimMinimum<R, C>>: Clone,
    InnerOwned<T, DimMinimum<R, C>, C>: Clone,
    InnerOwned<T::RealField, DimMinimum<R, C>>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            u: self.u.clone(),
            v_t: self.v_t.clone(),
            singular_values: self.singular_values.clone(),
        }
    }
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> fmt::Debug for SVD<T, R, C>
where
    DefaultAllocator: Allocator<T, DimMinimum<R, C>, C>
        + Allocator<T, R, DimMinimum<R, C>>
        + Allocator<T::RealField, DimMinimum<R, C>>,
    InnerOwned<T, R, DimMinimum<R, C>>: fmt::Debug,
    InnerOwned<T, DimMinimum<R, C>, C>: fmt::Debug,
    InnerOwned<T::RealField, DimMinimum<R, C>>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SVD")
            .field("u", &self.u)
            .field("v_t", &self.v_t)
            .field("singular_values", &self.singular_values)
            .finish()
    }
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> SVD<T, R, C>
where
    DimMinimum<R, C>: DimSub<U1>, // for Bidiagonal.
    DefaultAllocator: Allocator<T, R, C>
        + Allocator<T, C>
        + Allocator<T, R>
        + Allocator<T, DimDiff<DimMinimum<R, C>, U1>>
        + Allocator<T, DimMinimum<R, C>, C>
        + Allocator<T, R, DimMinimum<R, C>>
        + Allocator<T, DimMinimum<R, C>>
        + Allocator<T::RealField, DimMinimum<R, C>>
        + Allocator<T::RealField, DimDiff<DimMinimum<R, C>, U1>>,
{
    /// Computes the Singular Value Decomposition of `matrix` using implicit shift.
    pub fn new(matrix: OMatrix<T, R, C>, compute_u: bool, compute_v: bool) -> Self {
        Self::try_new(
            matrix,
            compute_u,
            compute_v,
            T::RealField::default_epsilon(),
            0,
        )
        .unwrap()
    }

    /// Attempts to compute the Singular Value Decomposition of `matrix` using implicit shift.
    ///
    /// # Arguments
    ///
    /// * `compute_u` − set this to `true` to enable the computation of left-singular vectors.
    /// * `compute_v` − set this to `true` to enable the computation of right-singular vectors.
    /// * `eps`       − tolerance used to determine when a value converged to 0.
    /// * `max_niter` − maximum total number of iterations performed by the algorithm. If this
    /// number of iteration is exceeded, `None` is returned. If `niter == 0`, then the algorithm
    /// continues indefinitely until convergence.
    pub fn try_new(
        mut matrix: OMatrix<T, R, C>,
        compute_u: bool,
        compute_v: bool,
        eps: T::RealField,
        max_niter: usize,
    ) -> Option<Self> {
        assert!(
            !matrix.is_empty(),
            "Cannot compute the SVD of an empty matrix."
        );
        let (nrows, ncols) = matrix.data.shape();
        let min_nrows_ncols = nrows.min(ncols);
        let dim = min_nrows_ncols.value();

        let m_amax = matrix.camax();

        if !m_amax.is_zero() {
            matrix.unscale_mut(m_amax);
        }

        let bi_matrix = Bidiagonal::new(matrix);
        let mut u = if compute_u { Some(bi_matrix.u()) } else { None };
        let mut v_t = if compute_v {
            Some(bi_matrix.v_t())
        } else {
            None
        };
        let mut diagonal = bi_matrix.diagonal();
        let mut off_diagonal = bi_matrix.off_diagonal();

        let mut niter = 0;
        let (mut start, mut end) = Self::delimit_subproblem(
            &mut diagonal,
            &mut off_diagonal,
            &mut u,
            &mut v_t,
            bi_matrix.is_upper_diagonal(),
            dim - 1,
            eps,
        );

        while end != start {
            let subdim = end - start + 1;

            // Solve the subproblem.
            #[allow(clippy::comparison_chain)]
            if subdim > 2 {
                let m = end - 1;
                let n = end;

                let mut vec;
                {
                    let dm = diagonal[m];
                    let dn = diagonal[n];
                    let fm = off_diagonal[m];

                    let tmm = dm * dm + off_diagonal[m - 1] * off_diagonal[m - 1];
                    let tmn = dm * fm;
                    let tnn = dn * dn + fm * fm;

                    let shift = symmetric_eigen::wilkinson_shift(tmm, tnn, tmn);

                    vec = Vector2::new(
                        diagonal[start] * diagonal[start] - shift,
                        diagonal[start] * off_diagonal[start],
                    );
                }

                for k in start..n {
                    let m12 = if k == n - 1 {
                        T::RealField::zero()
                    } else {
                        off_diagonal[k + 1]
                    };

                    let mut subm = Matrix2x3::new(
                        diagonal[k],
                        off_diagonal[k],
                        T::RealField::zero(),
                        T::RealField::zero(),
                        diagonal[k + 1],
                        m12,
                    );

                    if let Some((rot1, norm1)) = GivensRotation::cancel_y(&vec) {
                        rot1.inverse()
                            .rotate_rows(&mut subm.fixed_columns_mut::<2>(0));
                        let rot1 = GivensRotation::new_unchecked(rot1.c(), T::from_real(rot1.s()));

                        if k > start {
                            // This is not the first iteration.
                            off_diagonal[k - 1] = norm1;
                        }

                        let v = Vector2::new(subm[(0, 0)], subm[(1, 0)]);
                        // TODO: does the case `v.y == 0` ever happen?
                        let (rot2, norm2) = GivensRotation::cancel_y(&v)
                            .unwrap_or((GivensRotation::identity(), subm[(0, 0)]));

                        rot2.rotate(&mut subm.fixed_columns_mut::<2>(1));
                        let rot2 = GivensRotation::new_unchecked(rot2.c(), T::from_real(rot2.s()));

                        subm[(0, 0)] = norm2;

                        if let Some(ref mut v_t) = v_t {
                            if bi_matrix.is_upper_diagonal() {
                                rot1.rotate(&mut v_t.fixed_rows_mut::<2>(k));
                            } else {
                                rot2.rotate(&mut v_t.fixed_rows_mut::<2>(k));
                            }
                        }

                        if let Some(ref mut u) = u {
                            if bi_matrix.is_upper_diagonal() {
                                rot2.inverse().rotate_rows(&mut u.fixed_columns_mut::<2>(k));
                            } else {
                                rot1.inverse().rotate_rows(&mut u.fixed_columns_mut::<2>(k));
                            }
                        }

                        diagonal[k] = subm[(0, 0)];
                        diagonal[k + 1] = subm[(1, 1)];
                        off_diagonal[k] = subm[(0, 1)];

                        if k != n - 1 {
                            off_diagonal[k + 1] = subm[(1, 2)];
                        }

                        vec.x = subm[(0, 1)];
                        vec.y = subm[(0, 2)];
                    } else {
                        break;
                    }
                }
            } else if subdim == 2 {
                // Solve the remaining 2x2 subproblem.
                let (u2, s, v2) = compute_2x2_uptrig_svd(
                    diagonal[start],
                    off_diagonal[start],
                    diagonal[start + 1],
                    compute_u && bi_matrix.is_upper_diagonal()
                        || compute_v && !bi_matrix.is_upper_diagonal(),
                    compute_v && bi_matrix.is_upper_diagonal()
                        || compute_u && !bi_matrix.is_upper_diagonal(),
                );
                let u2 = u2.map(|u2| GivensRotation::new_unchecked(u2.c(), T::from_real(u2.s())));
                let v2 = v2.map(|v2| GivensRotation::new_unchecked(v2.c(), T::from_real(v2.s())));

                diagonal[start] = s[0];
                diagonal[start + 1] = s[1];
                off_diagonal[start] = T::RealField::zero();

                if let Some(ref mut u) = u {
                    let rot = if bi_matrix.is_upper_diagonal() {
                        u2.unwrap()
                    } else {
                        v2.unwrap()
                    };
                    rot.rotate_rows(&mut u.fixed_columns_mut::<2>(start));
                }

                if let Some(ref mut v_t) = v_t {
                    let rot = if bi_matrix.is_upper_diagonal() {
                        v2.unwrap()
                    } else {
                        u2.unwrap()
                    };
                    rot.inverse().rotate(&mut v_t.fixed_rows_mut::<2>(start));
                }

                end -= 1;
            }

            // Re-delimit the subproblem in case some decoupling occurred.
            let sub = Self::delimit_subproblem(
                &mut diagonal,
                &mut off_diagonal,
                &mut u,
                &mut v_t,
                bi_matrix.is_upper_diagonal(),
                end,
                eps,
            );
            start = sub.0;
            end = sub.1;

            niter += 1;
            if niter == max_niter {
                return None;
            }
        }

        diagonal *= m_amax;

        // Ensure all singular value are non-negative.
        for i in 0..dim {
            let sval = diagonal[i];

            if sval < T::RealField::zero() {
                diagonal[i] = -sval;

                if let Some(ref mut u) = u {
                    u.column_mut(i).neg_mut();
                }
            }
        }

        Some(Self {
            u,
            v_t,
            singular_values: diagonal,
        })
    }

    /*
    fn display_bidiag(b: &Bidiagonal<T, R, C>, begin: usize, end: usize) {
        for i in begin .. end {
            for k in begin .. i {
                print!("    ");
            }
            println!("{}  {}", b.diagonal[i], b.off_diagonal[i]);
        }
        for k in begin .. end {
            print!("    ");
        }
        println!("{}", b.diagonal[end]);
    }
    */

    fn delimit_subproblem(
        diagonal: &mut OVector<T::RealField, DimMinimum<R, C>>,
        off_diagonal: &mut OVector<T::RealField, DimDiff<DimMinimum<R, C>, U1>>,
        u: &mut Option<OMatrix<T, R, DimMinimum<R, C>>>,
        v_t: &mut Option<OMatrix<T, DimMinimum<R, C>, C>>,
        is_upper_diagonal: bool,
        end: usize,
        eps: T::RealField,
    ) -> (usize, usize) {
        let mut n = end;

        while n > 0 {
            let m = n - 1;

            if off_diagonal[m].is_zero()
                || off_diagonal[m].norm1() <= eps * (diagonal[n].norm1() + diagonal[m].norm1())
            {
                off_diagonal[m] = T::RealField::zero();
            } else if diagonal[m].norm1() <= eps {
                diagonal[m] = T::RealField::zero();
                Self::cancel_horizontal_off_diagonal_elt(
                    diagonal,
                    off_diagonal,
                    u,
                    v_t,
                    is_upper_diagonal,
                    m,
                    m + 1,
                );

                if m != 0 {
                    Self::cancel_vertical_off_diagonal_elt(
                        diagonal,
                        off_diagonal,
                        u,
                        v_t,
                        is_upper_diagonal,
                        m - 1,
                    );
                }
            } else if diagonal[n].norm1() <= eps {
                diagonal[n] = T::RealField::zero();
                Self::cancel_vertical_off_diagonal_elt(
                    diagonal,
                    off_diagonal,
                    u,
                    v_t,
                    is_upper_diagonal,
                    m,
                );
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

            if off_diagonal[m].norm1() <= eps * (diagonal[new_start].norm1() + diagonal[m].norm1())
            {
                off_diagonal[m] = T::RealField::zero();
                break;
            }
            // TODO: write a test that enters this case.
            else if diagonal[m].norm1() <= eps {
                diagonal[m] = T::RealField::zero();
                Self::cancel_horizontal_off_diagonal_elt(
                    diagonal,
                    off_diagonal,
                    u,
                    v_t,
                    is_upper_diagonal,
                    m,
                    n,
                );

                if m != 0 {
                    Self::cancel_vertical_off_diagonal_elt(
                        diagonal,
                        off_diagonal,
                        u,
                        v_t,
                        is_upper_diagonal,
                        m - 1,
                    );
                }
                break;
            }

            new_start -= 1;
        }

        (new_start, n)
    }

    // Cancels the i-th off-diagonal element using givens rotations.
    fn cancel_horizontal_off_diagonal_elt(
        diagonal: &mut OVector<T::RealField, DimMinimum<R, C>>,
        off_diagonal: &mut OVector<T::RealField, DimDiff<DimMinimum<R, C>, U1>>,
        u: &mut Option<OMatrix<T, R, DimMinimum<R, C>>>,
        v_t: &mut Option<OMatrix<T, DimMinimum<R, C>, C>>,
        is_upper_diagonal: bool,
        i: usize,
        end: usize,
    ) {
        let mut v = Vector2::new(off_diagonal[i], diagonal[i + 1]);
        off_diagonal[i] = T::RealField::zero();

        for k in i..end {
            if let Some((rot, norm)) = GivensRotation::cancel_x(&v) {
                let rot = GivensRotation::new_unchecked(rot.c(), T::from_real(rot.s()));
                diagonal[k + 1] = norm;

                if is_upper_diagonal {
                    if let Some(ref mut u) = *u {
                        rot.inverse()
                            .rotate_rows(&mut u.fixed_columns_with_step_mut::<2>(i, k - i));
                    }
                } else if let Some(ref mut v_t) = *v_t {
                    rot.rotate(&mut v_t.fixed_rows_with_step_mut::<2>(i, k - i));
                }

                if k + 1 != end {
                    v.x = -rot.s().real() * off_diagonal[k + 1];
                    v.y = diagonal[k + 2];
                    off_diagonal[k + 1] *= rot.c();
                }
            } else {
                break;
            }
        }
    }

    // Cancels the i-th off-diagonal element using givens rotations.
    fn cancel_vertical_off_diagonal_elt(
        diagonal: &mut OVector<T::RealField, DimMinimum<R, C>>,
        off_diagonal: &mut OVector<T::RealField, DimDiff<DimMinimum<R, C>, U1>>,
        u: &mut Option<OMatrix<T, R, DimMinimum<R, C>>>,
        v_t: &mut Option<OMatrix<T, DimMinimum<R, C>, C>>,
        is_upper_diagonal: bool,
        i: usize,
    ) {
        let mut v = Vector2::new(diagonal[i], off_diagonal[i]);
        off_diagonal[i] = T::RealField::zero();

        for k in (0..i + 1).rev() {
            if let Some((rot, norm)) = GivensRotation::cancel_y(&v) {
                let rot = GivensRotation::new_unchecked(rot.c(), T::from_real(rot.s()));
                diagonal[k] = norm;

                if is_upper_diagonal {
                    if let Some(ref mut v_t) = *v_t {
                        rot.rotate(&mut v_t.fixed_rows_with_step_mut::<2>(k, i - k));
                    }
                } else if let Some(ref mut u) = *u {
                    rot.inverse()
                        .rotate_rows(&mut u.fixed_columns_with_step_mut::<2>(k, i - k));
                }

                if k > 0 {
                    v.x = diagonal[k - 1];
                    v.y = rot.s().real() * off_diagonal[k - 1];
                    off_diagonal[k - 1] *= rot.c();
                }
            } else {
                break;
            }
        }
    }

    /// Computes the rank of the decomposed matrix, i.e., the number of singular values greater
    /// than `eps`.
    #[must_use]
    pub fn rank(&self, eps: T::RealField) -> usize {
        assert!(
            eps >= T::RealField::zero(),
            "SVD rank: the epsilon must be non-negative."
        );
        self.singular_values.iter().filter(|e| **e > eps).count()
    }

    /// Rebuild the original matrix.
    ///
    /// This is useful if some of the singular values have been manually modified.
    /// Returns `Err` if the right- and left- singular vectors have not been
    /// computed at construction-time.
    pub fn recompose(self) -> Result<OMatrix<T, R, C>, &'static str> {
        match (self.u, self.v_t) {
            (Some(mut u), Some(v_t)) => {
                for i in 0..self.singular_values.len() {
                    let val = self.singular_values[i];
                    u.column_mut(i).scale_mut(val);
                }
                Ok(u * v_t)
            }
            (None, None) => Err("SVD recomposition: U and V^t have not been computed."),
            (None, _) => Err("SVD recomposition: U has not been computed."),
            (_, None) => Err("SVD recomposition: V^t has not been computed."),
        }
    }

    /// Computes the pseudo-inverse of the decomposed matrix.
    ///
    /// Any singular value smaller than `eps` is assumed to be zero.
    /// Returns `Err` if the right- and left- singular vectors have not
    /// been computed at construction-time.
    pub fn pseudo_inverse(mut self, eps: T::RealField) -> Result<OMatrix<T, C, R>, &'static str>
    where
        DefaultAllocator: Allocator<T, C, R>,
    {
        if eps < T::RealField::zero() {
            Err("SVD pseudo inverse: the epsilon must be non-negative.")
        } else {
            for i in 0..self.singular_values.len() {
                let val = self.singular_values[i];

                if val > eps {
                    self.singular_values[i] = T::RealField::one() / val;
                } else {
                    self.singular_values[i] = T::RealField::zero();
                }
            }

            self.recompose().map(|m| m.adjoint())
        }
    }

    /// Solves the system `self * x = b` where `self` is the decomposed matrix and `x` the unknown.
    ///
    /// Any singular value smaller than `eps` is assumed to be zero.
    /// Returns `Err` if the singular vectors `U` and `V` have not been computed.
    // TODO: make this more generic wrt the storage types and the dimensions for `b`.
    pub fn solve<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<T, R2, C2, S2>,
        eps: T::RealField,
    ) -> Result<OMatrix<T, C, C2>, &'static str>
    where
        S2: Storage<T, R2, C2>,
        DefaultAllocator: Allocator<T, C, C2> + Allocator<T, DimMinimum<R, C>, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2>,
    {
        if eps < T::RealField::zero() {
            Err("SVD solve: the epsilon must be non-negative.")
        } else {
            match (&self.u, &self.v_t) {
                (Some(u), Some(v_t)) => {
                    let mut ut_b = u.ad_mul(b);

                    for j in 0..ut_b.ncols() {
                        let mut col = ut_b.column_mut(j);

                        for i in 0..self.singular_values.len() {
                            let val = self.singular_values[i];
                            if val > eps {
                                col[i] = col[i].unscale(val);
                            } else {
                                col[i] = T::zero();
                            }
                        }
                    }

                    Ok(v_t.ad_mul(&ut_b))
                }
                (None, None) => Err("SVD solve: U and V^t have not been computed."),
                (None, _) => Err("SVD solve: U has not been computed."),
                (_, None) => Err("SVD solve: V^t has not been computed."),
            }
        }
    }
}

impl<T: ComplexField, R: DimMin<C>, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S>
where
    DimMinimum<R, C>: DimSub<U1>, // for Bidiagonal.
    DefaultAllocator: Allocator<T, R, C>
        + Allocator<T, C>
        + Allocator<T, R>
        + Allocator<T, DimDiff<DimMinimum<R, C>, U1>>
        + Allocator<T, DimMinimum<R, C>, C>
        + Allocator<T, R, DimMinimum<R, C>>
        + Allocator<T, DimMinimum<R, C>>
        + Allocator<T::RealField, DimMinimum<R, C>>
        + Allocator<T::RealField, DimDiff<DimMinimum<R, C>, U1>>,
{
    /// Computes the singular values of this matrix.
    #[must_use]
    pub fn singular_values(&self) -> OVector<T::RealField, DimMinimum<R, C>> {
        SVD::new(self.clone_owned(), false, false).singular_values
    }

    /// Computes the rank of this matrix.
    ///
    /// All singular values below `eps` are considered equal to 0.
    #[must_use]
    pub fn rank(&self, eps: T::RealField) -> usize {
        let svd = SVD::new(self.clone_owned(), false, false);
        svd.rank(eps)
    }

    /// Computes the pseudo-inverse of this matrix.
    ///
    /// All singular values below `eps` are considered equal to 0.
    pub fn pseudo_inverse(self, eps: T::RealField) -> Result<OMatrix<T, C, R>, &'static str>
    where
        DefaultAllocator: Allocator<T, C, R>,
    {
        SVD::new(self.clone_owned(), true, true).pseudo_inverse(eps)
    }
}

// Explicit formulae inspired from the paper "Computing the Singular Values of 2-by-2 Complex
// Matrices", Sanzheng Qiao and Xiaohong Wang.
// http://www.cas.mcmaster.ca/sqrl/papers/sqrl5.pdf
fn compute_2x2_uptrig_svd<T: RealField>(
    m11: T,
    m12: T,
    m22: T,
    compute_u: bool,
    compute_v: bool,
) -> (
    Option<GivensRotation<T>>,
    Vector2<T>,
    Option<GivensRotation<T>>,
) {
    let two: T::RealField = crate::convert(2.0f64);
    let half: T::RealField = crate::convert(0.5f64);

    let denom = (m11 + m22).hypot(m12) + (m11 - m22).hypot(m12);

    // NOTE: v1 is the singular value that is the closest to m22.
    // This prevents cancellation issues when constructing the vector `csv` below. If we chose
    // otherwise, we would have v1 ~= m11 when m12 is small. This would cause catastrophic
    // cancellation on `v1 * v1 - m11 * m11` below.
    let mut v1 = m11 * m22 * two / denom;
    let mut v2 = half * denom;

    let mut u = None;
    let mut v_t = None;

    if compute_u || compute_v {
        let (csv, sgn_v) = GivensRotation::new(m11 * m12, v1 * v1 - m11 * m11);
        v1 *= sgn_v;
        v2 *= sgn_v;

        if compute_v {
            v_t = Some(csv);
        }

        if compute_u {
            let cu = (m11.scale(csv.c()) + m12 * csv.s()) / v1;
            let su = (m22 * csv.s()) / v1;
            let (csu, sgn_u) = GivensRotation::new(cu, su);

            v1 *= sgn_u;
            v2 *= sgn_u;
            u = Some(csu);
        }
    }

    (u, Vector2::new(v1, v2), v_t)
}
