//! Construction of householder elementary reflections.

use std::mem::MaybeUninit;

use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, OMatrix, OVector, Unit, Vector};
use crate::dimension::Dim;
use crate::storage::{Storage, StorageMut};
use num::Zero;
use simba::scalar::ComplexField;

use crate::geometry::Reflection;

/// Replaces `column` by the axis of the householder reflection that transforms `column` into
/// `(+/-|column|, 0, ..., 0)`.
///
/// The unit-length axis is output to `column`. Returns what would be the first component of
/// `column` after reflection and `false` if no reflection was necessary.
#[doc(hidden)]
#[inline(always)]
pub fn reflection_axis_mut<T: ComplexField, D: Dim, S: StorageMut<T, D>>(
    column: &mut Vector<T, D, S>,
) -> (T, bool) {
    let reflection_sq_norm = column.norm_squared();
    let reflection_norm = reflection_sq_norm.sqrt();

    let factor;
    let signed_norm;

    unsafe {
        let (modulus, sign) = column.vget_unchecked(0).to_exp();
        signed_norm = sign.scale(reflection_norm);
        factor = (reflection_sq_norm + modulus * reflection_norm) * crate::convert(2.0);
        *column.vget_unchecked_mut(0) += signed_norm;
    };

    if !factor.is_zero() {
        column.unscale_mut(factor.sqrt());
        (-signed_norm, true)
    } else {
        // TODO: not sure why we don't have a - sign here.
        (signed_norm, false)
    }
}

/// Uses an householder reflection to zero out the `icol`-th column, starting with the `shift + 1`-th
/// subdiagonal element.
///
/// # Safety
/// Behavior is undefined if any of the following conditions are violated:
///
/// - `diag_elt` must be valid for writes.
/// - `diag_elt` must be properly aligned.
///
/// Furthermore, if `diag_elt` was previously initialized, this method will leak
/// its data.
#[doc(hidden)]
pub unsafe fn clear_column_unchecked<T: ComplexField, R: Dim, C: Dim>(
    matrix: &mut OMatrix<T, R, C>,
    diag_elt: *mut T,
    icol: usize,
    shift: usize,
    bilateral: Option<&mut OVector<MaybeUninit<T>, R>>,
) where
    DefaultAllocator: Allocator<T, R, C> + Allocator<T, R>,
{
    let (mut left, mut right) = matrix.columns_range_pair_mut(icol, icol + 1..);
    let mut axis = left.rows_range_mut(icol + shift..);

    let (reflection_norm, not_zero) = reflection_axis_mut(&mut axis);
    diag_elt.write(reflection_norm);

    if not_zero {
        let refl = Reflection::new(Unit::new_unchecked(axis), T::zero());
        let sign = reflection_norm.signum();
        if let Some(mut work) = bilateral {
            refl.reflect_rows_with_sign(&mut right, &mut work, sign);
        }
        refl.reflect_with_sign(&mut right.rows_range_mut(icol + shift..), sign.conjugate());
    }
}

/// Uses an householder reflection to zero out the `irow`-th row, ending before the `shift + 1`-th
/// superdiagonal element.
///
/// # Safety
/// Behavior is undefined if any of the following conditions are violated:
///
/// - `diag_elt` must be valid for writes.
/// - `diag_elt` must be properly aligned.
///
/// Furthermore, if `diag_elt` was previously initialized, this method will leak
/// its data.
#[doc(hidden)]
pub unsafe fn clear_row_unchecked<T: ComplexField, R: Dim, C: Dim>(
    matrix: &mut OMatrix<T, R, C>,
    diag_elt: *mut T,
    axis_packed: &mut OVector<MaybeUninit<T>, C>,
    work: &mut OVector<MaybeUninit<T>, R>,
    irow: usize,
    shift: usize,
) where
    DefaultAllocator: Allocator<T, R, C> + Allocator<T, R> + Allocator<T, C>,
{
    let (mut top, mut bottom) = matrix.rows_range_pair_mut(irow, irow + 1..);
    let mut axis = axis_packed.rows_range_mut(irow + shift..);
    axis.tr_copy_init_from(&top.columns_range(irow + shift..));
    let mut axis = axis.assume_init_mut();

    let (reflection_norm, not_zero) = reflection_axis_mut(&mut axis);
    axis.conjugate_mut(); // So that reflect_rows actually cancels the first row.
    diag_elt.write(reflection_norm);

    if not_zero {
        let refl = Reflection::new(Unit::new_unchecked(axis), T::zero());
        refl.reflect_rows_with_sign(
            &mut bottom.columns_range_mut(irow + shift..),
            &mut work.rows_range_mut(irow + 1..),
            reflection_norm.signum().conjugate(),
        );
        top.columns_range_mut(irow + shift..)
            .tr_copy_from(refl.axis());
    } else {
        top.columns_range_mut(irow + shift..).tr_copy_from(&axis);
    }
}

/// Computes the orthogonal transformation described by the elementary reflector axii stored on
/// the lower-diagonal element of the given matrix.
/// matrices.
#[doc(hidden)]
pub fn assemble_q<T: ComplexField, D: Dim>(m: &OMatrix<T, D, D>, signs: &[T]) -> OMatrix<T, D, D>
where
    DefaultAllocator: Allocator<T, D, D>,
{
    assert!(m.is_square());
    let dim = m.data.shape().0;

    // NOTE: we could build the identity matrix and call p_mult on it.
    // Instead we don't so that we take in account the matrix sparseness.
    let mut res = OMatrix::identity_generic(dim, dim);

    for i in (0..dim.value() - 1).rev() {
        let axis = m.slice_range(i + 1.., i);
        let refl = Reflection::new(Unit::new_unchecked(axis), T::zero());

        let mut res_rows = res.slice_range_mut(i + 1.., i..);
        refl.reflect_with_sign(&mut res_rows, signs[i].signum());
    }

    res
}
