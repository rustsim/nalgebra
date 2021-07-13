#![allow(clippy::type_complexity)]
/*!
# nalgebra

**nalgebra** is a linear algebra library written for Rust targeting:

* General-purpose linear algebra (still lacks a lot of features…)
* Real-time computer graphics.
* Real-time computer physics.

## Using **nalgebra**
You will need the last stable build of the [rust compiler](https://www.rust-lang.org)
and the official package manager: [cargo](https://github.com/rust-lang/cargo).

Simply add the following to your `Cargo.toml` file:

```ignore
[dependencies]
// TODO: replace the * by the latest version.
nalgebra = "*"
```


Most useful functionalities of **nalgebra** are grouped in the root module `nalgebra::`.

However, the recommended way to use **nalgebra** is to import types and traits
explicitly, and call free-functions using the `na::` prefix:

```
#[macro_use]
extern crate approx; // For the macro relative_eq!
extern crate nalgebra as na;
use na::{Vector3, Rotation3};

fn main() {
    let axis  = Vector3::new(1.0, 0.0, 0.0);
    let angle = 1.57;
    let b     = Rotation3::from_axis_angle(&axis, angle);

    relative_eq!(b.axis().unwrap(), axis);
    relative_eq!(b.angle(), angle);
}
```


## Features
**nalgebra** is meant to be a general-purpose, low-dimensional, linear algebra library, with
an optimized set of tools for computer graphics and physics. Those features include:

* A single parametrizable type `Matrix` for vectors, (square or rectangular) matrices, and slices
  with dimensions known either at compile-time (using type-level integers) or at runtime.
* Matrices and vectors with compile-time sizes are statically allocated while dynamic ones are
  allocated on the heap.
* Convenient aliases for low-dimensional matrices and vectors: `Vector1` to `Vector6` and
  `Matrix1x1` to `Matrix6x6`, including rectangular matrices like `Matrix2x5`.
* Points sizes known at compile time, and convenience aliases: `Point1` to `Point6`.
* Translation (seen as a transformation that composes by multiplication): `Translation2`,
  `Translation3`.
* Rotation matrices: `Rotation2`, `Rotation3`.
* Quaternions: `Quaternion`, `UnitQuaternion` (for 3D rotation).
* Unit complex numbers can be used for 2D rotation: `UnitComplex`.
* Algebraic entities with a norm equal to one: `Unit<T>`, e.g., `Unit<Vector3<f32>>`.
* Isometries (translation ⨯ rotation): `Isometry2`, `Isometry3`
* Similarity transformations (translation ⨯ rotation ⨯ uniform scale): `Similarity2`, `Similarity3`.
* Affine transformations stored as a homogeneous matrix: `Affine2`, `Affine3`.
* Projective (i.e. invertible) transformations stored as a homogeneous matrix: `Projective2`,
  `Projective3`.
* General transformations that does not have to be invertible, stored as a homogeneous matrix:
  `Transform2`, `Transform3`.
* 3D projections for computer graphics: `Perspective3`, `Orthographic3`.
* Matrix factorizations: `Cholesky`, `QR`, `LU`, `FullPivLU`, `SVD`, `Schur`, `Hessenberg`, `SymmetricEigen`.
* Insertion and removal of rows of columns of a matrix.
*/

// #![feature(plugin)]
//
// #![plugin(clippy)]

#![deny(non_camel_case_types)]
#![deny(unused_parens)]
#![deny(non_upper_case_globals)]
#![deny(unused_qualifications)]
#![deny(unused_results)]
#![deny(missing_docs)]
#![doc(
    html_favicon_url = "https://nalgebra.org/img/favicon.ico",
    html_root_url = "https://docs.rs/nalgebra/0.25.0"
)]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "no_unsound_assume_init", allow(unreachable_code))]

#[cfg(feature = "rand-no-std")]
extern crate rand_package as rand;

#[cfg(feature = "serde-serialize-no-std")]
#[macro_use]
extern crate serde;

#[macro_use]
extern crate approx;
extern crate num_traits as num;

#[cfg(all(feature = "alloc", not(feature = "std")))]
#[cfg_attr(test, macro_use)]
extern crate alloc;

#[cfg(not(feature = "std"))]
extern crate core as std;

#[cfg(feature = "io")]
extern crate pest;
#[macro_use]
#[cfg(feature = "io")]
extern crate pest_derive;

pub mod base;
#[cfg(feature = "debug")]
pub mod debug;
pub mod geometry;
#[cfg(feature = "io")]
pub mod io;
pub mod linalg;
#[cfg(feature = "proptest-support")]
pub mod proptest;
#[cfg(feature = "sparse")]
pub mod sparse;
mod third_party;

pub use crate::base::*;
pub use crate::geometry::*;
pub use crate::linalg::*;
#[cfg(feature = "sparse")]
pub use crate::sparse::*;
#[cfg(feature = "std")]
#[deprecated(
    note = "The 'core' module is being renamed to 'base' to avoid conflicts with the 'core' crate."
)]
pub use base as core;

#[cfg(feature = "macros")]
pub use nalgebra_macros::{dmatrix, dvector, matrix, point, vector};

use simba::scalar::SupersetOf;
use std::cmp::{self, Ordering, PartialOrd};

use num::{One, Signed, Zero};

use base::allocator::Allocator;
pub use num_complex::Complex;
pub use simba::scalar::{
    ClosedAdd, ClosedDiv, ClosedMul, ClosedSub, ComplexField, Field, RealField,
};
pub use simba::simd::{SimdBool, SimdComplexField, SimdPartialOrd, SimdRealField, SimdValue};

/// Gets the multiplicative identity element.
///
/// # See also:
///
/// * [`origin`](../nalgebra/fn.origin.html)
/// * [`zero`](fn.zero.html)
#[inline]
pub fn one<T: One>() -> T {
    T::one()
}

/// Gets the additive identity element.
///
/// # See also:
///
/// * [`one`](fn.one.html)
/// * [`origin`](../nalgebra/fn.origin.html)
#[inline]
pub fn zero<T: Zero>() -> T {
    T::zero()
}

/*
 *
 * Ordering
 *
 */
// XXX: this is very naive and could probably be optimized for specific types.
// XXX: also, we might just want to use divisions, but assuming `val` is usually not far from `min`
// or `max`, would it still be more efficient?
/// Wraps `val` into the range `[min, max]` using modular arithmetics.
///
/// The range must not be empty.
#[must_use]
#[inline]
pub fn wrap<T>(mut val: T, min: T, max: T) -> T
where
    T: Copy + PartialOrd + ClosedAdd + ClosedSub,
{
    assert!(min < max, "Invalid wrapping bounds.");
    let width = max - min;

    if val < min {
        val += width;

        while val < min {
            val += width
        }
    } else if val > max {
        val -= width;

        while val > max {
            val -= width
        }
    }

    val
}

/// Returns a reference to the input value clamped to the interval `[min, max]`.
///
/// In particular:
///     * If `min < val < max`, this returns `val`.
///     * If `val <= min`, this returns `min`.
///     * If `val >= max`, this returns `max`.
#[must_use]
#[inline]
pub fn clamp<T: PartialOrd>(val: T, min: T, max: T) -> T {
    if val > min {
        if val < max {
            val
        } else {
            max
        }
    } else {
        min
    }
}

/// Same as `cmp::max`.
#[inline]
pub fn max<T: Ord>(a: T, b: T) -> T {
    cmp::max(a, b)
}

/// Same as `cmp::min`.
#[inline]
pub fn min<T: Ord>(a: T, b: T) -> T {
    cmp::min(a, b)
}

/// The absolute value of `a`.
///
/// Deprecated: Use [Matrix::abs] or [RealField::abs] instead.
#[deprecated(note = "use the inherent method `Matrix::abs` or `RealField::abs` instead")]
#[inline]
pub fn abs<T: Signed>(a: &T) -> T {
    a.abs()
}

/// Returns the infimum of `a` and `b`.
#[deprecated(note = "use the inherent method `Matrix::inf` instead")]
#[inline]
pub fn inf<T, R: Dim, C: Dim>(a: &OMatrix<T, R, C>, b: &OMatrix<T, R, C>) -> OMatrix<T, R, C>
where
    T: Scalar + SimdPartialOrd,
    DefaultAllocator: Allocator<T, R, C>,
{
    a.inf(b)
}

/// Returns the supremum of `a` and `b`.
#[deprecated(note = "use the inherent method `Matrix::sup` instead")]
#[inline]
pub fn sup<T, R: Dim, C: Dim>(a: &OMatrix<T, R, C>, b: &OMatrix<T, R, C>) -> OMatrix<T, R, C>
where
    T: Scalar + SimdPartialOrd,
    DefaultAllocator: Allocator<T, R, C>,
{
    a.sup(b)
}

/// Returns simultaneously the infimum and supremum of `a` and `b`.
#[deprecated(note = "use the inherent method `Matrix::inf_sup` instead")]
#[inline]
pub fn inf_sup<T, R: Dim, C: Dim>(
    a: &OMatrix<T, R, C>,
    b: &OMatrix<T, R, C>,
) -> (OMatrix<T, R, C>, OMatrix<T, R, C>)
where
    T: Scalar + SimdPartialOrd,
    DefaultAllocator: Allocator<T, R, C>,
{
    a.inf_sup(b)
}

/// Compare `a` and `b` using a partial ordering relation.
#[inline]
pub fn partial_cmp<T: PartialOrd>(a: &T, b: &T) -> Option<Ordering> {
    a.partial_cmp(b)
}

/// Returns `true` iff `a` and `b` are comparable and `a < b`.
#[inline]
pub fn partial_lt<T: PartialOrd>(a: &T, b: &T) -> bool {
    a.lt(b)
}

/// Returns `true` iff `a` and `b` are comparable and `a <= b`.
#[inline]
pub fn partial_le<T: PartialOrd>(a: &T, b: &T) -> bool {
    a.le(b)
}

/// Returns `true` iff `a` and `b` are comparable and `a > b`.
#[inline]
pub fn partial_gt<T: PartialOrd>(a: &T, b: &T) -> bool {
    a.gt(b)
}

/// Returns `true` iff `a` and `b` are comparable and `a >= b`.
#[inline]
pub fn partial_ge<T: PartialOrd>(a: &T, b: &T) -> bool {
    a.ge(b)
}

/// Return the minimum of `a` and `b` if they are comparable.
#[inline]
pub fn partial_min<'a, T: PartialOrd>(a: &'a T, b: &'a T) -> Option<&'a T> {
    if let Some(ord) = a.partial_cmp(b) {
        match ord {
            Ordering::Greater => Some(b),
            _ => Some(a),
        }
    } else {
        None
    }
}

/// Return the maximum of `a` and `b` if they are comparable.
#[inline]
pub fn partial_max<'a, T: PartialOrd>(a: &'a T, b: &'a T) -> Option<&'a T> {
    if let Some(ord) = a.partial_cmp(b) {
        match ord {
            Ordering::Less => Some(b),
            _ => Some(a),
        }
    } else {
        None
    }
}

/// Clamp `value` between `min` and `max`. Returns `None` if `value` is not comparable to
/// `min` or `max`.
#[inline]
pub fn partial_clamp<'a, T: PartialOrd>(value: &'a T, min: &'a T, max: &'a T) -> Option<&'a T> {
    if let (Some(cmp_min), Some(cmp_max)) = (value.partial_cmp(min), value.partial_cmp(max)) {
        if cmp_min == Ordering::Less {
            Some(min)
        } else if cmp_max == Ordering::Greater {
            Some(max)
        } else {
            Some(value)
        }
    } else {
        None
    }
}

/// Sorts two values in increasing order using a partial ordering.
#[inline]
pub fn partial_sort2<'a, T: PartialOrd>(a: &'a T, b: &'a T) -> Option<(&'a T, &'a T)> {
    if let Some(ord) = a.partial_cmp(b) {
        match ord {
            Ordering::Less => Some((a, b)),
            _ => Some((b, a)),
        }
    } else {
        None
    }
}

/*
 *
 * Point operations.
 *
 */
/// The center of two points.
///
/// # See also:
///
/// * [distance](fn.distance.html)
/// * [distance_squared](fn.distance_squared.html)
#[inline]
pub fn center<T: SimdComplexField, const D: usize>(
    p1: &Point<T, D>,
    p2: &Point<T, D>,
) -> Point<T, D> {
    ((p1.coords + p2.coords) * convert::<_, T>(0.5)).into()
}

/// The distance between two points.
///
/// # See also:
///
/// * [center](fn.center.html)
/// * [distance_squared](fn.distance_squared.html)
#[inline]
pub fn distance<T: SimdComplexField, const D: usize>(
    p1: &Point<T, D>,
    p2: &Point<T, D>,
) -> T::SimdRealField {
    (p2.coords - p1.coords).norm()
}

/// The squared distance between two points.
///
/// # See also:
///
/// * [center](fn.center.html)
/// * [distance](fn.distance.html)
#[inline]
pub fn distance_squared<T: SimdComplexField, const D: usize>(
    p1: &Point<T, D>,
    p2: &Point<T, D>,
) -> T::SimdRealField {
    (p2.coords - p1.coords).norm_squared()
}

/*
 * Cast
 */
/// Converts an object from one type to an equivalent or more general one.
///
/// See also [`try_convert`](fn.try_convert.html) for conversion to more specific types.
///
/// # See also:
///
/// * [convert_ref](fn.convert_ref.html)
/// * [convert_ref_unchecked](fn.convert_ref_unchecked.html)
/// * [is_convertible](../nalgebra/fn.is_convertible.html)
/// * [try_convert](fn.try_convert.html)
/// * [try_convert_ref](fn.try_convert_ref.html)
#[inline]
pub fn convert<From, To: SupersetOf<From>>(t: From) -> To {
    To::from_subset(&t)
}

/// Attempts to convert an object to a more specific one.
///
/// See also [`convert`](fn.convert.html) for conversion to more general types.
///
/// # See also:
///
/// * [convert](fn.convert.html)
/// * [convert_ref](fn.convert_ref.html)
/// * [convert_ref_unchecked](fn.convert_ref_unchecked.html)
/// * [is_convertible](../nalgebra/fn.is_convertible.html)
/// * [try_convert_ref](fn.try_convert_ref.html)
#[inline]
pub fn try_convert<From: SupersetOf<To>, To>(t: From) -> Option<To> {
    t.to_subset()
}

/// Indicates if [`try_convert`](fn.try_convert.html) will succeed without
/// actually performing the conversion.
///
/// # See also:
///
/// * [convert](fn.convert.html)
/// * [convert_ref](fn.convert_ref.html)
/// * [convert_ref_unchecked](fn.convert_ref_unchecked.html)
/// * [try_convert](fn.try_convert.html)
/// * [try_convert_ref](fn.try_convert_ref.html)
#[inline]
pub fn is_convertible<From: SupersetOf<To>, To>(t: &From) -> bool {
    t.is_in_subset()
}

/// Use with care! Same as [`try_convert`](fn.try_convert.html) but
/// without any property checks.
///
/// # See also:
///
/// * [convert](fn.convert.html)
/// * [convert_ref](fn.convert_ref.html)
/// * [convert_ref_unchecked](fn.convert_ref_unchecked.html)
/// * [is_convertible](../nalgebra/fn.is_convertible.html)
/// * [try_convert](fn.try_convert.html)
/// * [try_convert_ref](fn.try_convert_ref.html)
#[inline]
pub fn convert_unchecked<From: SupersetOf<To>, To>(t: From) -> To {
    t.to_subset_unchecked()
}

/// Converts an object from one type to an equivalent or more general one.
///
/// # See also:
///
/// * [convert](fn.convert.html)
/// * [convert_ref_unchecked](fn.convert_ref_unchecked.html)
/// * [is_convertible](../nalgebra/fn.is_convertible.html)
/// * [try_convert](fn.try_convert.html)
/// * [try_convert_ref](fn.try_convert_ref.html)
#[inline]
pub fn convert_ref<From, To: SupersetOf<From>>(t: &From) -> To {
    To::from_subset(t)
}

/// Attempts to convert an object to a more specific one.
///
/// # See also:
///
/// * [convert](fn.convert.html)
/// * [convert_ref](fn.convert_ref.html)
/// * [convert_ref_unchecked](fn.convert_ref_unchecked.html)
/// * [is_convertible](../nalgebra/fn.is_convertible.html)
/// * [try_convert](fn.try_convert.html)
#[inline]
pub fn try_convert_ref<From: SupersetOf<To>, To>(t: &From) -> Option<To> {
    t.to_subset()
}

/// Use with care! Same as [`try_convert`](fn.try_convert.html) but
/// without any property checks.
///
/// # See also:
///
/// * [convert](fn.convert.html)
/// * [convert_ref](fn.convert_ref.html)
/// * [is_convertible](../nalgebra/fn.is_convertible.html)
/// * [try_convert](fn.try_convert.html)
/// * [try_convert_ref](fn.try_convert_ref.html)
#[inline]
pub fn convert_ref_unchecked<From: SupersetOf<To>, To>(t: &From) -> To {
    t.to_subset_unchecked()
}
