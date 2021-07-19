use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use std::fmt;
use std::hash;
#[cfg(feature = "abomonation-serialize")]
use std::io::{Result as IOResult, Write};

#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

use simba::scalar::{RealField, SubsetOf};
use simba::simd::SimdRealField;

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimNameAdd, DimNameSum, U1};
use crate::base::storage::InnerOwned;
use crate::base::{Const, DefaultAllocator, OMatrix, SVector, Scalar, Unit};
use crate::geometry::{AbstractRotation, Point, Translation};

/// A direct isometry, i.e., a rotation followed by a translation (aka. a rigid-body motion).
///
/// This is also known as an element of a Special Euclidean (SE) group.
/// The `Isometry` type can either represent a 2D or 3D isometry.
/// A 2D isometry is composed of:
/// - A translation part of type [`Translation2`](crate::Translation2)
/// - A rotation part which can either be a [`UnitComplex`](crate::UnitComplex) or a [`Rotation2`](crate::Rotation2).
/// A 3D isometry is composed of:
/// - A translation part of type [`Translation3`](crate::Translation3)
/// - A rotation part which can either be a [`UnitQuaternion`](crate::UnitQuaternion) or a [`Rotation3`](crate::Rotation3).
///
/// Note that instead of using the [`Isometry`](crate::Isometry) type in your code directly, you should use one
/// of its aliases: [`Isometry2`](crate::Isometry2), [`Isometry3`](crate::Isometry3),
/// [`IsometryMatrix2`](crate::IsometryMatrix2), [`IsometryMatrix3`](crate::IsometryMatrix3). Though
/// keep in mind that all the documentation of all the methods of these aliases will also appears on
/// this page.
///
/// # Construction
/// * [From a 2D vector and/or an angle <span style="float:right;">`new`, `translation`, `rotation`…</span>](#construction-from-a-2d-vector-andor-a-rotation-angle)
/// * [From a 3D vector and/or an axis-angle <span style="float:right;">`new`, `translation`, `rotation`…</span>](#construction-from-a-3d-vector-andor-an-axis-angle)
/// * [From a 3D eye position and target point <span style="float:right;">`look_at`, `look_at_lh`, `face_towards`…</span>](#construction-from-a-3d-eye-position-and-target-point)
/// * [From the translation and rotation parts <span style="float:right;">`from_parts`…</span>](#from-the-translation-and-rotation-parts)
///
/// # Transformation and composition
/// Note that transforming vectors and points can be done by multiplication, e.g., `isometry * point`.
/// Composing an isometry with another transformation can also be done by multiplication or division.
///
/// * [Transformation of a vector or a point <span style="float:right;">`transform_vector`, `inverse_transform_point`…</span>](#transformation-of-a-vector-or-a-point)
/// * [Inversion and in-place composition <span style="float:right;">`inverse`, `append_rotation_wrt_point_mut`…</span>](#inversion-and-in-place-composition)
/// * [Interpolation <span style="float:right;">`lerp_slerp`…</span>](#interpolation)
///
/// # Conversion to a matrix
/// * [Conversion to a matrix <span style="float:right;">`to_matrix`…</span>](#conversion-to-a-matrix)
///
#[derive(Debug)]
#[cfg_attr(feature = "serde-serialize-no-std", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(serialize = "R: Serialize,
                     DefaultAllocator: Allocator<T, Const<D>>,
                     Owned<T, Const<D>>: Serialize,
                     T: Scalar"))
)]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(deserialize = "R: Deserialize<'de>,
                       DefaultAllocator: Allocator<T, Const<D>>,
                       Owned<T, Const<D>>: Deserialize<'de>,
                       T: Scalar"))
)]
pub struct Isometry<T, R, const D: usize> {
    /// The pure rotational part of this isometry.
    pub rotation: R,
    /// The pure translational part of this isometry.
    pub translation: Translation<T, D>,
}

#[cfg(feature = "abomonation-serialize")]
impl<T, R, const D: usize> Abomonation for Isometry<T, R, D>
where
    R: Abomonation,
    Translation<T, D>: Abomonation,
{
    unsafe fn entomb<W: Write>(&self, writer: &mut W) -> IOResult<()> {
        self.rotation.entomb(writer)?;
        self.translation.entomb(writer)
    }

    fn extent(&self) -> usize {
        self.rotation.extent() + self.translation.extent()
    }

    unsafe fn exhume<'a, 'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
        self.rotation
            .exhume(bytes)
            .and_then(|bytes| self.translation.exhume(bytes))
    }
}

#[cfg(feature = "rkyv-serialize-no-std")]
mod rkyv_impl {
    use super::Isometry;
    use crate::{base::Scalar, geometry::Translation};
    use rkyv::{offset_of, project_struct, Archive, Deserialize, Fallible, Serialize};

    impl<T: Archive, R: Archive, const D: usize> Archive for Isometry<T, R, D> {
        type Archived = Isometry<T::Archived, R::Archived, D>;
        type Resolver = (R::Resolver, <Translation<T, D> as Archive>::Resolver);

        fn resolve(
            &self,
            pos: usize,
            resolver: Self::Resolver,
            out: &mut core::mem::MaybeUninit<Self::Archived>,
        ) {
            self.rotation.resolve(
                pos + offset_of!(Self::Archived, rotation),
                resolver.0,
                project_struct!(out: Self::Archived => rotation),
            );
            self.translation.resolve(
                pos + offset_of!(Self::Archived, translation),
                resolver.1,
                project_struct!(out: Self::Archived => translation),
            );
        }
    }

    impl<T: Serialize<S>, R: Serialize<S>, S: Fallible + ?Sized, const D: usize> Serialize<S>
        for Isometry<T, R, D>
    where
        T::Archived: Scalar,
    {
        fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
            Ok((
                self.rotation.serialize(serializer)?,
                self.translation.serialize(serializer)?,
            ))
        }
    }

    impl<T: Archive, R: Archive, _D: Fallible + ?Sized, const D: usize>
        Deserialize<Isometry<T, R, D>, _D> for Isometry<T::Archived, R::Archived, D>
    where
        T::Archived: Scalar + Deserialize<T, _D>,
        R::Archived: Scalar + Deserialize<R, _D>,
    {
        fn deserialize(&self, deserializer: &mut _D) -> Result<Isometry<T, R, D>, _D::Error> {
            Ok(Isometry {
                rotation: self.rotation.deserialize(deserializer)?,
                translation: self.translation.deserialize(deserializer)?,
            })
        }
    }
}

impl<T: hash::Hash, R: hash::Hash, const D: usize> hash::Hash for Isometry<T, R, D>
where
    InnerOwned<T, Const<D>>: hash::Hash,
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.translation.hash(state);
        self.rotation.hash(state);
    }
}

impl<T: Copy, R: Copy, const D: usize> Copy for Isometry<T, R, D> where InnerOwned<T, Const<D>>: Copy {}

impl<T: Clone, R: Clone, const D: usize> Clone for Isometry<T, R, D> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            rotation: self.rotation.clone(),
            translation: self.translation.clone(),
        }
    }
}
/// # From the translation and rotation parts
impl<T: Scalar, R: AbstractRotation<T, D>, const D: usize> Isometry<T, R, D> {
    /// Creates a new isometry from its rotational and translational parts.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3, Point3};
    /// let tra = Translation3::new(0.0, 0.0, 3.0);
    /// let rot = UnitQuaternion::from_scaled_axis(Vector3::y() * f32::consts::PI);
    /// let iso = Isometry3::from_parts(tra, rot);
    ///
    /// assert_relative_eq!(iso * Point3::new(1.0, 2.0, 3.0), Point3::new(-1.0, 2.0, 0.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn from_parts(translation: Translation<T, D>, rotation: R) -> Self {
        Self {
            rotation,
            translation,
        }
    }
}

/// # Inversion and in-place composition
impl<T: SimdRealField, R: AbstractRotation<T, D>, const D: usize> Isometry<T, R, D>
where
    T::Element: SimdRealField,
{
    /// Inverts `self`.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Point2, Vector2};
    /// let iso = Isometry2::new(Vector2::new(1.0, 2.0), f32::consts::FRAC_PI_2);
    /// let inv = iso.inverse();
    /// let pt = Point2::new(1.0, 2.0);
    ///
    /// assert_eq!(inv * (iso * pt), pt);
    /// ```
    #[inline]
    #[must_use = "Did you mean to use inverse_mut()?"]
    pub fn inverse(&self) -> Self {
        let mut res = self.clone();
        res.inverse_mut();
        res
    }

    /// Inverts `self` in-place.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Point2, Vector2};
    /// let mut iso = Isometry2::new(Vector2::new(1.0, 2.0), f32::consts::FRAC_PI_2);
    /// let pt = Point2::new(1.0, 2.0);
    /// let transformed_pt = iso * pt;
    /// iso.inverse_mut();
    ///
    /// assert_eq!(iso * transformed_pt, pt);
    /// ```
    #[inline]
    pub fn inverse_mut(&mut self) {
        self.rotation.inverse_mut();
        self.translation.inverse_mut();
        self.translation.vector = self.rotation.transform_vector(&self.translation.vector);
    }

    /// Computes `self.inverse() * rhs` in a more efficient way.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Point2, Vector2};
    /// let mut iso1 = Isometry2::new(Vector2::new(1.0, 2.0), f32::consts::FRAC_PI_2);
    /// let mut iso2 = Isometry2::new(Vector2::new(10.0, 20.0), f32::consts::FRAC_PI_4);
    ///
    /// assert_eq!(iso1.inverse() * iso2, iso1.inv_mul(&iso2));
    /// ```
    #[inline]
    #[must_use]
    pub fn inv_mul(&self, rhs: &Isometry<T, R, D>) -> Self {
        let inv_rot1 = self.rotation.inverse();
        let tr_12 = rhs.translation.vector - self.translation.vector;
        Isometry::from_parts(
            inv_rot1.transform_vector(&tr_12).into(),
            inv_rot1 * rhs.rotation.clone(),
        )
    }

    /// Appends to `self` the given translation in-place.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Translation2, Vector2};
    /// let mut iso = Isometry2::new(Vector2::new(1.0, 2.0), f32::consts::FRAC_PI_2);
    /// let tra = Translation2::new(3.0, 4.0);
    /// // Same as `iso = tra * iso`.
    /// iso.append_translation_mut(&tra);
    ///
    /// assert_eq!(iso.translation, Translation2::new(4.0, 6.0));
    /// ```
    #[inline]
    pub fn append_translation_mut(&mut self, t: &Translation<T, D>) {
        self.translation.vector += &t.vector
    }

    /// Appends to `self` the given rotation in-place.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Translation2, UnitComplex, Vector2};
    /// let mut iso = Isometry2::new(Vector2::new(1.0, 2.0), f32::consts::PI / 6.0);
    /// let rot = UnitComplex::new(f32::consts::PI / 2.0);
    /// // Same as `iso = rot * iso`.
    /// iso.append_rotation_mut(&rot);
    ///
    /// assert_relative_eq!(iso, Isometry2::new(Vector2::new(-2.0, 1.0), f32::consts::PI * 2.0 / 3.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn append_rotation_mut(&mut self, r: &R) {
        self.rotation = r.clone() * self.rotation.clone();
        self.translation.vector = r.transform_vector(&self.translation.vector);
    }

    /// Appends in-place to `self` a rotation centered at the point `p`, i.e., the rotation that
    /// lets `p` invariant.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Translation2, UnitComplex, Vector2, Point2};
    /// let mut iso = Isometry2::new(Vector2::new(1.0, 2.0), f32::consts::FRAC_PI_2);
    /// let rot = UnitComplex::new(f32::consts::FRAC_PI_2);
    /// let pt = Point2::new(1.0, 0.0);
    /// iso.append_rotation_wrt_point_mut(&rot, &pt);
    ///
    /// assert_relative_eq!(iso * pt, Point2::new(-2.0, 0.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    pub fn append_rotation_wrt_point_mut(&mut self, r: &R, p: &Point<T, D>) {
        self.translation.vector -= &p.coords;
        self.append_rotation_mut(r);
        self.translation.vector += &p.coords;
    }

    /// Appends in-place to `self` a rotation centered at the point with coordinates
    /// `self.translation`.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Translation2, UnitComplex, Vector2, Point2};
    /// let mut iso = Isometry2::new(Vector2::new(1.0, 2.0), f32::consts::FRAC_PI_2);
    /// let rot = UnitComplex::new(f32::consts::FRAC_PI_2);
    /// iso.append_rotation_wrt_center_mut(&rot);
    ///
    /// // The translation part should not have changed.
    /// assert_eq!(iso.translation.vector, Vector2::new(1.0, 2.0));
    /// assert_eq!(iso.rotation, UnitComplex::new(f32::consts::PI));
    /// ```
    #[inline]
    pub fn append_rotation_wrt_center_mut(&mut self, r: &R) {
        self.rotation = r.clone() * self.rotation.clone();
    }
}

/// # Transformation of a vector or a point
impl<T: SimdRealField, R: AbstractRotation<T, D>, const D: usize> Isometry<T, R, D>
where
    T::Element: SimdRealField,
{
    /// Transform the given point by this isometry.
    ///
    /// This is the same as the multiplication `self * pt`.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3, Point3};
    /// let tra = Translation3::new(0.0, 0.0, 3.0);
    /// let rot = UnitQuaternion::from_scaled_axis(Vector3::y() * f32::consts::FRAC_PI_2);
    /// let iso = Isometry3::from_parts(tra, rot);
    ///
    /// let transformed_point = iso.transform_point(&Point3::new(1.0, 2.0, 3.0));
    /// assert_relative_eq!(transformed_point, Point3::new(3.0, 2.0, 2.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    #[must_use]
    pub fn transform_point(&self, pt: &Point<T, D>) -> Point<T, D> {
        self * pt
    }

    /// Transform the given vector by this isometry, ignoring the translation
    /// component of the isometry.
    ///
    /// This is the same as the multiplication `self * v`.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
    /// let tra = Translation3::new(0.0, 0.0, 3.0);
    /// let rot = UnitQuaternion::from_scaled_axis(Vector3::y() * f32::consts::FRAC_PI_2);
    /// let iso = Isometry3::from_parts(tra, rot);
    ///
    /// let transformed_point = iso.transform_vector(&Vector3::new(1.0, 2.0, 3.0));
    /// assert_relative_eq!(transformed_point, Vector3::new(3.0, 2.0, -1.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    #[must_use]
    pub fn transform_vector(&self, v: &SVector<T, D>) -> SVector<T, D> {
        self * v
    }

    /// Transform the given point by the inverse of this isometry. This may be
    /// less expensive than computing the entire isometry inverse and then
    /// transforming the point.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3, Point3};
    /// let tra = Translation3::new(0.0, 0.0, 3.0);
    /// let rot = UnitQuaternion::from_scaled_axis(Vector3::y() * f32::consts::FRAC_PI_2);
    /// let iso = Isometry3::from_parts(tra, rot);
    ///
    /// let transformed_point = iso.inverse_transform_point(&Point3::new(1.0, 2.0, 3.0));
    /// assert_relative_eq!(transformed_point, Point3::new(0.0, 2.0, 1.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    #[must_use]
    pub fn inverse_transform_point(&self, pt: &Point<T, D>) -> Point<T, D> {
        self.rotation
            .inverse_transform_point(&(pt - self.translation.vector))
    }

    /// Transform the given vector by the inverse of this isometry, ignoring the
    /// translation component of the isometry. This may be
    /// less expensive than computing the entire isometry inverse and then
    /// transforming the point.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
    /// let tra = Translation3::new(0.0, 0.0, 3.0);
    /// let rot = UnitQuaternion::from_scaled_axis(Vector3::y() * f32::consts::FRAC_PI_2);
    /// let iso = Isometry3::from_parts(tra, rot);
    ///
    /// let transformed_point = iso.inverse_transform_vector(&Vector3::new(1.0, 2.0, 3.0));
    /// assert_relative_eq!(transformed_point, Vector3::new(-3.0, 2.0, 1.0), epsilon = 1.0e-6);
    /// ```
    #[inline]
    #[must_use]
    pub fn inverse_transform_vector(&self, v: &SVector<T, D>) -> SVector<T, D> {
        self.rotation.inverse_transform_vector(v)
    }

    /// Transform the given unit vector by the inverse of this isometry, ignoring the
    /// translation component of the isometry. This may be
    /// less expensive than computing the entire isometry inverse and then
    /// transforming the point.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
    /// let tra = Translation3::new(0.0, 0.0, 3.0);
    /// let rot = UnitQuaternion::from_scaled_axis(Vector3::z() * f32::consts::FRAC_PI_2);
    /// let iso = Isometry3::from_parts(tra, rot);
    ///
    /// let transformed_point = iso.inverse_transform_unit_vector(&Vector3::x_axis());
    /// assert_relative_eq!(transformed_point, -Vector3::y_axis(), epsilon = 1.0e-6);
    /// ```
    #[inline]
    #[must_use]
    pub fn inverse_transform_unit_vector(&self, v: &Unit<SVector<T, D>>) -> Unit<SVector<T, D>> {
        self.rotation.inverse_transform_unit_vector(v)
    }
}

// NOTE: we don't require `R: Rotation<...>` here because this is not useful for the implementation
// and makes it hard to use it, e.g., for Transform × Isometry implementation.
// This is OK since all constructors of the isometry enforce the Rotation bound already (and
// explicit struct construction is prevented by the dummy ZST field).
/// # Conversion to a matrix
impl<T: SimdRealField, R, const D: usize> Isometry<T, R, D> {
    /// Converts this isometry into its equivalent homogeneous transformation matrix.
    ///
    /// This is the same as `self.to_matrix()`.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Vector2, Matrix3};
    /// let iso = Isometry2::new(Vector2::new(10.0, 20.0), f32::consts::FRAC_PI_6);
    /// let expected = Matrix3::new(0.8660254, -0.5,      10.0,
    ///                             0.5,       0.8660254, 20.0,
    ///                             0.0,       0.0,       1.0);
    ///
    /// assert_relative_eq!(iso.to_homogeneous(), expected, epsilon = 1.0e-6);
    /// ```
    #[inline]
    #[must_use]
    pub fn to_homogeneous(&self) -> OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
    where
        Const<D>: DimNameAdd<U1>,
        R: SubsetOf<OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>>,
        DefaultAllocator: Allocator<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
    {
        let mut res: OMatrix<T, _, _> = crate::convert_ref(&self.rotation);
        res.fixed_slice_mut::<D, 1>(0, D)
            .copy_from(&self.translation.vector);

        res
    }

    /// Converts this isometry into its equivalent homogeneous transformation matrix.
    ///
    /// This is the same as `self.to_homogeneous()`.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use std::f32;
    /// # use nalgebra::{Isometry2, Vector2, Matrix3};
    /// let iso = Isometry2::new(Vector2::new(10.0, 20.0), f32::consts::FRAC_PI_6);
    /// let expected = Matrix3::new(0.8660254, -0.5,      10.0,
    ///                             0.5,       0.8660254, 20.0,
    ///                             0.0,       0.0,       1.0);
    ///
    /// assert_relative_eq!(iso.to_matrix(), expected, epsilon = 1.0e-6);
    /// ```
    #[inline]
    #[must_use]
    pub fn to_matrix(&self) -> OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
    where
        Const<D>: DimNameAdd<U1>,
        R: SubsetOf<OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>>,
        DefaultAllocator: Allocator<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
    {
        self.to_homogeneous()
    }
}

impl<T: SimdRealField, R, const D: usize> Eq for Isometry<T, R, D> where
    R: AbstractRotation<T, D> + Eq
{
}

impl<T: SimdRealField, R, const D: usize> PartialEq for Isometry<T, R, D>
where
    R: AbstractRotation<T, D> + PartialEq,
{
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.translation == right.translation && self.rotation == right.rotation
    }
}

impl<T: RealField, R, const D: usize> AbsDiffEq for Isometry<T, R, D>
where
    R: AbstractRotation<T, D> + AbsDiffEq<Epsilon = T::Epsilon>,
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.translation.abs_diff_eq(&other.translation, epsilon)
            && self.rotation.abs_diff_eq(&other.rotation, epsilon)
    }
}

impl<T: RealField, R, const D: usize> RelativeEq for Isometry<T, R, D>
where
    R: AbstractRotation<T, D> + RelativeEq<Epsilon = T::Epsilon>,
    T::Epsilon: Copy,
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.translation
            .relative_eq(&other.translation, epsilon, max_relative)
            && self
                .rotation
                .relative_eq(&other.rotation, epsilon, max_relative)
    }
}

impl<T: RealField, R, const D: usize> UlpsEq for Isometry<T, R, D>
where
    R: AbstractRotation<T, D> + UlpsEq<Epsilon = T::Epsilon>,
    T::Epsilon: Copy,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.translation
            .ulps_eq(&other.translation, epsilon, max_ulps)
            && self.rotation.ulps_eq(&other.rotation, epsilon, max_ulps)
    }
}

/*
 *
 * Display
 *
 */
impl<T: Scalar + fmt::Display, R, const D: usize> fmt::Display for Isometry<T, R, D>
where
    R: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);

        writeln!(f, "Isometry {{")?;
        write!(f, "{:.*}", precision, self.translation)?;
        write!(f, "{:.*}", precision, self.rotation)?;
        writeln!(f, "}}")
    }
}
