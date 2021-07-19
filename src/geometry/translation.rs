use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num::{One, Zero};
use std::fmt;
use std::hash;
#[cfg(feature = "abomonation-serialize")]
use std::io::{Result as IOResult, Write};

#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

use simba::scalar::{ClosedAdd, ClosedNeg, ClosedSub};

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimNameAdd, DimNameSum, U1};
use crate::base::storage::InnerOwned;
use crate::base::{Const, DefaultAllocator, OMatrix, SVector, Scalar};

use crate::geometry::Point;

/// A translation.
#[repr(transparent)]
#[derive(Debug)]
pub struct Translation<T, const D: usize> {
    /// The translation coordinates, i.e., how much is added to a point's coordinates when it is
    /// translated.
    pub vector: SVector<T, D>,
}

impl<T: hash::Hash, const D: usize> hash::Hash for Translation<T, D>
where
    InnerOwned<T, Const<D>>: hash::Hash,
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.vector.hash(state)
    }
}

impl<T: Copy, const D: usize> Copy for Translation<T, D> {}

impl<T: Clone, const D: usize> Clone for Translation<T, D>
where
    InnerOwned<T, Const<D>>: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Translation::from(self.vector.clone())
    }
}

#[cfg(feature = "abomonation-serialize")]
impl<T, const D: usize> Abomonation for Translation<T, D>
where
    SVector<T, D>: Abomonation,
{
    unsafe fn entomb<W: Write>(&self, writer: &mut W) -> IOResult<()> {
        self.vector.entomb(writer)
    }

    fn extent(&self) -> usize {
        self.vector.extent()
    }

    unsafe fn exhume<'a, 'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
        self.vector.exhume(bytes)
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<T, const D: usize> Serialize for Translation<T, D>
where
    InnerOwned<T, Const<D>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.vector.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<'a, T, const D: usize> Deserialize<'a> for Translation<T, D>
where
    InnerOwned<T, Const<D>>: Deserialize<'a>,
{
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'a>,
    {
        let matrix = SVector::<T, D>::deserialize(deserializer)?;

        Ok(Translation::from(matrix))
    }
}

#[cfg(feature = "rkyv-serialize-no-std")]
mod rkyv_impl {
    use super::Translation;
    use crate::base::SVector;
    use rkyv::{offset_of, project_struct, Archive, Deserialize, Fallible, Serialize};

    impl<T: Archive, const D: usize> Archive for Translation<T, D> {
        type Archived = Translation<T::Archived, D>;
        type Resolver = <SVector<T, D> as Archive>::Resolver;

        fn resolve(
            &self,
            pos: usize,
            resolver: Self::Resolver,
            out: &mut core::mem::MaybeUninit<Self::Archived>,
        ) {
            self.vector.resolve(
                pos + offset_of!(Self::Archived, vector),
                resolver,
                project_struct!(out: Self::Archived => vector),
            );
        }
    }

    impl<T: Serialize<S>, S: Fallible + ?Sized, const D: usize> Serialize<S> for Translation<T, D> {
        fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
            self.vector.serialize(serializer)
        }
    }

    impl<T: Archive, _D: Fallible + ?Sized, const D: usize> Deserialize<Translation<T, D>, _D>
        for Translation<T::Archived, D>
    where
        T::Archived: Deserialize<T, _D>,
    {
        fn deserialize(&self, deserializer: &mut _D) -> Result<Translation<T, D>, _D::Error> {
            Ok(Translation {
                vector: self.vector.deserialize(deserializer)?,
            })
        }
    }
}

impl<T, const D: usize> Translation<T, D> {
    /// Creates a new translation from the given vector.
    #[inline]
    #[deprecated(note = "Use `::from` instead.")]
    pub fn from_vector(vector: SVector<T, D>) -> Translation<T, D> {
        Translation { vector }
    }

    /// Inverts `self`.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Translation2, Translation3};
    /// let t = Translation3::new(1.0, 2.0, 3.0);
    /// assert_eq!(t * t.inverse(), Translation3::identity());
    /// assert_eq!(t.inverse() * t, Translation3::identity());
    ///
    /// // Work in all dimensions.
    /// let t = Translation2::new(1.0, 2.0);
    /// assert_eq!(t * t.inverse(), Translation2::identity());
    /// assert_eq!(t.inverse() * t, Translation2::identity());
    /// ```
    #[inline]
    #[must_use = "Did you mean to use inverse_mut()?"]
    pub fn inverse(&self) -> Translation<T, D>
    where
        T: ClosedNeg + Scalar,
    {
        Translation::from(-&self.vector)
    }

    /// Converts this translation into its equivalent homogeneous transformation matrix.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Translation2, Translation3, Matrix3, Matrix4};
    /// let t = Translation3::new(10.0, 20.0, 30.0);
    /// let expected = Matrix4::new(1.0, 0.0, 0.0, 10.0,
    ///                             0.0, 1.0, 0.0, 20.0,
    ///                             0.0, 0.0, 1.0, 30.0,
    ///                             0.0, 0.0, 0.0, 1.0);
    /// assert_eq!(t.to_homogeneous(), expected);
    ///
    /// let t = Translation2::new(10.0, 20.0);
    /// let expected = Matrix3::new(1.0, 0.0, 10.0,
    ///                             0.0, 1.0, 20.0,
    ///                             0.0, 0.0, 1.0);
    /// assert_eq!(t.to_homogeneous(), expected);
    /// ```
    #[inline]
    #[must_use]
    pub fn to_homogeneous(&self) -> OMatrix<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>
    where
        T: Zero + One + Scalar,
        Const<D>: DimNameAdd<U1>,
        DefaultAllocator: Allocator<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
    {
        let mut res = OMatrix::<T, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>::identity();
        res.fixed_slice_mut::<D, 1>(0, D).copy_from(&self.vector);

        res
    }

    /// Inverts `self` in-place.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Translation2, Translation3};
    /// let t = Translation3::new(1.0, 2.0, 3.0);
    /// let mut inv_t = Translation3::new(1.0, 2.0, 3.0);
    /// inv_t.inverse_mut();
    /// assert_eq!(t * inv_t, Translation3::identity());
    /// assert_eq!(inv_t * t, Translation3::identity());
    ///
    /// // Work in all dimensions.
    /// let t = Translation2::new(1.0, 2.0);
    /// let mut inv_t = Translation2::new(1.0, 2.0);
    /// inv_t.inverse_mut();
    /// assert_eq!(t * inv_t, Translation2::identity());
    /// assert_eq!(inv_t * t, Translation2::identity());
    /// ```
    #[inline]
    pub fn inverse_mut(&mut self)
    where
        T: ClosedNeg + Scalar,
    {
        self.vector.neg_mut()
    }
}

impl<T: Scalar + ClosedAdd, const D: usize> Translation<T, D> {
    /// Translate the given point.
    ///
    /// This is the same as the multiplication `self * pt`.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Translation3, Point3};
    /// let t = Translation3::new(1.0, 2.0, 3.0);
    /// let transformed_point = t.transform_point(&Point3::new(4.0, 5.0, 6.0));
    /// assert_eq!(transformed_point, Point3::new(5.0, 7.0, 9.0));
    #[inline]
    #[must_use]
    pub fn transform_point(&self, pt: &Point<T, D>) -> Point<T, D> {
        pt + &self.vector
    }
}

impl<T: Scalar + ClosedSub, const D: usize> Translation<T, D> {
    /// Translate the given point by the inverse of this translation.
    ///
    /// # Example
    /// ```
    /// # use nalgebra::{Translation3, Point3};
    /// let t = Translation3::new(1.0, 2.0, 3.0);
    /// let transformed_point = t.inverse_transform_point(&Point3::new(4.0, 5.0, 6.0));
    /// assert_eq!(transformed_point, Point3::new(3.0, 3.0, 3.0));
    #[inline]
    #[must_use]
    pub fn inverse_transform_point(&self, pt: &Point<T, D>) -> Point<T, D> {
        pt - &self.vector
    }
}

impl<T: Eq, const D: usize> Eq for Translation<T, D> {}

impl<T: PartialEq, const D: usize> PartialEq for Translation<T, D> {
    #[inline]
    fn eq(&self, right: &Translation<T, D>) -> bool {
        self.vector == right.vector
    }
}

impl<T: AbsDiffEq, const D: usize> AbsDiffEq for Translation<T, D>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.vector.abs_diff_eq(&other.vector, epsilon)
    }
}

impl<T: RelativeEq, const D: usize> RelativeEq for Translation<T, D>
where
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
        self.vector
            .relative_eq(&other.vector, epsilon, max_relative)
    }
}

impl<T: UlpsEq, const D: usize> UlpsEq for Translation<T, D>
where
    T::Epsilon: Copy,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.vector.ulps_eq(&other.vector, epsilon, max_ulps)
    }
}

/*
 *
 * Display
 *
 */
impl<T: Scalar + fmt::Display, const D: usize> fmt::Display for Translation<T, D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);

        writeln!(f, "Translation {{")?;
        write!(f, "{:.*}", precision, self.vector)?;
        writeln!(f, "}}")
    }
}
