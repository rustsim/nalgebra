#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};
#[cfg(feature = "rand-no-std")]
use rand::{
    distributions::{Distribution, Standard},
    Rng,
};

#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

use simba::scalar::RealField;

use crate::base::dimension::U3;
use crate::base::storage::Storage;
use crate::base::{Matrix4, Vector, Vector3};

use crate::geometry::{Point3, Projective3};

/// A 3D perspective projection stored as a homogeneous 4x4 matrix.
pub struct Perspective3<T> {
    matrix: Matrix4<T>,
}

impl<T: RealField> Copy for Perspective3<T> {}

impl<T: RealField> Clone for Perspective3<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self::from_matrix_unchecked(self.matrix)
    }
}

impl<T: RealField> fmt::Debug for Perspective3<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.matrix.fmt(f)
    }
}

impl<T: RealField> PartialEq for Perspective3<T> {
    #[inline]
    fn eq(&self, right: &Self) -> bool {
        self.matrix == right.matrix
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<T: RealField + Serialize> Serialize for Perspective3<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.matrix.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<'a, T: RealField + Deserialize<'a>> Deserialize<'a> for Perspective3<T> {
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'a>,
    {
        let matrix = Matrix4::<T>::deserialize(deserializer)?;

        Ok(Self::from_matrix_unchecked(matrix))
    }
}

impl<T: RealField> Perspective3<T> {
    /// Creates a new perspective matrix from the aspect ratio, y field of view, and near/far planes.
    pub fn new(aspect: T, fovy: T, znear: T, zfar: T) -> Self {
        assert!(
            !relative_eq!(zfar - znear, T::zero()),
            "The near-plane and far-plane must not be superimposed."
        );
        assert!(
            !relative_eq!(aspect, T::zero()),
            "The aspect ratio must not be zero."
        );

        let matrix = Matrix4::identity();
        let mut res = Self::from_matrix_unchecked(matrix);

        res.set_fovy(fovy);
        res.set_aspect(aspect);
        res.set_znear_and_zfar(znear, zfar);

        res.matrix[(3, 3)] = T::zero();
        res.matrix[(3, 2)] = -T::one();

        res
    }

    /// Wraps the given matrix to interpret it as a 3D perspective matrix.
    ///
    /// It is not checked whether or not the given matrix actually represents a perspective
    /// projection.
    #[inline]
    pub fn from_matrix_unchecked(matrix: Matrix4<T>) -> Self {
        Self { matrix }
    }

    /// Retrieves the inverse of the underlying homogeneous matrix.
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Matrix4<T> {
        let mut res = self.to_homogeneous();

        res[(0, 0)] = T::one() / self.matrix[(0, 0)];
        res[(1, 1)] = T::one() / self.matrix[(1, 1)];
        res[(2, 2)] = T::zero();

        let m23 = self.matrix[(2, 3)];
        let m32 = self.matrix[(3, 2)];

        res[(2, 3)] = T::one() / m32;
        res[(3, 2)] = T::one() / m23;
        res[(3, 3)] = -self.matrix[(2, 2)] / (m23 * m32);

        res
    }

    /// Computes the corresponding homogeneous matrix.
    #[inline]
    #[must_use]
    pub fn to_homogeneous(self) -> Matrix4<T> {
        self.matrix.clone_owned()
    }

    /// A reference to the underlying homogeneous transformation matrix.
    #[inline]
    #[must_use]
    pub fn as_matrix(&self) -> &Matrix4<T> {
        &self.matrix
    }

    /// A reference to this transformation seen as a `Projective3`.
    #[inline]
    #[must_use]
    pub fn as_projective(&self) -> &Projective3<T> {
        // Safety: Self and Projective3 are both #[repr(transparent)] of a matrix.
        unsafe { &*(self as *const _ as *const _) }
    }

    /// This transformation seen as a `Projective3`.
    #[inline]
    #[must_use]
    pub fn to_projective(self) -> Projective3<T> {
        Projective3::from_matrix_unchecked(self.matrix)
    }

    /// Retrieves the underlying homogeneous matrix.
    #[inline]
    pub fn into_inner(self) -> Matrix4<T> {
        self.matrix
    }

    /// Retrieves the underlying homogeneous matrix.
    /// Deprecated: Use [Perspective3::into_inner] instead.
    #[deprecated(note = "use `.into_inner()` instead")]
    #[inline]
    pub fn unwrap(self) -> Matrix4<T> {
        self.matrix
    }

    /// Gets the `width / height` aspect ratio of the view frustum.
    #[inline]
    #[must_use]
    pub fn aspect(&self) -> T {
        self.matrix[(1, 1)] / self.matrix[(0, 0)]
    }

    /// Gets the y field of view of the view frustum.
    #[inline]
    #[must_use]
    pub fn fovy(&self) -> T {
        (T::one() / self.matrix[(1, 1)]).atan() * crate::convert(2.0)
    }

    /// Gets the near plane offset of the view frustum.
    #[inline]
    #[must_use]
    pub fn znear(&self) -> T {
        let ratio = (-self.matrix[(2, 2)] + T::one()) / (-self.matrix[(2, 2)] - T::one());

        self.matrix[(2, 3)] / (ratio * crate::convert(2.0))
            - self.matrix[(2, 3)] / crate::convert(2.0)
    }

    /// Gets the far plane offset of the view frustum.
    #[inline]
    #[must_use]
    pub fn zfar(&self) -> T {
        let ratio = (-self.matrix[(2, 2)] + T::one()) / (-self.matrix[(2, 2)] - T::one());

        (self.matrix[(2, 3)] - ratio * self.matrix[(2, 3)]) / crate::convert(2.0)
    }

    // TODO: add a method to retrieve znear and zfar simultaneously?

    // TODO: when we get specialization, specialize the Mul impl instead.
    /// Projects a point. Faster than matrix multiplication.
    #[inline]
    #[must_use]
    pub fn project_point(&self, p: &Point3<T>) -> Point3<T> {
        let inverse_denom = -T::one() / p[2];
        Point3::new(
            self.matrix[(0, 0)] * p[0] * inverse_denom,
            self.matrix[(1, 1)] * p[1] * inverse_denom,
            (self.matrix[(2, 2)] * p[2] + self.matrix[(2, 3)]) * inverse_denom,
        )
    }

    /// Un-projects a point. Faster than multiplication by the matrix inverse.
    #[inline]
    #[must_use]
    pub fn unproject_point(&self, p: &Point3<T>) -> Point3<T> {
        let inverse_denom = self.matrix[(2, 3)] / (p[2] + self.matrix[(2, 2)]);

        Point3::new(
            p[0] * inverse_denom / self.matrix[(0, 0)],
            p[1] * inverse_denom / self.matrix[(1, 1)],
            -inverse_denom,
        )
    }

    // TODO: when we get specialization, specialize the Mul impl instead.
    /// Projects a vector. Faster than matrix multiplication.
    #[inline]
    #[must_use]
    pub fn project_vector<SB>(&self, p: &Vector<T, U3, SB>) -> Vector3<T>
    where
        SB: Storage<T, U3>,
    {
        let inverse_denom = -T::one() / p[2];
        Vector3::new(
            self.matrix[(0, 0)] * p[0] * inverse_denom,
            self.matrix[(1, 1)] * p[1] * inverse_denom,
            self.matrix[(2, 2)],
        )
    }

    /// Updates this perspective matrix with a new `width / height` aspect ratio of the view
    /// frustum.
    #[inline]
    pub fn set_aspect(&mut self, aspect: T) {
        assert!(
            !relative_eq!(aspect, T::zero()),
            "The aspect ratio must not be zero."
        );
        self.matrix[(0, 0)] = self.matrix[(1, 1)] / aspect;
    }

    /// Updates this perspective with a new y field of view of the view frustum.
    #[inline]
    pub fn set_fovy(&mut self, fovy: T) {
        let old_m22 = self.matrix[(1, 1)];
        let new_m22 = T::one() / (fovy / crate::convert(2.0)).tan();
        self.matrix[(1, 1)] = new_m22;
        self.matrix[(0, 0)] *= new_m22 / old_m22;
    }

    /// Updates this perspective matrix with a new near plane offset of the view frustum.
    #[inline]
    pub fn set_znear(&mut self, znear: T) {
        let zfar = self.zfar();
        self.set_znear_and_zfar(znear, zfar);
    }

    /// Updates this perspective matrix with a new far plane offset of the view frustum.
    #[inline]
    pub fn set_zfar(&mut self, zfar: T) {
        let znear = self.znear();
        self.set_znear_and_zfar(znear, zfar);
    }

    /// Updates this perspective matrix with new near and far plane offsets of the view frustum.
    #[inline]
    pub fn set_znear_and_zfar(&mut self, znear: T, zfar: T) {
        self.matrix[(2, 2)] = (zfar + znear) / (znear - zfar);
        self.matrix[(2, 3)] = zfar * znear * crate::convert(2.0) / (znear - zfar);
    }
}

#[cfg(feature = "rand-no-std")]
impl<T: RealField> Distribution<Perspective3<T>> for Standard
where
    Standard: Distribution<T>,
{
    /// Generate an arbitrary random variate for testing purposes.
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> Perspective3<T> {
        use crate::base::helper;
        let znear = r.gen();
        let zfar = helper::reject_rand(r, |&x: &T| !(x - znear).is_zero());
        let aspect = helper::reject_rand(r, |&x: &T| !x.is_zero());

        Perspective3::new(aspect, r.gen(), znear, zfar)
    }
}

#[cfg(feature = "arbitrary")]
impl<T: RealField + Arbitrary> Arbitrary for Perspective3<T> {
    fn arbitrary(g: &mut Gen) -> Self {
        use crate::base::helper;
        let znear = Arbitrary::arbitrary(g);
        let zfar = helper::reject(g, |&x: &T| !(x - znear).is_zero());
        let aspect = helper::reject(g, |&x: &T| !x.is_zero());

        Self::new(aspect, Arbitrary::arbitrary(g), znear, zfar)
    }
}

impl<T: RealField> From<Perspective3<T>> for Matrix4<T> {
    #[inline]
    fn from(pers: Perspective3<T>) -> Self {
        pers.into_inner()
    }
}
