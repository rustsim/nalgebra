#[cfg(feature = "abomonation-serialize")]
use std::io::{Result as IOResult, Write};

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;

use crate::allocator::InnerAllocator;
use crate::base::constraint::{SameNumberOfRows, ShapeConstraint};
use crate::base::default_allocator::DefaultAllocator;
use crate::base::dimension::{Dim, DimName, Dynamic, U1};
use crate::base::storage::{
    ContiguousStorage, ContiguousStorageMut, ReshapableStorage, Storage, StorageMut,
};
use crate::base::{Owned, Vector};

#[cfg(feature = "serde-serialize-no-std")]
use serde::{
    de::{Deserialize, Deserializer, Error},
    ser::{Serialize, Serializer},
};

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

/*
 *
 * Storage.
 *
 */
/// A Vec-based matrix data storage. It may be dynamically-sized.
#[derive(Eq, Debug, Clone, PartialEq)]
pub struct VecStorage<T, R: Dim, C: Dim> {
    data: Vec<T>,
    pub(crate) nrows: R,
    pub(crate) ncols: C,
}

#[cfg(feature = "serde-serialize")]
impl<T, R: Dim, C: Dim> Serialize for VecStorage<T, R, C>
where
    T: Serialize,
    R: Serialize,
    C: Serialize,
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: Serializer,
    {
        (&self.data, &self.nrows, &self.ncols).serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize")]
impl<'a, T, R: Dim, C: Dim> Deserialize<'a> for VecStorage<T, R, C>
where
    T: Deserialize<'a>,
    R: Deserialize<'a>,
    C: Deserialize<'a>,
{
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'a>,
    {
        let (data, nrows, ncols): (Vec<T>, R, C) = Deserialize::deserialize(deserializer)?;

        // SAFETY: make sure the data we deserialize have the
        //         correct number of elements.
        if nrows.value() * ncols.value() != data.len() {
            return Err(Des::Error::custom(format!(
                "Expected {} components, found {}",
                nrows.value() * ncols.value(),
                data.len()
            )));
        }

        Ok(Self { data, nrows, ncols })
    }
}

#[deprecated(note = "renamed to `VecStorage`")]
/// Renamed to [VecStorage].
pub type MatrixVec<T, R, C> = VecStorage<T, R, C>;

impl<T, R: Dim, C: Dim> VecStorage<T, R, C> {
    /// Creates a new dynamic matrix data storage from the given vector and shape.
    #[inline]
    pub fn new(nrows: R, ncols: C, data: Vec<T>) -> Self {
        assert!(
            nrows.value() * ncols.value() == data.len(),
            "Data storage buffer dimension mismatch."
        );
        Self { data, nrows, ncols }
    }

    /// The underlying data storage.
    #[inline]
    #[must_use]
    pub fn as_vec(&self) -> &Vec<T> {
        &self.data
    }

    /// The underlying mutable data storage.
    ///
    /// # Safety
    /// This is unsafe because this may cause UB if the size of the vector is changed
    /// by the user.
    #[inline]
    pub unsafe fn as_vec_mut(&mut self) -> &mut Vec<T> {
        &mut self.data
    }

    /// Resizes the underlying mutable data storage and unwraps it.
    ///
    /// # Safety
    /// If `sz` is larger than the current size, additional elements are uninitialized.
    /// If `sz` is smaller than the current size, additional elements are truncated.
    #[inline]
    pub unsafe fn resize(mut self, sz: usize) -> Vec<T> {
        let len = self.len();

        if sz < len {
            self.data.set_len(sz);
            self.data.shrink_to_fit();
        } else {
            self.data.reserve_exact(sz - len);
            self.data.set_len(sz);
        }

        self.data
    }

    /// The number of elements on the underlying vector.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the underlying vector contains no elements.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T, R: Dim, C: Dim> From<VecStorage<T, R, C>> for Vec<T> {
    fn from(vec: VecStorage<T, R, C>) -> Self {
        vec.data
    }
}

/*
 *
 * Dynamic − Static
 * Dynamic − Dynamic
 *
 */
unsafe impl<T, C: Dim> Storage<T, Dynamic, C> for VecStorage<T, Dynamic, C>
where
    DefaultAllocator: InnerAllocator<T, Dynamic, C, Buffer = Self>,
{
    type RStride = U1;
    type CStride = Dynamic;

    #[inline]
    fn ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    #[inline]
    fn shape(&self) -> (Dynamic, C) {
        (self.nrows, self.ncols)
    }

    #[inline]
    fn strides(&self) -> (Self::RStride, Self::CStride) {
        (Self::RStride::name(), self.nrows)
    }

    #[inline]
    fn is_contiguous(&self) -> bool {
        true
    }

    #[inline]
    fn into_owned(self) -> Owned<T, Dynamic, C> {
        Owned(self)
    }

    #[inline]
    fn clone_owned(&self) -> Owned<T, Dynamic, C>
    where
        T: Clone,
    {
        Owned(self.clone())
    }

    #[inline]
    unsafe fn as_slice_unchecked(&self) -> &[T] {
        &self.data
    }
}

unsafe impl<T, R: DimName> Storage<T, R, Dynamic> for VecStorage<T, R, Dynamic>
where
    DefaultAllocator: InnerAllocator<T, R, Dynamic, Buffer = Self>,
{
    type RStride = U1;
    type CStride = R;

    #[inline]
    fn ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    #[inline]
    fn shape(&self) -> (R, Dynamic) {
        (self.nrows, self.ncols)
    }

    #[inline]
    fn strides(&self) -> (Self::RStride, Self::CStride) {
        (Self::RStride::name(), self.nrows)
    }

    #[inline]
    fn is_contiguous(&self) -> bool {
        true
    }

    #[inline]
    fn into_owned(self) -> Owned<T, R, Dynamic> {
        Owned(self)
    }

    #[inline]
    fn clone_owned(&self) -> Owned<T, R, Dynamic>
    where
        T: Clone,
    {
        Owned(self.clone())
    }

    #[inline]
    unsafe fn as_slice_unchecked(&self) -> &[T] {
        &self.data
    }
}

/*
 *
 * StorageMut, ContiguousStorage.
 *
 */
unsafe impl<T, C: Dim> StorageMut<T, Dynamic, C> for VecStorage<T, Dynamic, C>
where
    DefaultAllocator: InnerAllocator<T, Dynamic, C, Buffer = Self>,
{
    #[inline]
    fn ptr_mut(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    #[inline]
    unsafe fn as_mut_slice_unchecked(&mut self) -> &mut [T] {
        &mut self.data[..]
    }
}

unsafe impl<T, C: Dim> ContiguousStorage<T, Dynamic, C> for VecStorage<T, Dynamic, C> where
    DefaultAllocator: InnerAllocator<T, Dynamic, C, Buffer = Self>
{
}

unsafe impl<T, C: Dim> ContiguousStorageMut<T, Dynamic, C> for VecStorage<T, Dynamic, C> where
    DefaultAllocator: InnerAllocator<T, Dynamic, C, Buffer = Self>
{
}

impl<T, C1: Dim, C2: Dim> ReshapableStorage<T, Dynamic, C1, Dynamic, C2>
    for VecStorage<T, Dynamic, C1>
{
    type Output = VecStorage<T, Dynamic, C2>;

    fn reshape_generic(self, nrows: Dynamic, ncols: C2) -> Self::Output {
        assert_eq!(nrows.value() * ncols.value(), self.data.len());
        VecStorage {
            data: self.data,
            nrows,
            ncols,
        }
    }
}

impl<T, C1: Dim, R2: DimName> ReshapableStorage<T, Dynamic, C1, R2, Dynamic>
    for VecStorage<T, Dynamic, C1>
{
    type Output = VecStorage<T, R2, Dynamic>;

    fn reshape_generic(self, nrows: R2, ncols: Dynamic) -> Self::Output {
        assert_eq!(nrows.value() * ncols.value(), self.data.len());
        VecStorage {
            data: self.data,
            nrows,
            ncols,
        }
    }
}

unsafe impl<T, R: DimName> StorageMut<T, R, Dynamic> for VecStorage<T, R, Dynamic>
where
    DefaultAllocator: InnerAllocator<T, R, Dynamic, Buffer = Self>,
{
    #[inline]
    fn ptr_mut(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    #[inline]
    unsafe fn as_mut_slice_unchecked(&mut self) -> &mut [T] {
        &mut self.data[..]
    }
}

impl<T, R1: DimName, C2: Dim> ReshapableStorage<T, R1, Dynamic, Dynamic, C2>
    for VecStorage<T, R1, Dynamic>
{
    type Output = VecStorage<T, Dynamic, C2>;

    fn reshape_generic(self, nrows: Dynamic, ncols: C2) -> Self::Output {
        assert_eq!(nrows.value() * ncols.value(), self.data.len());
        VecStorage {
            data: self.data,
            nrows,
            ncols,
        }
    }
}

impl<T, R1: DimName, R2: DimName> ReshapableStorage<T, R1, Dynamic, R2, Dynamic>
    for VecStorage<T, R1, Dynamic>
{
    type Output = VecStorage<T, R2, Dynamic>;

    fn reshape_generic(self, nrows: R2, ncols: Dynamic) -> Self::Output {
        assert_eq!(nrows.value() * ncols.value(), self.data.len());
        VecStorage {
            data: self.data,
            nrows,
            ncols,
        }
    }
}

#[cfg(feature = "abomonation-serialize")]
impl<T: Abomonation, R: Dim, C: Dim> Abomonation for VecStorage<T, R, C> {
    unsafe fn entomb<W: Write>(&self, writer: &mut W) -> IOResult<()> {
        self.data.entomb(writer)
    }

    unsafe fn exhume<'a, 'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
        self.data.exhume(bytes)
    }

    fn extent(&self) -> usize {
        self.data.extent()
    }
}

unsafe impl<T, R: DimName> ContiguousStorage<T, R, Dynamic> for VecStorage<T, R, Dynamic> where
    DefaultAllocator: InnerAllocator<T, R, Dynamic, Buffer = Self>
{
}

unsafe impl<T, R: DimName> ContiguousStorageMut<T, R, Dynamic> for VecStorage<T, R, Dynamic> where
    DefaultAllocator: InnerAllocator<T, R, Dynamic, Buffer = Self>
{
}

impl<T, R: Dim> Extend<T> for VecStorage<T, R, Dynamic> {
    /// Extends the number of columns of the `VecStorage` with elements
    /// from the given iterator.
    ///
    /// # Panics
    /// This function panics if the number of elements yielded by the
    /// given iterator is not a multiple of the number of rows of the
    /// `VecStorage`.
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.data.extend(iter);
        self.ncols = Dynamic::new(self.data.len() / self.nrows.value());
        assert!(self.data.len() % self.nrows.value() == 0,
          "The number of elements produced by the given iterator was not a multiple of the number of rows.");
    }
}

impl<'a, T: 'a + Copy, R: Dim> Extend<&'a T> for VecStorage<T, R, Dynamic> {
    /// Extends the number of columns of the `VecStorage` with elements
    /// from the given iterator.
    ///
    /// # Panics
    /// This function panics if the number of elements yielded by the
    /// given iterator is not a multiple of the number of rows of the
    /// `VecStorage`.
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().copied())
    }
}

impl<T: Clone, R: Dim, RV: Dim, SV> Extend<Vector<T, RV, SV>> for VecStorage<T, R, Dynamic>
where
    SV: Storage<T, RV>,
    ShapeConstraint: SameNumberOfRows<R, RV>,
{
    /// Extends the number of columns of the `VecStorage` with vectors
    /// from the given iterator.
    ///
    /// # Panics
    /// This function panics if the number of rows of each `Vector`
    /// yielded by the iterator is not equal to the number of rows
    /// of this `VecStorage`.
    fn extend<I: IntoIterator<Item = Vector<T, RV, SV>>>(&mut self, iter: I) {
        let nrows = self.nrows.value();
        let iter = iter.into_iter();
        let (lower, _upper) = iter.size_hint();
        self.data.reserve(nrows * lower);
        for vector in iter {
            assert_eq!(nrows, vector.shape().0);
            self.data.extend(vector.iter().cloned());
        }
        self.ncols = Dynamic::new(self.data.len() / nrows);
    }
}

impl<T> Extend<T> for VecStorage<T, Dynamic, U1> {
    /// Extends the number of rows of the `VecStorage` with elements
    /// from the given iterator.
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.data.extend(iter);
        self.nrows = Dynamic::new(self.data.len());
    }
}
