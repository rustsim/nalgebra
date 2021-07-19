//! The default matrix data storage allocator.
//!
//! This will use stack-allocated buffers for matrices with dimensions known at compile-time, and
//! heap-allocated buffers for matrices with at least one dimension unknown at compile-time.

use std::cmp;
use std::fmt;
use std::mem::{self, ManuallyDrop, MaybeUninit};
use std::ptr;

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;

#[cfg(any(feature = "alloc", feature = "std"))]
use crate::base::dimension::Dynamic;

use super::Const;
use crate::base::allocator::{Allocator, InnerAllocator, Reallocator};
use crate::base::array_storage::ArrayStorage;
use crate::base::dimension::{Dim, DimName};
use crate::base::storage::{
    ContiguousStorage, ContiguousStorageMut, InnerOwned, Storage, StorageMut,
};
use crate::base::vec_storage::VecStorage;
use crate::U1;

/*
 *
 * Allocator.
 *
 */
/// A helper struct that controls how the storage for a matrix should be allocated.
///
/// This struct is useless on its own. Instead, it's used in trait
/// An allocator based on `GenericArray` and `VecStorage` for statically-sized and dynamically-sized
/// matrices respectively.
pub struct DefaultAllocator;

// Static - Static
impl<T, const R: usize, const C: usize> InnerAllocator<T, Const<R>, Const<C>> for DefaultAllocator {
    type Buffer = ArrayStorage<T, R, C>;

    #[inline]
    fn allocate_from_iterator<I: IntoIterator<Item = T>>(
        nrows: Const<R>,
        ncols: Const<C>,
        iter: I,
    ) -> Self::Buffer {
        let mut res = Self::allocate_uninitialized(nrows, ncols);
        let mut count = 0;

        for (res, e) in res.as_mut_slice().iter_mut().zip(iter.into_iter()) {
            *res = MaybeUninit::new(e);
            count += 1;
        }

        assert!(
            count == nrows.value() * ncols.value(),
            "Matrix init. from iterator: iterator not long enough."
        );

        // Safety: we have initialized all entries.
        unsafe { <Self as Allocator<T, Const<R>, Const<C>>>::assume_init(res) }
    }
}

impl<T, const R: usize, const C: usize> Allocator<T, Const<R>, Const<C>> for DefaultAllocator {
    #[inline]
    fn allocate_uninitialized(
        _: Const<R>,
        _: Const<C>,
    ) -> InnerOwned<MaybeUninit<T>, Const<R>, Const<C>> {
        // SAFETY: An uninitialized `[MaybeUninit<_>; _]` is valid.
        let array = unsafe { MaybeUninit::uninit().assume_init() };
        ArrayStorage(array)
    }

    #[inline]
    unsafe fn assume_init(
        uninit: <Self as InnerAllocator<MaybeUninit<T>, Const<R>, Const<C>>>::Buffer,
    ) -> InnerOwned<T, Const<R>, Const<C>> {
        // Safety:
        // * The caller guarantees that all elements of the array are initialized
        // * `MaybeUninit<T>` and T are guaranteed to have the same layout
        // * `MaybeUnint` does not drop, so there are no double-frees
        // And thus the conversion is safe
        ArrayStorage((&uninit as *const _ as *const [_; C]).read())
    }

    /// Specifies that a given buffer's entries should be manually dropped.
    #[inline]
    fn manually_drop(
        buf: <Self as InnerAllocator<T, Const<R>, Const<C>>>::Buffer,
    ) -> <Self as InnerAllocator<ManuallyDrop<T>, Const<R>, Const<C>>>::Buffer {
        // SAFETY:
        // * `ManuallyDrop<T>` and T are guaranteed to have the same layout
        // * `ManuallyDrop` does not drop, so there are no double-frees
        // And thus the conversion is safe
        unsafe { ArrayStorage((&ManuallyDrop::new(buf) as *const _ as *const [_; C]).read()) }
    }
}

// Dynamic - Static
// Dynamic - Dynamic
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, C: Dim> InnerAllocator<T, Dynamic, C> for DefaultAllocator {
    type Buffer = VecStorage<T, Dynamic, C>;

    #[inline]
    fn allocate_from_iterator<I: IntoIterator<Item = T>>(
        nrows: Dynamic,
        ncols: C,
        iter: I,
    ) -> Self::Buffer {
        let it = iter.into_iter();
        let res: Vec<T> = it.collect();
        assert!(res.len() == nrows.value() * ncols.value(),
                "Allocation from iterator error: the iterator did not yield the correct number of elements.");

        VecStorage::new(nrows, ncols, res)
    }
}

impl<T, C: Dim> Allocator<T, Dynamic, C> for DefaultAllocator {
    #[inline]
    fn allocate_uninitialized(nrows: Dynamic, ncols: C) -> InnerOwned<MaybeUninit<T>, Dynamic, C> {
        let mut data = Vec::new();
        let length = nrows.value() * ncols.value();
        data.reserve_exact(length);
        data.resize_with(length, MaybeUninit::uninit);

        VecStorage::new(nrows, ncols, data)
    }

    #[inline]
    unsafe fn assume_init(
        uninit: InnerOwned<MaybeUninit<T>, Dynamic, C>,
    ) -> InnerOwned<T, Dynamic, C> {
        // Avoids a double-drop.
        let (nrows, ncols) = uninit.shape();
        let vec: Vec<_> = uninit.into();
        let mut md = ManuallyDrop::new(vec);

        // Safety:
        // - MaybeUninit<T> has the same alignment and layout as T.
        // - The length and capacity come from a valid vector.
        let new_data = Vec::from_raw_parts(md.as_mut_ptr() as *mut _, md.len(), md.capacity());

        VecStorage::new(nrows, ncols, new_data)
    }

    #[inline]
    fn manually_drop(
        buf: <Self as InnerAllocator<T, Dynamic, C>>::Buffer,
    ) -> <Self as InnerAllocator<ManuallyDrop<T>, Dynamic, C>>::Buffer {
        // Avoids a double-drop.
        let (nrows, ncols) = buf.shape();
        let vec: Vec<_> = buf.into();
        let mut md = ManuallyDrop::new(vec);

        // Safety:
        // - ManuallyDrop<T> has the same alignment and layout as T.
        // - The length and capacity come from a valid vector.
        let new_data =
            unsafe { Vec::from_raw_parts(md.as_mut_ptr() as *mut _, md.len(), md.capacity()) };

        VecStorage::new(nrows, ncols, new_data)
    }
}

// Static - Dynamic
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, R: DimName> InnerAllocator<T, R, Dynamic> for DefaultAllocator {
    type Buffer = VecStorage<T, R, Dynamic>;

    #[inline]
    fn allocate_from_iterator<I: IntoIterator<Item = T>>(
        nrows: R,
        ncols: Dynamic,
        iter: I,
    ) -> InnerOwned<T, R, Dynamic> {
        let it = iter.into_iter();
        let res: Vec<T> = it.collect();
        assert!(res.len() == nrows.value() * ncols.value(),
                "Allocation from iterator error: the iterator did not yield the correct number of elements.");

        VecStorage::new(nrows, ncols, res)
    }
}

impl<T, R: DimName> Allocator<T, R, Dynamic> for DefaultAllocator {
    #[inline]
    fn allocate_uninitialized(nrows: R, ncols: Dynamic) -> InnerOwned<MaybeUninit<T>, R, Dynamic> {
        let mut data = Vec::new();
        let length = nrows.value() * ncols.value();
        data.reserve_exact(length);
        data.resize_with(length, MaybeUninit::uninit);

        VecStorage::new(nrows, ncols, data)
    }

    #[inline]
    unsafe fn assume_init(
        uninit: InnerOwned<MaybeUninit<T>, R, Dynamic>,
    ) -> InnerOwned<T, R, Dynamic> {
        // Avoids a double-drop.
        let (nrows, ncols) = uninit.shape();
        let vec: Vec<_> = uninit.into();
        let mut md = ManuallyDrop::new(vec);

        // Safety:
        // - MaybeUninit<T> has the same alignment and layout as T.
        // - The length and capacity come from a valid vector.
        let new_data = Vec::from_raw_parts(md.as_mut_ptr() as *mut _, md.len(), md.capacity());

        VecStorage::new(nrows, ncols, new_data)
    }

    #[inline]
    fn manually_drop(
        buf: <Self as InnerAllocator<T, R, Dynamic>>::Buffer,
    ) -> <Self as InnerAllocator<ManuallyDrop<T>, R, Dynamic>>::Buffer {
        // Avoids a double-drop.
        let (nrows, ncols) = buf.shape();
        let vec: Vec<_> = buf.into();
        let mut md = ManuallyDrop::new(vec);

        // Safety:
        // - ManuallyDrop<T> has the same alignment and layout as T.
        // - The length and capacity come from a valid vector.
        let new_data =
            unsafe { Vec::from_raw_parts(md.as_mut_ptr() as *mut _, md.len(), md.capacity()) };

        VecStorage::new(nrows, ncols, new_data)
    }
}

/// The owned storage type for a matrix.
#[repr(transparent)]
pub struct Owned<T, R: Dim, C: Dim>(pub InnerOwned<T, R, C>)
where
    DefaultAllocator: Allocator<T, R, C>;

impl<T: Copy, R: DimName, C: DimName> Copy for Owned<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
    InnerOwned<T, R, C>: Copy,
{
}

impl<T: Clone, R: Dim, C: Dim> Clone for Owned<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    fn clone(&self) -> Self {
        if Self::is_array() {
            // We first clone the data.
            let slice = unsafe { self.as_slice_unchecked() };
            let vec = ManuallyDrop::new(slice.to_owned());

            // We then transmute it back into an array and then an Owned.
            unsafe { mem::transmute_copy(&*vec.as_ptr()) }

            // TODO: check that the auxiliary copy is elided.
        } else {
            // We first clone the data.
            let clone = ManuallyDrop::new(self.as_vec_storage().clone());

            // We then transmute it back into an Owned.
            unsafe { mem::transmute_copy(&clone) }

            // TODO: check that the auxiliary copy is elided.
        }
    }
}

impl<T: fmt::Debug, R: Dim, C: Dim> fmt::Debug for Owned<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if Self::is_array() {
            let slice = unsafe { self.as_slice_unchecked() };
            slice.fmt(f)
        } else {
            self.as_vec_storage().fmt(f)
        }
    }
}

impl<T, R: Dim, C: Dim> Owned<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    /// Returns whether `Self` stores an [`ArrayStorage`].
    fn is_array() -> bool {
        R::is_static() && C::is_static()
    }

    /// Returns whether `Self` stores a [`VecStorage`].
    fn is_vec() -> bool {
        !Self::is_array()
    }

    /// Returns the underlying [`VecStorage`]. Does not do any sort of static
    /// type checking.
    ///
    /// # Panics
    /// This method will panic if `Self` does not contain a [`VecStorage`].
    fn as_vec_storage(&self) -> &VecStorage<T, R, C> {
        assert!(Self::is_vec());

        // Safety: `self` is transparent and must contain a `VecStorage`.
        unsafe { &*(&self as *const _ as *const _) }
    }
}

unsafe impl<T, R: Dim, C: Dim> Storage<T, R, C> for Owned<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    type RStride = U1;

    type CStride = R;

    fn ptr(&self) -> *const T {
        if Self::is_array() {
            &self as *const _ as *const T
        } else {
            self.as_vec_storage().as_vec().as_ptr()
        }
    }

    fn shape(&self) -> (R, C) {
        if Self::is_array() {
            (R::default(), C::default())
        } else {
            let vec = self.as_vec_storage();
            (vec.nrows, vec.ncols)
        }
    }

    fn strides(&self) -> (Self::RStride, Self::CStride) {
        if Self::is_array() {
            (U1::name(), R::default())
        } else {
            let vec = self.as_vec_storage();
            (U1::name(), vec.nrows)
        }
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    unsafe fn as_slice_unchecked(&self) -> &[T] {
        if Self::is_array() {
            std::slice::from_raw_parts(
                self.ptr(),
                R::try_to_usize().unwrap() * C::try_to_usize().unwrap(),
            )
        } else {
            self.as_vec_storage().as_vec().as_ref()
        }
    }

    fn into_owned(self) -> Owned<T, R, C> {
        self
    }

    fn clone_owned(&self) -> Owned<T, R, C>
    where
        T: Clone,
    {
        self.clone()
    }
}

unsafe impl<T, R: Dim, C: Dim> StorageMut<T, R, C> for Owned<T, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    fn ptr_mut(&mut self) -> *mut T {
        todo!()
    }

    unsafe fn as_mut_slice_unchecked(&mut self) -> &mut [T] {
        todo!()
    }
}

unsafe impl<T, R: Dim, C: Dim> ContiguousStorage<T, R, C> for Owned<T, R, C> where
    DefaultAllocator: Allocator<T, R, C>
{
}

unsafe impl<T, R: Dim, C: Dim> ContiguousStorageMut<T, R, C> for Owned<T, R, C> where
    DefaultAllocator: Allocator<T, R, C>
{
}

/*
 *
 * Reallocator.
 *
 */
// Anything -> Static × Static
impl<T, RFrom: Dim, CFrom: Dim, const RTO: usize, const CTO: usize>
    Reallocator<T, RFrom, CFrom, Const<RTO>, Const<CTO>> for DefaultAllocator
where
    Self: Allocator<T, RFrom, CFrom>,
{
    #[inline]
    unsafe fn reallocate_copy(
        rto: Const<RTO>,
        cto: Const<CTO>,
        buf: InnerOwned<T, RFrom, CFrom>,
    ) -> ArrayStorage<T, RTO, CTO> {
        let mut res =
            <Self as Allocator<_, Const<RTO>, Const<CTO>>>::allocate_uninitialized(rto, cto);

        let (rfrom, cfrom) = buf.shape();

        let len_from = rfrom.value() * cfrom.value();
        let len_to = rto.value() * cto.value();
        ptr::copy_nonoverlapping(
            buf.ptr(),
            res.ptr_mut() as *mut T,
            cmp::min(len_from, len_to),
        );

        // Safety: TODO
        <Self as Allocator<_, Const<RTO>, Const<CTO>>>::assume_init(res)
    }
}

// Static × Static -> Dynamic × Any
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, CTo, const RFROM: usize, const CFROM: usize>
    Reallocator<T, Const<RFROM>, Const<CFROM>, Dynamic, CTo> for DefaultAllocator
where
    CTo: Dim,
{
    #[inline]
    unsafe fn reallocate_copy(
        rto: Dynamic,
        cto: CTo,
        buf: ArrayStorage<T, RFROM, CFROM>,
    ) -> VecStorage<T, Dynamic, CTo> {
        let mut res = <Self as Allocator<T, Dynamic, CTo>>::allocate_uninitialized(rto, cto);

        let (rfrom, cfrom) = buf.shape();

        let len_from = rfrom.value() * cfrom.value();
        let len_to = rto.value() * cto.value();
        ptr::copy_nonoverlapping(
            buf.ptr(),
            res.ptr_mut() as *mut T,
            cmp::min(len_from, len_to),
        );

        <Self as Allocator<T, Dynamic, CTo>>::assume_init(res)
    }
}

// Static × Static -> Static × Dynamic
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, RTo, const RFROM: usize, const CFROM: usize>
    Reallocator<T, Const<RFROM>, Const<CFROM>, RTo, Dynamic> for DefaultAllocator
where
    RTo: DimName,
{
    #[inline]
    unsafe fn reallocate_copy(
        rto: RTo,
        cto: Dynamic,
        buf: ArrayStorage<T, RFROM, CFROM>,
    ) -> VecStorage<T, RTo, Dynamic> {
        let mut res = <Self as Allocator<T, RTo, Dynamic>>::allocate_uninitialized(rto, cto);

        let (rfrom, cfrom) = buf.shape();

        let len_from = rfrom.value() * cfrom.value();
        let len_to = rto.value() * cto.value();
        ptr::copy_nonoverlapping(
            buf.ptr(),
            res.ptr_mut() as *mut T,
            cmp::min(len_from, len_to),
        );

        <Self as Allocator<T, RTo, Dynamic>>::assume_init(res)
    }
}

// All conversion from a dynamic buffer to a dynamic buffer.
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, CFrom: Dim, CTo: Dim> Reallocator<T, Dynamic, CFrom, Dynamic, CTo> for DefaultAllocator {
    #[inline]
    unsafe fn reallocate_copy(
        rto: Dynamic,
        cto: CTo,
        buf: VecStorage<T, Dynamic, CFrom>,
    ) -> VecStorage<T, Dynamic, CTo> {
        let new_buf = buf.resize(rto.value() * cto.value());
        VecStorage::new(rto, cto, new_buf)
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, CFrom: Dim, RTo: DimName> Reallocator<T, Dynamic, CFrom, RTo, Dynamic>
    for DefaultAllocator
{
    #[inline]
    unsafe fn reallocate_copy(
        rto: RTo,
        cto: Dynamic,
        buf: VecStorage<T, Dynamic, CFrom>,
    ) -> VecStorage<T, RTo, Dynamic> {
        let new_buf = buf.resize(rto.value() * cto.value());
        VecStorage::new(rto, cto, new_buf)
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, RFrom: DimName, CTo: Dim> Reallocator<T, RFrom, Dynamic, Dynamic, CTo>
    for DefaultAllocator
{
    #[inline]
    unsafe fn reallocate_copy(
        rto: Dynamic,
        cto: CTo,
        buf: VecStorage<T, RFrom, Dynamic>,
    ) -> VecStorage<T, Dynamic, CTo> {
        let new_buf = buf.resize(rto.value() * cto.value());
        VecStorage::new(rto, cto, new_buf)
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, RFrom: DimName, RTo: DimName> Reallocator<T, RFrom, Dynamic, RTo, Dynamic>
    for DefaultAllocator
{
    #[inline]
    unsafe fn reallocate_copy(
        rto: RTo,
        cto: Dynamic,
        buf: VecStorage<T, RFrom, Dynamic>,
    ) -> VecStorage<T, RTo, Dynamic> {
        let new_buf = buf.resize(rto.value() * cto.value());
        VecStorage::new(rto, cto, new_buf)
    }
}
