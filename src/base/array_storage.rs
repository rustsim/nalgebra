use std::fmt::{self, Debug, Formatter};
use std::mem;
// use std::hash::{Hash, Hasher};
#[cfg(feature = "abomonation-serialize")]
use std::io::{Result as IOResult, Write};
use std::ops::Mul;

#[cfg(feature = "serde-serialize-no-std")]
use serde::de::{Error, SeqAccess, Visitor};
#[cfg(feature = "serde-serialize-no-std")]
use serde::ser::SerializeSeq;
#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};
#[cfg(feature = "serde-serialize-no-std")]
use std::marker::PhantomData;

#[cfg(feature = "abomonation-serialize")]
use abomonation::Abomonation;

use crate::allocator::InnerAllocator;
use crate::base::default_allocator::DefaultAllocator;
use crate::base::dimension::{Const, ToTypenum};
use crate::base::storage::{
    ContiguousStorage, ContiguousStorageMut, ReshapableStorage, Storage, StorageMut,
};
use crate::base::Owned;

/*
 *
 * Static Storage.
 *
 */
/// A array-based statically sized matrix data storage.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ArrayStorage<T, const R: usize, const C: usize>(pub [[T; R]; C]);

// TODO: remove this once the stdlib implements Default for arrays.
impl<T: Default, const R: usize, const C: usize> Default for ArrayStorage<T, R, C>
where
    [[T; R]; C]: Default,
{
    #[inline]
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T: Debug, const R: usize, const C: usize> Debug for ArrayStorage<T, R, C> {
    #[inline]
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
        self.0.fmt(fmt)
    }
}

unsafe impl<T, const R: usize, const C: usize> Storage<T, Const<R>, Const<C>>
    for ArrayStorage<T, R, C>
where
    DefaultAllocator: InnerAllocator<T, Const<R>, Const<C>, Buffer = Self>,
{
    type RStride = Const<1>;
    type CStride = Const<R>;

    #[inline]
    fn ptr(&self) -> *const T {
        self.0.as_ptr() as *const T
    }

    #[inline]
    fn shape(&self) -> (Const<R>, Const<C>) {
        (Const, Const)
    }

    #[inline]
    fn strides(&self) -> (Self::RStride, Self::CStride) {
        (Const, Const)
    }

    #[inline]
    fn is_contiguous(&self) -> bool {
        true
    }

    #[inline]
    fn into_owned(self) -> Owned<T, Const<R>, Const<C>>
    where
        DefaultAllocator: InnerAllocator<T, Const<R>, Const<C>>,
    {
        Owned(self)
    }

    #[inline]
    fn clone_owned(&self) -> Owned<T, Const<R>, Const<C>>
    where
        T: Clone,
        DefaultAllocator: InnerAllocator<T, Const<R>, Const<C>>,
    {
        let it = self.as_slice().iter().cloned();
        Owned(DefaultAllocator::allocate_from_iterator(
            self.shape().0,
            self.shape().1,
            it,
        ))
    }

    #[inline]
    unsafe fn as_slice_unchecked(&self) -> &[T] {
        std::slice::from_raw_parts(self.ptr(), R * C)
    }
}

unsafe impl<T, const R: usize, const C: usize> StorageMut<T, Const<R>, Const<C>>
    for ArrayStorage<T, R, C>
where
    DefaultAllocator: InnerAllocator<T, Const<R>, Const<C>, Buffer = Self>,
{
    #[inline]
    fn ptr_mut(&mut self) -> *mut T {
        self.0.as_mut_ptr() as *mut T
    }

    #[inline]
    unsafe fn as_mut_slice_unchecked(&mut self) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.ptr_mut(), R * C)
    }
}

unsafe impl<T, const R: usize, const C: usize> ContiguousStorage<T, Const<R>, Const<C>>
    for ArrayStorage<T, R, C>
where
    DefaultAllocator: InnerAllocator<T, Const<R>, Const<C>, Buffer = Self>,
{
}

unsafe impl<T, const R: usize, const C: usize> ContiguousStorageMut<T, Const<R>, Const<C>>
    for ArrayStorage<T, R, C>
where
    DefaultAllocator: InnerAllocator<T, Const<R>, Const<C>, Buffer = Self>,
{
}

impl<T, const R1: usize, const C1: usize, const R2: usize, const C2: usize>
    ReshapableStorage<T, Const<R1>, Const<C1>, Const<R2>, Const<C2>> for ArrayStorage<T, R1, C1>
where
    Const<R1>: ToTypenum,
    Const<C1>: ToTypenum,
    Const<R2>: ToTypenum,
    Const<C2>: ToTypenum,
    <Const<R1> as ToTypenum>::Typenum: Mul<<Const<C1> as ToTypenum>::Typenum>,
    <Const<R2> as ToTypenum>::Typenum: Mul<
        <Const<C2> as ToTypenum>::Typenum,
        Output = typenum::Prod<
            <Const<R1> as ToTypenum>::Typenum,
            <Const<C1> as ToTypenum>::Typenum,
        >,
    >,
{
    type Output = ArrayStorage<T, R2, C2>;

    fn reshape_generic(self, _: Const<R2>, _: Const<C2>) -> Self::Output {
        unsafe {
            let data: [[T; R2]; C2] = mem::transmute_copy(&self.0);
            mem::forget(self.0);
            ArrayStorage(data)
        }
    }
}

/*
 *
 * Serialization.
 *
 */
// XXX: open an issue for serde so that it allows the serialization/deserialization of all arrays?
#[cfg(feature = "serde-serialize-no-std")]
impl<T, const R: usize, const C: usize> Serialize for ArrayStorage<T, R, C>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut serializer = serializer.serialize_seq(Some(R * C))?;

        for e in self.as_slice().iter() {
            serializer.serialize_element(e)?;
        }

        serializer.end()
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<'a, T, const R: usize, const C: usize> Deserialize<'a> for ArrayStorage<T, R, C>
where
    T: Deserialize<'a>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        deserializer.deserialize_seq(ArrayStorageVisitor::new())
    }
}

#[cfg(feature = "serde-serialize-no-std")]
/// A visitor that produces a matrix array.
struct ArrayStorageVisitor<T, const R: usize, const C: usize> {
    marker: PhantomData<T>,
}

#[cfg(feature = "serde-serialize-no-std")]
impl<T, const R: usize, const C: usize> ArrayStorageVisitor<T, R, C> {
    /// Construct a new sequence visitor.
    pub fn new() -> Self {
        ArrayStorageVisitor {
            marker: PhantomData,
        }
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<'a, T, const R: usize, const C: usize> Visitor<'a> for ArrayStorageVisitor<T, R, C>
where
    T: Deserialize<'a>,
{
    type Value = ArrayStorage<T, R, C>;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        formatter.write_str("a matrix array")
    }

    #[inline]
    fn visit_seq<V>(self, mut visitor: V) -> Result<ArrayStorage<T, R, C>, V::Error>
    where
        V: SeqAccess<'a>,
    {
        let mut out: Self::Value = unsafe { mem::MaybeUninit::uninit().assume_init() };
        let mut curr = 0;

        while let Some(value) = visitor.next_element()? {
            *out.as_mut_slice()
                .get_mut(curr)
                .ok_or_else(|| V::Error::invalid_length(curr, &self))? = value;
            curr += 1;
        }

        if curr == R * C {
            Ok(out)
        } else {
            Err(V::Error::invalid_length(curr, &self))
        }
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: Copy + bytemuck::Zeroable, const R: usize, const C: usize> bytemuck::Zeroable
    for ArrayStorage<T, R, C>
{
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: Copy + bytemuck::Pod, const R: usize, const C: usize> bytemuck::Pod
    for ArrayStorage<T, R, C>
{
}

#[cfg(feature = "abomonation-serialize")]
impl<T, const R: usize, const C: usize> Abomonation for ArrayStorage<T, R, C>
where
    T: Abomonation,
{
    unsafe fn entomb<W: Write>(&self, writer: &mut W) -> IOResult<()> {
        for element in self.as_slice() {
            element.entomb(writer)?;
        }

        Ok(())
    }

    unsafe fn exhume<'a, 'b>(&'a mut self, mut bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
        for element in self.as_mut_slice() {
            let temp = bytes;
            bytes = element.exhume(temp)?
        }
        Some(bytes)
    }

    fn extent(&self) -> usize {
        self.as_slice().iter().fold(0, |acc, e| acc + e.extent())
    }
}

#[cfg(feature = "rkyv-serialize-no-std")]
mod rkyv_impl {
    use super::ArrayStorage;
    use rkyv::{offset_of, project_struct, Archive, Deserialize, Fallible, Serialize};

    impl<T: Archive, const R: usize, const C: usize> Archive for ArrayStorage<T, R, C> {
        type Archived = ArrayStorage<T::Archived, R, C>;
        type Resolver = <[[T; R]; C] as Archive>::Resolver;

        fn resolve(
            &self,
            pos: usize,
            resolver: Self::Resolver,
            out: &mut core::mem::MaybeUninit<Self::Archived>,
        ) {
            self.0.resolve(
                pos + offset_of!(Self::Archived, 0),
                resolver,
                project_struct!(out: Self::Archived => 0),
            );
        }
    }

    impl<T: Serialize<S>, S: Fallible + ?Sized, const R: usize, const C: usize> Serialize<S>
        for ArrayStorage<T, R, C>
    {
        fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
            self.0.serialize(serializer)
        }
    }

    impl<T: Archive, D: Fallible + ?Sized, const R: usize, const C: usize>
        Deserialize<ArrayStorage<T, R, C>, D> for ArrayStorage<T::Archived, R, C>
    where
        T::Archived: Deserialize<T, D>,
    {
        fn deserialize(&self, deserializer: &mut D) -> Result<ArrayStorage<T, R, C>, D::Error> {
            Ok(ArrayStorage(self.0.deserialize(deserializer)?))
        }
    }
}
