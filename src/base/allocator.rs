//! Abstract definition of a matrix data storage allocator.

use std::mem::{ManuallyDrop, MaybeUninit};

use crate::base::constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use crate::base::dimension::{Dim, U1};
use crate::base::storage::ContiguousStorageMut;
use crate::base::DefaultAllocator;

/// A matrix allocator of a memory buffer that may contain `R::to_usize() * C::to_usize()`
/// elements of type `T`.
///
/// An allocator is said to be:
///   − static:  if `R` and `C` both implement `DimName`.
///   − dynamic: if either one (or both) of `R` or `C` is equal to `Dynamic`.
///
/// Every allocator must be both static and dynamic. Though not all implementations may share the
/// same `Buffer` type.
///
/// If you also want to be able to create uninitizalized or manually dropped memory buffers, see
/// [`Allocator`].
pub trait InnerAllocator<T, R: Dim, C: Dim = U1>: 'static + Sized {
    /// The type of buffer this allocator can instanciate.
    type Buffer: ContiguousStorageMut<T, R, C>;

    /// Allocates a buffer initialized with the content of the given iterator.
    fn allocate_from_iterator<I: IntoIterator<Item = T>>(
        nrows: R,
        ncols: C,
        iter: I,
    ) -> Self::Buffer;
}

/// Same as the [`InnerAllocator`] trait, but also provides methods to build uninitialized buffers,
/// or buffers whose entries must be manually dropped.
pub trait Allocator<T, R: Dim, C: Dim = U1>:
    InnerAllocator<T, R, C>
    + InnerAllocator<MaybeUninit<T>, R, C>
    + InnerAllocator<ManuallyDrop<T>, R, C>
{
    /// Allocates a buffer with the given number of rows and columns without initializing its content.
    fn allocate_uninitialized(
        nrows: R,
        ncols: C,
    ) -> <Self as InnerAllocator<MaybeUninit<T>, R, C>>::Buffer;

    /// Assumes a data buffer to be initialized. This operation should be near zero-cost.
    ///
    /// # Safety
    /// The user must make sure that every single entry of the buffer has been initialized,
    /// or Undefined Behavior will immediately occur.    
    unsafe fn assume_init(
        uninit: <Self as InnerAllocator<MaybeUninit<T>, R, C>>::Buffer,
    ) -> <Self as InnerAllocator<T, R, C>>::Buffer;

    /// Specifies that a given buffer's entries should be manually dropped.
    fn manually_drop(
        buf: <Self as InnerAllocator<T, R, C>>::Buffer,
    ) -> <Self as InnerAllocator<ManuallyDrop<T>, R, C>>::Buffer;
}


/// A matrix reallocator. Changes the size of the memory buffer that initially contains (RFrom ×
/// CFrom) elements to a smaller or larger size (RTo, CTo).
pub trait Reallocator<T, RFrom: Dim, CFrom: Dim, RTo: Dim, CTo: Dim>:
    Allocator<T, RFrom, CFrom> + Allocator<T, RTo, CTo>
{
    /// Reallocates a buffer of shape `(RTo, CTo)`, possibly reusing a previously allocated buffer
    /// `buf`. Data stored by `buf` are linearly copied to the output:
    ///
    /// # Safety
    /// **NO! THIS IS STILL UB!**
    /// * The copy is performed as if both were just arrays (without a matrix structure).
    /// * If `buf` is larger than the output size, then extra elements of `buf` are truncated.
    /// * If `buf` is smaller than the output size, then extra elements of the output are left
    /// uninitialized.
    unsafe fn reallocate_copy(
        nrows: RTo,
        ncols: CTo,
        buf: <Self as InnerAllocator<T, RFrom, CFrom>>::Buffer,
    ) -> <Self as InnerAllocator<T, RTo, CTo>>::Buffer;
}

/// The number of rows of the result of a componentwise operation on two matrices.
pub type SameShapeR<R1, R2> = <ShapeConstraint as SameNumberOfRows<R1, R2>>::Representative;

/// The number of columns of the result of a componentwise operation on two matrices.
pub type SameShapeC<C1, C2> = <ShapeConstraint as SameNumberOfColumns<C1, C2>>::Representative;

// TODO: Bad name.
/// Restricts the given number of rows and columns to be respectively the same.
pub trait SameShapeAllocator<T, R1: Dim, C1: Dim, R2: Dim, C2: Dim>:
    Allocator<T, R1, C1> + Allocator<T, SameShapeR<R1, R2>, SameShapeC<C1, C2>>
where
    ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
{
}

impl<T, R1: Dim, R2: Dim, C1: Dim, C2: Dim> SameShapeAllocator<T, R1, C1, R2, C2>
    for DefaultAllocator
where
    DefaultAllocator: Allocator<T, R1, C1> + Allocator<T, SameShapeR<R1, R2>, SameShapeC<C1, C2>>,
    ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
{
}

// XXX: Bad name.
/// Restricts the given number of rows to be equal.
pub trait SameShapeVectorAllocator<T, R1: Dim, R2: Dim>:
    Allocator<T, R1> + Allocator<T, SameShapeR<R1, R2>> + SameShapeAllocator<T, R1, U1, R2, U1>
where
    ShapeConstraint: SameNumberOfRows<R1, R2>,
{
}

impl<T, R1: Dim, R2: Dim> SameShapeVectorAllocator<T, R1, R2> for DefaultAllocator
where
    DefaultAllocator: Allocator<T, R1, U1> + Allocator<T, SameShapeR<R1, R2>>,
    ShapeConstraint: SameNumberOfRows<R1, R2>,
{
}
