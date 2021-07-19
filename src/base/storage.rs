//! Abstract definition of a matrix data storage.

use std::ptr;

use crate::base::allocator::{Allocator, InnerAllocator, SameShapeC, SameShapeR};
use crate::base::default_allocator::DefaultAllocator;
use crate::base::dimension::{Dim, U1};
use crate::base::Owned;

/*
 * Aliases for allocation results.
 */
/// The data storage for the sum of two matrices with dimensions `(R1, C1)` and `(R2, C2)`.
pub type SameShapeStorage<T, R1, C1, R2, C2> =
    <DefaultAllocator as InnerAllocator<T, SameShapeR<R1, R2>, SameShapeC<C1, C2>>>::Buffer;

// TODO: better name than Owned ?
/// The owned data storage that can be allocated from `S`.
pub type InnerOwned<T, R, C = U1> = <DefaultAllocator as InnerAllocator<T, R, C>>::Buffer;

/// The row-stride of the owned data storage for a buffer of dimension `(R, C)`.
pub type RStride<T, R, C = U1> =
    <<DefaultAllocator as InnerAllocator<T, R, C>>::Buffer as Storage<T, R, C>>::RStride;

/// The column-stride of the owned data storage for a buffer of dimension `(R, C)`.
pub type CStride<T, R, C = U1> =
    <<DefaultAllocator as InnerAllocator<T, R, C>>::Buffer as Storage<T, R, C>>::CStride;

/// The trait shared by all matrix data storage.
///
/// TODO: doc
///
/// Note that `Self` must always have a number of elements compatible with the matrix length (given
/// by `R` and `C` if they are known at compile-time). For example, implementors of this trait
/// should **not** allow the user to modify the size of the underlying buffer with safe methods
/// (for example the `VecStorage::data_mut` method is unsafe because the user could change the
/// vector's size so that it no longer contains enough elements: this will lead to UB.
pub unsafe trait Storage<T, R: Dim, C: Dim = U1>: Sized {
    /// The static stride of this storage's rows.
    type RStride: Dim;

    /// The static stride of this storage's columns.
    type CStride: Dim;

    /// The matrix data pointer.
    fn ptr(&self) -> *const T;

    /// The dimension of the matrix at run-time. Arr length of zero indicates the additive identity
    /// element of any dimension. Must be equal to `Self::dimension()` if it is not `None`.
    fn shape(&self) -> (R, C);

    /// The spacing between consecutive row elements and consecutive column elements.
    ///
    /// For example this returns `(1, 5)` for a row-major matrix with 5 columns.
    fn strides(&self) -> (Self::RStride, Self::CStride);

    /// Compute the index corresponding to the irow-th row and icol-th column of this matrix. The
    /// index must be such that the following holds:
    ///
    /// ```ignore
    /// let lindex = self.linear_index(irow, icol);
    /// assert!(*self.get_unchecked(irow, icol) == *self.get_unchecked_linear(lindex))
    /// ```
    #[inline]
    fn linear_index(&self, irow: usize, icol: usize) -> usize {
        let (rstride, cstride) = self.strides();

        irow * rstride.value() + icol * cstride.value()
    }

    /// Gets the address of the i-th matrix component without performing bound-checking.
    ///
    /// # Safety
    /// If the index is out of bounds, dereferencing the result will cause undefined behavior.
    #[inline]
    fn get_address_unchecked_linear(&self, i: usize) -> *const T {
        self.ptr().wrapping_add(i)
    }

    /// Gets the address of the i-th matrix component without performing bound-checking.
    ///
    /// # Safety
    /// If the index is out of bounds, dereferencing the result will cause undefined behavior.
    #[inline]
    fn get_address_unchecked(&self, irow: usize, icol: usize) -> *const T {
        self.get_address_unchecked_linear(self.linear_index(irow, icol))
    }

    /// Retrieves a reference to the i-th element without bound-checking.
    ///
    /// # Safety
    /// If the index is out of bounds, the method will cause undefined behavior.
    #[inline]
    unsafe fn get_unchecked_linear(&self, i: usize) -> &T {
        &*self.get_address_unchecked_linear(i)
    }

    /// Retrieves a reference to the i-th element without bound-checking.
    ///
    /// # Safety
    /// If the index is out of bounds, the method will cause undefined behavior.
    #[inline]
    unsafe fn get_unchecked(&self, irow: usize, icol: usize) -> &T {
        self.get_unchecked_linear(self.linear_index(irow, icol))
    }

    /// Indicates whether this data buffer stores its elements contiguously.
    ///
    /// # Safety
    /// This function must not return `true` if the underlying storage is not contiguous,
    /// or undefined behaviour will occur.
    fn is_contiguous(&self) -> bool;

    /// Retrieves the data buffer as a contiguous slice.
    ///
    /// # Safety
    /// The matrix components may not be stored in a contiguous way, depending on the strides.
    /// This method is unsafe because this can yield to invalid aliasing when called on some pairs
    /// of matrix slices originating from the same matrix with strides.
    ///
    /// Call the safe alternative `matrix.as_slice()` instead.
    unsafe fn as_slice_unchecked(&self) -> &[T];

    /// Builds a matrix data storage that does not contain any reference.
    fn into_owned(self) -> Owned<T, R, C>
    where
        T: Clone,
        DefaultAllocator: Allocator<T, R, C>;

    /// Clones this data storage to one that does not contain any reference.
    fn clone_owned(&self) -> Owned<T, R, C>
    where
        T: Clone,
        DefaultAllocator: Allocator<T, R, C>;
}

/// Trait implemented by matrix data storage that can provide a mutable access to its elements.
///
/// Note that a mutable access does not mean that the matrix owns its data. For example, a mutable
/// matrix slice can provide mutable access to its elements even if it does not own its data (it
/// contains only an internal reference to them).
pub unsafe trait StorageMut<T, R: Dim, C: Dim = U1>: Storage<T, R, C> {
    /// The matrix mutable data pointer.
    fn ptr_mut(&mut self) -> *mut T;

    /// Gets the mutable address of the i-th matrix component without performing bound-checking.
    ///
    /// # Safety
    /// If the index is out of bounds, dereferencing the result will cause undefined behavior.
    #[inline]
    fn get_address_unchecked_linear_mut(&mut self, i: usize) -> *mut T {
        self.ptr_mut().wrapping_add(i)
    }

    /// Gets the mutable address of the i-th matrix component without performing bound-checking.
    ///
    /// # Safety
    /// If the index is out of bounds, dereferencing the result will cause undefined behavior.
    #[inline]
    fn get_address_unchecked_mut(&mut self, irow: usize, icol: usize) -> *mut T {
        let lid = self.linear_index(irow, icol);
        self.get_address_unchecked_linear_mut(lid)
    }

    /// Retrieves a mutable reference to the i-th element without bound-checking.
    ///
    /// # Safety
    /// If the index is out of bounds, the method will cause undefined behavior.
    unsafe fn get_unchecked_linear_mut(&mut self, i: usize) -> &mut T {
        &mut *self.get_address_unchecked_linear_mut(i)
    }

    /// Retrieves a mutable reference to the element at `(irow, icol)` without bound-checking.
    ///
    /// # Safety
    /// If the index is out of bounds, the method will cause undefined behavior.
    #[inline]
    unsafe fn get_unchecked_mut(&mut self, irow: usize, icol: usize) -> &mut T {
        &mut *self.get_address_unchecked_mut(irow, icol)
    }

    /// Swaps two elements using their linear index without bound-checking.
    ///
    /// # Safety
    /// If the indices are out of bounds, the method will cause undefined behavior.
    #[inline]
    unsafe fn swap_unchecked_linear(&mut self, i1: usize, i2: usize) {
        let a = self.get_address_unchecked_linear_mut(i1);
        let b = self.get_address_unchecked_linear_mut(i2);

        ptr::swap(a, b);
    }

    /// Swaps two elements without bound-checking.
    ///
    /// # Safety
    /// If the indices are out of bounds, the method will cause undefined behavior.
    #[inline]
    unsafe fn swap_unchecked(&mut self, row_col1: (usize, usize), row_col2: (usize, usize)) {
        let lid1 = self.linear_index(row_col1.0, row_col1.1);
        let lid2 = self.linear_index(row_col2.0, row_col2.1);

        self.swap_unchecked_linear(lid1, lid2)
    }

    /// Retrieves the mutable data buffer as a contiguous slice.
    ///
    /// Matrix components may not be contiguous, depending on its strides.    
    ///
    /// # Safety
    /// The matrix components may not be stored in a contiguous way, depending on the strides.
    /// This method is unsafe because this can yield to invalid aliasing when called on some pairs
    /// of matrix slices originating from the same matrix with strides.
    unsafe fn as_mut_slice_unchecked(&mut self) -> &mut [T];
}

/// A matrix storage that is stored contiguously in memory.
///
/// The storage requirement means that for any value of `i` in `[0, nrows * ncols - 1]`, the value
/// `.get_unchecked_linear` returns one of the matrix component. This trait is unsafe because
/// failing to comply to this may cause Undefined Behaviors.
pub unsafe trait ContiguousStorage<T, R: Dim, C: Dim = U1>: Storage<T, R, C> {
    /// Converts this data storage to a contiguous slice.
    fn as_slice(&self) -> &[T] {
        // SAFETY: this is safe because this trait guarantees the fact
        //         that the data is stored contiguously.
        unsafe { self.as_slice_unchecked() }
    }
}

/// A mutable matrix storage that is stored contiguously in memory.
///
/// The storage requirement means that for any value of `i` in `[0, nrows * ncols - 1]`, the value
/// `.get_unchecked_linear` returns one of the matrix component. This trait is unsafe because
/// failing to comply to this may cause Undefined Behaviors.
pub unsafe trait ContiguousStorageMut<T, R: Dim, C: Dim = U1>:
    ContiguousStorage<T, R, C> + StorageMut<T, R, C>
{
    /// Converts this data storage to a contiguous mutable slice.
    fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: this is safe because this trait guarantees the fact
        //         that the data is stored contiguously.
        unsafe { self.as_mut_slice_unchecked() }
    }
}

/// A matrix storage that can be reshaped in-place.
pub trait ReshapableStorage<T, R1: Dim, C1: Dim, R2: Dim, C2: Dim>: Storage<T, R1, C1> {
    /// The reshaped storage type.
    type Output: Storage<T, R2, C2>;

    /// Reshapes the storage into the output storage type.
    fn reshape_generic(self, nrows: R2, ncols: C2) -> Self::Output;
}
