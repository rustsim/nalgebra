use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo};
use std::slice;

use crate::base::allocator::{Allocator, InnerAllocator};
use crate::base::default_allocator::DefaultAllocator;
use crate::base::dimension::{Const, Dim, DimName, Dynamic, IsNotStaticOne, U1};
use crate::base::iter::MatrixIter;
use crate::base::storage::{ContiguousStorage, ContiguousStorageMut, Storage, StorageMut};
use crate::base::{Matrix, Owned};

macro_rules! slice_storage_impl(
    ($doc: expr; $Storage: ident as $SRef: ty; $T: ident.$get_addr: ident ($Ptr: ty as $Ref: ty)) => {
        #[doc = $doc]
        #[derive(Debug)]
        pub struct $T<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> {
            ptr:       $Ptr,
            shape:     (R, C),
            strides:   (RStride, CStride),
            _phantoms: PhantomData<$Ref>,
        }

        unsafe impl<'a, T: Send, R: Dim, C: Dim, RStride: Dim, CStride: Dim> Send
            for $T<'a, T, R, C, RStride, CStride>
        {}

        unsafe impl<'a, T: Sync, R: Dim, C: Dim, RStride: Dim, CStride: Dim> Sync
            for $T<'a, T, R, C, RStride, CStride>
        {}

        impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> $T<'a, T, R, C, RStride, CStride> {
            /// Create a new matrix slice without bound checking and from a raw pointer.
            #[inline]
            pub unsafe fn from_raw_parts(ptr:     $Ptr,
                                         shape:   (R, C),
                                         strides: (RStride, CStride))
                                        -> Self
                where RStride: Dim,
                      CStride: Dim {

                $T {
                    ptr,
                    shape,
                    strides,
                    _phantoms: PhantomData
                }
            }
        }

        // Dynamic is arbitrary. It's just to be able to call the constructors with `Slice::`
        impl<'a, T, R: Dim, C: Dim> $T<'a, T, R, C, Dynamic, Dynamic> {
            /// Create a new matrix slice without bound checking.
            #[inline]
            pub unsafe fn new_unchecked<RStor, CStor, S>(storage: $SRef, start: (usize, usize), shape: (R, C))
                -> $T<'a, T, R, C, S::RStride, S::CStride>
                where RStor: Dim,
                      CStor: Dim,
                      S:     $Storage<T, RStor, CStor> {

                let strides = storage.strides();
                $T::new_with_strides_unchecked(storage, start, shape, strides)
            }

            /// Create a new matrix slice without bound checking.
            #[inline]
            pub unsafe fn new_with_strides_unchecked<S, RStor, CStor, RStride, CStride>(storage: $SRef,
                                                                                        start:   (usize, usize),
                                                                                        shape:   (R, C),
                                                                                        strides: (RStride, CStride))
                -> $T<'a, T, R, C, RStride, CStride>
                where RStor: Dim,
                      CStor: Dim,
                      S: $Storage<T, RStor, CStor>,
                      RStride: Dim,
                      CStride: Dim {

                $T::from_raw_parts(storage.$get_addr(start.0, start.1), shape, strides)
            }
        }

        impl <'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim>
            $T<'a, T, R, C, RStride, CStride>
        where
            Self: ContiguousStorage<T, R, C>
        {
            /// Extracts the original slice from this storage
            pub fn into_slice(self) -> &'a [T] {
                let (nrows, ncols) = self.shape();
                if nrows.value() != 0 && ncols.value() != 0 {
                    let sz = self.linear_index(nrows.value() - 1, ncols.value() - 1);
                    unsafe { slice::from_raw_parts(self.ptr, sz + 1) }
                } else {
                    unsafe { slice::from_raw_parts(self.ptr, 0) }
                }
            }
        }
    }
);

slice_storage_impl!("A matrix data storage for a matrix slice. Only contains an internal reference \
                     to another matrix data storage.";
    Storage as &'a S; SliceStorage.get_address_unchecked(*const T as &'a T));

slice_storage_impl!("A mutable matrix data storage for mutable matrix slice. Only contains an \
                     internal mutable reference to another matrix data storage.";
    StorageMut as &'a mut S; SliceStorageMut.get_address_unchecked_mut(*mut T as &'a mut T)
);

impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> Copy
    for SliceStorage<'a, T, R, C, RStride, CStride>
{
}

impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> Clone
    for SliceStorage<'a, T, R, C, RStride, CStride>
{
    #[inline]
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            shape: self.shape,
            strides: self.strides,
            _phantoms: PhantomData,
        }
    }
}

impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim>
    SliceStorageMut<'a, T, R, C, RStride, CStride>
where
    Self: ContiguousStorageMut<T, R, C>,
{
    /// Extracts the original slice from this storage
    pub fn into_slice_mut(self) -> &'a mut [T] {
        let (nrows, ncols) = self.shape();
        if nrows.value() != 0 && ncols.value() != 0 {
            let sz = self.linear_index(nrows.value() - 1, ncols.value() - 1);
            unsafe { slice::from_raw_parts_mut(self.ptr, sz + 1) }
        } else {
            unsafe { slice::from_raw_parts_mut(self.ptr, 0) }
        }
    }
}

macro_rules! storage_impl(
    ($($T: ident),* $(,)*) => {$(
        unsafe impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> Storage<T, R, C>
            for $T<'a, T, R, C, RStride, CStride> {

            type RStride = RStride;
            type CStride = CStride;

            #[inline]
            fn ptr(&self) -> *const T {
                self.ptr
            }

            #[inline]
            fn shape(&self) -> (R, C) {
                self.shape
            }

            #[inline]
            fn strides(&self) -> (Self::RStride, Self::CStride) {
                self.strides
            }

            #[inline]
            fn is_contiguous(&self) -> bool {
                // Common cases that can be deduced at compile-time even if one of the dimensions
                // is Dynamic.
                if (RStride::is::<U1>() && C::is::<U1>()) || // Column vector.
                   (CStride::is::<U1>() && R::is::<U1>()) {  // Row vector.
                    true
                }
                else {
                    let (nrows, _)     = self.shape();
                    let (srows, scols) = self.strides();

                    srows.value() == 1 && scols.value() == nrows.value()
                }
            }

            #[inline]
            fn into_owned(self) -> Owned<T, R, C>
            where
                T: Clone,
                DefaultAllocator: Allocator<T, R, C>
            {
                self.clone_owned()
            }

            #[inline]
            fn clone_owned(&self) -> Owned<T, R, C>
            where
                T: Clone,
                DefaultAllocator: Allocator<T, R, C>
            {
                let (nrows, ncols) = self.shape();
                let it = MatrixIter::new(self).cloned();
             Owned(   DefaultAllocator::allocate_from_iterator(nrows, ncols, it))
            }

            #[inline]
            unsafe fn as_slice_unchecked(&self) -> &[T] {
                let (nrows, ncols) = self.shape();
                if nrows.value() != 0 && ncols.value() != 0 {
                    let sz = self.linear_index(nrows.value() - 1, ncols.value() - 1);
                    slice::from_raw_parts(self.ptr, sz + 1)
                }
                else {
                    slice::from_raw_parts(self.ptr, 0)
                }
            }
        }
    )*}
);

storage_impl!(SliceStorage, SliceStorageMut);

impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim>
    SliceStorage<'a, MaybeUninit<T>, R, C, RStride, CStride>
{
    /// Assumes a slice storage's entries to be initialized. This operation
    /// should be near zero-cost.
    ///
    /// # Safety
    /// All of the slice storage's entries must be initialized, otherwise
    /// Undefined Behavior will be triggered.
    pub unsafe fn assume_init(self) -> SliceStorage<'a, T, R, C, RStride, CStride> {
        SliceStorage::from_raw_parts(self.ptr as *const T, self.shape, self.strides)
    }
}

impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim>
    SliceStorageMut<'a, MaybeUninit<T>, R, C, RStride, CStride>
{
    /// Assumes a slice storage's entries to be initialized. This operation should be near zero-cost.
    ///
    /// # Safety
    /// The user must make sure that every single entry of the buffer has been initialized,
    /// or Undefined Behavior will immediately occur.    
    pub unsafe fn assume_init(self) -> SliceStorageMut<'a, T, R, C, RStride, CStride> {
        SliceStorageMut::from_raw_parts(self.ptr as *mut T, self.shape, self.strides)
    }
}

unsafe impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> StorageMut<T, R, C>
    for SliceStorageMut<'a, T, R, C, RStride, CStride>
{
    #[inline]
    fn ptr_mut(&mut self) -> *mut T {
        self.ptr
    }

    #[inline]
    unsafe fn as_mut_slice_unchecked(&mut self) -> &mut [T] {
        let (nrows, ncols) = self.shape();
        if nrows.value() != 0 && ncols.value() != 0 {
            let sz = self.linear_index(nrows.value() - 1, ncols.value() - 1);
            slice::from_raw_parts_mut(self.ptr, sz + 1)
        } else {
            slice::from_raw_parts_mut(self.ptr, 0)
        }
    }
}

unsafe impl<'a, T, R: Dim, CStride: Dim> ContiguousStorage<T, R, U1>
    for SliceStorage<'a, T, R, U1, U1, CStride>
{
}

unsafe impl<'a, T, R: Dim, CStride: Dim> ContiguousStorage<T, R, U1>
    for SliceStorageMut<'a, T, R, U1, U1, CStride>
{
}

unsafe impl<'a, T, R: Dim, CStride: Dim> ContiguousStorageMut<T, R, U1>
    for SliceStorageMut<'a, T, R, U1, U1, CStride>
{
}

unsafe impl<'a, T, R: DimName, C: Dim + IsNotStaticOne> ContiguousStorage<T, R, C>
    for SliceStorage<'a, T, R, C, U1, R>
{
}

unsafe impl<'a, T, R: DimName, C: Dim + IsNotStaticOne> ContiguousStorage<T, R, C>
    for SliceStorageMut<'a, T, R, C, U1, R>
{
}

unsafe impl<'a, T, R: DimName, C: Dim + IsNotStaticOne> ContiguousStorageMut<T, R, C>
    for SliceStorageMut<'a, T, R, C, U1, R>
{
}

impl<T, R: Dim, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S> {
    #[inline]
    fn assert_slice_index(
        &self,
        start: (usize, usize),
        shape: (usize, usize),
        steps: (usize, usize),
    ) {
        let my_shape = self.shape();
        // NOTE: we don't do any subtraction to avoid underflow for zero-sized matrices.
        //
        // Terms that would have been negative are moved to the other side of the inequality
        // instead.
        assert!(
            start.0 + (steps.0 + 1) * shape.0 <= my_shape.0 + steps.0,
            "Matrix slicing out of bounds."
        );
        assert!(
            start.1 + (steps.1 + 1) * shape.1 <= my_shape.1 + steps.1,
            "Matrix slicing out of bounds."
        );
    }
}

macro_rules! matrix_slice_impl(
    ($me: ident: $Me: ty, $MatrixSlice: ident, $SliceStorage: ident, $Storage: ident.$get_addr: ident (), $data: expr;
     $row: ident,
     $row_part: ident,
     $rows: ident,
     $rows_with_step: ident,
     $fixed_rows: ident,
     $fixed_rows_with_step: ident,
     $rows_generic: ident,
     $rows_generic_with_step: ident,
     $column: ident,
     $column_part: ident,
     $columns: ident,
     $columns_with_step: ident,
     $fixed_columns: ident,
     $fixed_columns_with_step: ident,
     $columns_generic: ident,
     $columns_generic_with_step: ident,
     $slice: ident,
     $slice_with_steps: ident,
     $fixed_slice: ident,
     $fixed_slice_with_steps: ident,
     $generic_slice: ident,
     $generic_slice_with_steps: ident,
     $full_slice: ident,
     $rows_range_pair: ident,
     $columns_range_pair: ident) => {
        /*
         *
         * Row slicing.
         *
         */
        /// Returns a slice containing the i-th row of this matrix.
        #[inline]
        pub fn $row($me: $Me, i: usize) -> $MatrixSlice<T, U1, C, S::RStride, S::CStride> {
            $me.$fixed_rows::<1>(i)
        }

        /// Returns a slice containing the `n` first elements of the i-th row of this matrix.
        #[inline]
        pub fn $row_part($me: $Me, i: usize, n: usize) -> $MatrixSlice<T, U1, Dynamic, S::RStride, S::CStride> {
            $me.$generic_slice((i, 0), (Const::<1>, Dynamic::new(n)))
        }

        /// Extracts from this matrix a set of consecutive rows.
        #[inline]
        pub fn $rows($me: $Me, first_row: usize, nrows: usize)
            -> $MatrixSlice<T, Dynamic, C, S::RStride, S::CStride> {

            $me.$rows_generic(first_row, Dynamic::new(nrows))
        }

        /// Extracts from this matrix a set of consecutive rows regularly skipping `step` rows.
        #[inline]
        pub fn $rows_with_step($me: $Me, first_row: usize, nrows: usize, step: usize)
            -> $MatrixSlice<T, Dynamic, C, Dynamic, S::CStride> {

            $me.$rows_generic_with_step(first_row, Dynamic::new(nrows), step)
        }

        /// Extracts a compile-time number of consecutive rows from this matrix.
        #[inline]
        pub fn $fixed_rows<const RSLICE: usize>($me: $Me, first_row: usize)
            -> $MatrixSlice<T, Const<RSLICE>, C, S::RStride, S::CStride> {

            $me.$rows_generic(first_row, Const::<RSLICE>)
        }

        /// Extracts from this matrix a compile-time number of rows regularly skipping `step`
        /// rows.
        #[inline]
        pub fn $fixed_rows_with_step<const RSLICE: usize>($me: $Me, first_row: usize, step: usize)
            -> $MatrixSlice<T, Const<RSLICE>, C, Dynamic, S::CStride> {

            $me.$rows_generic_with_step(first_row, Const::<RSLICE>, step)
        }

        /// Extracts from this matrix `nrows` rows regularly skipping `step` rows. Both
        /// argument may or may not be values known at compile-time.
        #[inline]
        pub fn $rows_generic<RSlice: Dim>($me: $Me, row_start: usize, nrows: RSlice)
            -> $MatrixSlice<T, RSlice, C, S::RStride, S::CStride> {

            let my_shape  = $me.data.shape();
            $me.assert_slice_index((row_start, 0), (nrows.value(), my_shape.1.value()), (0, 0));

            let shape = (nrows, my_shape.1);

            unsafe {
                let data = $SliceStorage::new_unchecked($data, (row_start, 0), shape);
                Matrix::from_data(data)
            }
        }

        /// Extracts from this matrix `nrows` rows regularly skipping `step` rows. Both
        /// argument may or may not be values known at compile-time.
        #[inline]
        pub fn $rows_generic_with_step<RSlice>($me: $Me, row_start: usize, nrows: RSlice, step: usize)
            -> $MatrixSlice<T, RSlice, C, Dynamic, S::CStride>
            where RSlice: Dim {

            let my_shape = $me.data.shape();
            let my_strides = $me.data.strides();
            $me.assert_slice_index((row_start, 0), (nrows.value(), my_shape.1.value()), (step, 0));

            let strides = (Dynamic::new((step + 1) * my_strides.0.value()), my_strides.1);
            let shape = (nrows, my_shape.1);

            unsafe {
                let data = $SliceStorage::new_with_strides_unchecked($data, (row_start, 0), shape, strides);
                Matrix::from_data(data)
            }
        }

        /*
         *
         * Column slicing.
         *
         */
        /// Returns a slice containing the i-th column of this matrix.
        #[inline]
        pub fn $column($me: $Me, i: usize) -> $MatrixSlice<T, R, U1, S::RStride, S::CStride> {
            $me.$fixed_columns::<1>(i)
        }

        /// Returns a slice containing the `n` first elements of the i-th column of this matrix.
        #[inline]
        pub fn $column_part($me: $Me, i: usize, n: usize) -> $MatrixSlice<T, Dynamic, U1, S::RStride, S::CStride> {
            $me.$generic_slice((0, i), (Dynamic::new(n), Const::<1>))
        }

        /// Extracts from this matrix a set of consecutive columns.
        #[inline]
        pub fn $columns($me: $Me, first_col: usize, ncols: usize)
            -> $MatrixSlice<T, R, Dynamic, S::RStride, S::CStride> {

            $me.$columns_generic(first_col, Dynamic::new(ncols))
        }

        /// Extracts from this matrix a set of consecutive columns regularly skipping `step`
        /// columns.
        #[inline]
        pub fn $columns_with_step($me: $Me, first_col: usize, ncols: usize, step: usize)
            -> $MatrixSlice<T, R, Dynamic, S::RStride, Dynamic> {

            $me.$columns_generic_with_step(first_col, Dynamic::new(ncols), step)
        }

        /// Extracts a compile-time number of consecutive columns from this matrix.
        #[inline]
        pub fn $fixed_columns<const CSLICE: usize>($me: $Me, first_col: usize)
            -> $MatrixSlice<T, R, Const<CSLICE>, S::RStride, S::CStride> {

            $me.$columns_generic(first_col, Const::<CSLICE>)
        }

        /// Extracts from this matrix a compile-time number of columns regularly skipping
        /// `step` columns.
        #[inline]
        pub fn $fixed_columns_with_step<const CSLICE: usize>($me: $Me, first_col: usize, step: usize)
            -> $MatrixSlice<T, R, Const<CSLICE>, S::RStride, Dynamic> {

            $me.$columns_generic_with_step(first_col, Const::<CSLICE>, step)
        }

        /// Extracts from this matrix `ncols` columns. The number of columns may or may not be
        /// known at compile-time.
        #[inline]
        pub fn $columns_generic<CSlice: Dim>($me: $Me, first_col: usize, ncols: CSlice)
            -> $MatrixSlice<T, R, CSlice, S::RStride, S::CStride> {

            let my_shape = $me.data.shape();
            $me.assert_slice_index((0, first_col), (my_shape.0.value(), ncols.value()), (0, 0));
            let shape = (my_shape.0, ncols);

            unsafe {
                let data = $SliceStorage::new_unchecked($data, (0, first_col), shape);
                Matrix::from_data(data)
            }
        }

        /// Extracts from this matrix `ncols` columns skipping `step` columns. Both argument may
        /// or may not be values known at compile-time.
        #[inline]
        pub fn $columns_generic_with_step<CSlice: Dim>($me: $Me, first_col: usize, ncols: CSlice, step: usize)
            -> $MatrixSlice<T, R, CSlice, S::RStride, Dynamic> {

            let my_shape = $me.data.shape();
            let my_strides = $me.data.strides();

            $me.assert_slice_index((0, first_col), (my_shape.0.value(), ncols.value()), (0, step));

            let strides = (my_strides.0, Dynamic::new((step + 1) * my_strides.1.value()));
            let shape = (my_shape.0, ncols);

            unsafe {
                let data = $SliceStorage::new_with_strides_unchecked($data, (0, first_col), shape, strides);
                Matrix::from_data(data)
            }
        }

        /*
         *
         * General slicing.
         *
         */
        /// Slices this matrix starting at its component `(irow, icol)` and with `(nrows, ncols)`
        /// consecutive elements.
        #[inline]
        pub fn $slice($me: $Me, start: (usize, usize), shape: (usize, usize))
            -> $MatrixSlice<T, Dynamic, Dynamic, S::RStride, S::CStride> {

            $me.assert_slice_index(start, shape, (0, 0));
            let shape = (Dynamic::new(shape.0), Dynamic::new(shape.1));

            unsafe {
                let data = $SliceStorage::new_unchecked($data, start, shape);
                Matrix::from_data(data)
            }
        }

        /// Slices this matrix starting at its component `(start.0, start.1)` and with
        /// `(shape.0, shape.1)` components. Each row (resp. column) of the sliced matrix is
        /// separated by `steps.0` (resp. `steps.1`) ignored rows (resp. columns) of the
        /// original matrix.
        #[inline]
        pub fn $slice_with_steps($me: $Me, start: (usize, usize), shape: (usize, usize), steps: (usize, usize))
            -> $MatrixSlice<T, Dynamic, Dynamic, Dynamic, Dynamic> {
            let shape = (Dynamic::new(shape.0), Dynamic::new(shape.1));

            $me.$generic_slice_with_steps(start, shape, steps)
        }

        /// Slices this matrix starting at its component `(irow, icol)` and with `(R::dim(),
        /// CSlice::dim())` consecutive components.
        #[inline]
        pub fn $fixed_slice<const RSLICE: usize, const CSLICE: usize>($me: $Me, irow: usize, icol: usize)
            -> $MatrixSlice<T, Const<RSLICE>, Const<CSLICE>, S::RStride, S::CStride> {

            $me.assert_slice_index((irow, icol), (RSLICE, CSLICE), (0, 0));
            let shape = (Const::<RSLICE>, Const::<CSLICE>);

            unsafe {
                let data = $SliceStorage::new_unchecked($data, (irow, icol), shape);
                Matrix::from_data(data)
            }
        }

        /// Slices this matrix starting at its component `(start.0, start.1)` and with
        /// `(RSLICE, CSLICE)` components. Each row (resp. column) of the sliced
        /// matrix is separated by `steps.0` (resp. `steps.1`) ignored rows (resp. columns) of
        /// the original matrix.
        #[inline]
        pub fn $fixed_slice_with_steps<const RSLICE: usize, const CSLICE: usize>($me: $Me, start: (usize, usize), steps: (usize, usize))
            -> $MatrixSlice<T, Const<RSLICE>, Const<CSLICE>, Dynamic, Dynamic> {
            let shape = (Const::<RSLICE>, Const::<CSLICE>);
            $me.$generic_slice_with_steps(start, shape, steps)
        }

        /// Creates a slice that may or may not have a fixed size and stride.
        #[inline]
        pub fn $generic_slice<RSlice: Dim, CSlice: Dim>($me: $Me, start: (usize, usize), shape: (RSlice, CSlice))
            -> $MatrixSlice<T, RSlice, CSlice, S::RStride, S::CStride>
        {
            $me.assert_slice_index(start, (shape.0.value(), shape.1.value()), (0, 0));

            unsafe {
                let data = $SliceStorage::new_unchecked($data, start, shape);
                Matrix::from_data(data)
            }
        }

        /// Creates a slice that may or may not have a fixed size and stride.
        #[inline]
        pub fn $generic_slice_with_steps<RSlice, CSlice>($me: $Me,
                                                         start: (usize, usize),
                                                         shape: (RSlice, CSlice),
                                                         steps: (usize, usize))
            -> $MatrixSlice<T, RSlice, CSlice, Dynamic, Dynamic>
            where RSlice: Dim,
                  CSlice: Dim {

            $me.assert_slice_index(start, (shape.0.value(), shape.1.value()), steps);

            let my_strides = $me.data.strides();
            let strides    = (Dynamic::new((steps.0 + 1) * my_strides.0.value()),
                              Dynamic::new((steps.1 + 1) * my_strides.1.value()));

            unsafe {
                let data = $SliceStorage::new_with_strides_unchecked($data, start, shape, strides);
                Matrix::from_data(data)
            }
        }

        /// Returns a slice containing the entire matrix.
        pub fn $full_slice($me: $Me) -> $MatrixSlice<T, R, C, S::RStride, S::CStride> {
            let (nrows, ncols) = $me.shape();
            $me.$generic_slice((0, 0), (R::from_usize(nrows), C::from_usize(ncols)))
        }

        /*
         *
         * Splitting.
         *
         */
        /// Splits this NxM matrix into two parts delimited by two ranges.
        ///
        /// Panics if the ranges overlap or if the first range is empty.
        #[inline]
        pub fn $rows_range_pair<Range1: SliceRange<R>, Range2: SliceRange<R>>($me: $Me, r1: Range1, r2: Range2)
            -> ($MatrixSlice<T, Range1::Size, C, S::RStride, S::CStride>,
                $MatrixSlice<T, Range2::Size, C, S::RStride, S::CStride>) {

            let (nrows, ncols) = $me.data.shape();
            let strides        = $me.data.strides();

            let start1 = r1.begin(nrows);
            let start2 = r2.begin(nrows);

            let end1 = r1.end(nrows);
            let end2 = r2.end(nrows);

            let nrows1 = r1.size(nrows);
            let nrows2 = r2.size(nrows);

            assert!(start2 >= end1 || start1 >= end2, "Rows range pair: the slice ranges must not overlap.");
            assert!(end2 <= nrows.value(), "Rows range pair: index out of range.");

            unsafe {
                let ptr1 = $data.$get_addr(start1, 0);
                let ptr2 = $data.$get_addr(start2, 0);

                let data1  = $SliceStorage::from_raw_parts(ptr1, (nrows1, ncols), strides);
                let data2  = $SliceStorage::from_raw_parts(ptr2, (nrows2, ncols), strides);
                let slice1 = Matrix::from_data(data1);
                let slice2 = Matrix::from_data(data2);

                (slice1, slice2)
            }
        }

        /// Splits this NxM matrix into two parts delimited by two ranges.
        ///
        /// Panics if the ranges overlap or if the first range is empty.
        #[inline]
        pub fn $columns_range_pair<Range1: SliceRange<C>, Range2: SliceRange<C>>($me: $Me, r1: Range1, r2: Range2)
            -> ($MatrixSlice<T, R, Range1::Size, S::RStride, S::CStride>,
                $MatrixSlice<T, R, Range2::Size, S::RStride, S::CStride>) {

            let (nrows, ncols) = $me.data.shape();
            let strides        = $me.data.strides();

            let start1 = r1.begin(ncols);
            let start2 = r2.begin(ncols);

            let end1 = r1.end(ncols);
            let end2 = r2.end(ncols);

            let ncols1 = r1.size(ncols);
            let ncols2 = r2.size(ncols);

            assert!(start2 >= end1 || start1 >= end2, "Columns range pair: the slice ranges must not overlap.");
            assert!(end2 <= ncols.value(), "Columns range pair: index out of range.");

            unsafe {
                let ptr1 = $data.$get_addr(0, start1);
                let ptr2 = $data.$get_addr(0, start2);

                let data1  = $SliceStorage::from_raw_parts(ptr1, (nrows, ncols1), strides);
                let data2  = $SliceStorage::from_raw_parts(ptr2, (nrows, ncols2), strides);
                let slice1 = Matrix::from_data(data1);
                let slice2 = Matrix::from_data(data2);

                (slice1, slice2)
            }
        }
    }
);

/// A matrix slice.
pub type MatrixSlice<'a, T, R, C, RStride = U1, CStride = R> =
    Matrix<T, R, C, SliceStorage<'a, T, R, C, RStride, CStride>>;
/// A mutable matrix slice.
pub type MatrixSliceMut<'a, T, R, C, RStride = U1, CStride = R> =
    Matrix<T, R, C, SliceStorageMut<'a, T, R, C, RStride, CStride>>;

/// # Slicing based on index and length
impl<T, R: Dim, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S> {
    matrix_slice_impl!(
     self: &Self, MatrixSlice, SliceStorage, Storage.get_address_unchecked(), &self.data;
     row,
     row_part,
     rows,
     rows_with_step,
     fixed_rows,
     fixed_rows_with_step,
     rows_generic,
     rows_generic_with_step,
     column,
     column_part,
     columns,
     columns_with_step,
     fixed_columns,
     fixed_columns_with_step,
     columns_generic,
     columns_generic_with_step,
     slice,
     slice_with_steps,
     fixed_slice,
     fixed_slice_with_steps,
     generic_slice,
     generic_slice_with_steps,
     full_slice,
     rows_range_pair,
     columns_range_pair);
}

/// # Mutable slicing based on index and length
impl<T, R: Dim, C: Dim, S: StorageMut<T, R, C>> Matrix<T, R, C, S> {
    matrix_slice_impl!(
     self: &mut Self, MatrixSliceMut, SliceStorageMut, StorageMut.get_address_unchecked_mut(), &mut self.data;
     row_mut,
     row_part_mut,
     rows_mut,
     rows_with_step_mut,
     fixed_rows_mut,
     fixed_rows_with_step_mut,
     rows_generic_mut,
     rows_generic_with_step_mut,
     column_mut,
     column_part_mut,
     columns_mut,
     columns_with_step_mut,
     fixed_columns_mut,
     fixed_columns_with_step_mut,
     columns_generic_mut,
     columns_generic_with_step_mut,
     slice_mut,
     slice_with_steps_mut,
     fixed_slice_mut,
     fixed_slice_with_steps_mut,
     generic_slice_mut,
     generic_slice_with_steps_mut,
     full_slice_mut,
     rows_range_pair_mut,
     columns_range_pair_mut);
}

impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim>
    MatrixSlice<'a, MaybeUninit<T>, R, C, RStride, CStride>
{
    /// Assumes a matrix slices's entries to be initialized. This operation should be near zero-cost.
    pub unsafe fn slice_assume_init(self) -> MatrixSlice<'a, T, R, C, RStride, CStride> {
        Matrix::from_data(self.data.assume_init())
    }
}

impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim>
    MatrixSliceMut<'a, MaybeUninit<T>, R, C, RStride, CStride>
{
    /// Assumes a matrix slices's entries to be initialized. This operation should be near zero-cost.
    pub unsafe fn slice_assume_init(self) -> MatrixSliceMut<'a, T, R, C, RStride, CStride> {
        Matrix::from_data(self.data.assume_init())
    }
}

/// A range with a size that may be known at compile-time.
///
/// This may be:
/// * A single `usize` index, e.g., `4`
/// * A left-open range `std::ops::RangeTo`, e.g., `.. 4`
/// * A right-open range `std::ops::RangeFrom`, e.g., `4 ..`
/// * A full range `std::ops::RangeFull`, e.g., `..`
pub trait SliceRange<D: Dim> {
    /// Type of the range size. May be a type-level integer.
    type Size: Dim;

    /// The start index of the range.
    fn begin(&self, shape: D) -> usize;
    // NOTE: this is the index immediately after the last index.
    /// The index immediately after the last index inside of the range.
    fn end(&self, shape: D) -> usize;
    /// The number of elements of the range, i.e., `self.end - self.begin`.
    fn size(&self, shape: D) -> Self::Size;
}

impl<D: Dim> SliceRange<D> for usize {
    type Size = U1;

    #[inline(always)]
    fn begin(&self, _: D) -> usize {
        *self
    }

    #[inline(always)]
    fn end(&self, _: D) -> usize {
        *self + 1
    }

    #[inline(always)]
    fn size(&self, _: D) -> Self::Size {
        Const::<1>
    }
}

impl<D: Dim> SliceRange<D> for Range<usize> {
    type Size = Dynamic;

    #[inline(always)]
    fn begin(&self, _: D) -> usize {
        self.start
    }

    #[inline(always)]
    fn end(&self, _: D) -> usize {
        self.end
    }

    #[inline(always)]
    fn size(&self, _: D) -> Self::Size {
        Dynamic::new(self.end - self.start)
    }
}

impl<D: Dim> SliceRange<D> for RangeFrom<usize> {
    type Size = Dynamic;

    #[inline(always)]
    fn begin(&self, _: D) -> usize {
        self.start
    }

    #[inline(always)]
    fn end(&self, dim: D) -> usize {
        dim.value()
    }

    #[inline(always)]
    fn size(&self, dim: D) -> Self::Size {
        Dynamic::new(dim.value() - self.start)
    }
}

impl<D: Dim> SliceRange<D> for RangeTo<usize> {
    type Size = Dynamic;

    #[inline(always)]
    fn begin(&self, _: D) -> usize {
        0
    }

    #[inline(always)]
    fn end(&self, _: D) -> usize {
        self.end
    }

    #[inline(always)]
    fn size(&self, _: D) -> Self::Size {
        Dynamic::new(self.end)
    }
}

impl<D: Dim> SliceRange<D> for RangeFull {
    type Size = D;

    #[inline(always)]
    fn begin(&self, _: D) -> usize {
        0
    }

    #[inline(always)]
    fn end(&self, dim: D) -> usize {
        dim.value()
    }

    #[inline(always)]
    fn size(&self, dim: D) -> Self::Size {
        dim
    }
}

impl<D: Dim> SliceRange<D> for RangeInclusive<usize> {
    type Size = Dynamic;

    #[inline(always)]
    fn begin(&self, _: D) -> usize {
        *self.start()
    }

    #[inline(always)]
    fn end(&self, _: D) -> usize {
        *self.end() + 1
    }

    #[inline(always)]
    fn size(&self, _: D) -> Self::Size {
        Dynamic::new(*self.end() + 1 - *self.start())
    }
}

// TODO: see how much of this overlaps with the general indexing
// methods from indexing.rs.
impl<T, R: Dim, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S> {
    /// Slices a sub-matrix containing the rows indexed by the range `rows` and the columns indexed
    /// by the range `cols`.
    #[inline]
    #[must_use]
    pub fn slice_range<RowRange, ColRange>(
        &self,
        rows: RowRange,
        cols: ColRange,
    ) -> MatrixSlice<T, RowRange::Size, ColRange::Size, S::RStride, S::CStride>
    where
        RowRange: SliceRange<R>,
        ColRange: SliceRange<C>,
    {
        let (nrows, ncols) = self.data.shape();
        self.generic_slice(
            (rows.begin(nrows), cols.begin(ncols)),
            (rows.size(nrows), cols.size(ncols)),
        )
    }

    /// Slice containing all the rows indexed by the range `rows`.
    #[inline]
    #[must_use]
    pub fn rows_range<RowRange: SliceRange<R>>(
        &self,
        rows: RowRange,
    ) -> MatrixSlice<T, RowRange::Size, C, S::RStride, S::CStride> {
        self.slice_range(rows, ..)
    }

    /// Slice containing all the columns indexed by the range `rows`.
    #[inline]
    #[must_use]
    pub fn columns_range<ColRange: SliceRange<C>>(
        &self,
        cols: ColRange,
    ) -> MatrixSlice<T, R, ColRange::Size, S::RStride, S::CStride> {
        self.slice_range(.., cols)
    }
}

// TODO: see how much of this overlaps with the general indexing
// methods from indexing.rs.
impl<T, R: Dim, C: Dim, S: StorageMut<T, R, C>> Matrix<T, R, C, S> {
    /// Slices a mutable sub-matrix containing the rows indexed by the range `rows` and the columns
    /// indexed by the range `cols`.
    pub fn slice_range_mut<RowRange, ColRange>(
        &mut self,
        rows: RowRange,
        cols: ColRange,
    ) -> MatrixSliceMut<T, RowRange::Size, ColRange::Size, S::RStride, S::CStride>
    where
        RowRange: SliceRange<R>,
        ColRange: SliceRange<C>,
    {
        let (nrows, ncols) = self.data.shape();
        self.generic_slice_mut(
            (rows.begin(nrows), cols.begin(ncols)),
            (rows.size(nrows), cols.size(ncols)),
        )
    }

    /// Slice containing all the rows indexed by the range `rows`.
    #[inline]
    pub fn rows_range_mut<RowRange: SliceRange<R>>(
        &mut self,
        rows: RowRange,
    ) -> MatrixSliceMut<T, RowRange::Size, C, S::RStride, S::CStride> {
        self.slice_range_mut(rows, ..)
    }

    /// Slice containing all the columns indexed by the range `cols`.
    #[inline]
    pub fn columns_range_mut<ColRange: SliceRange<C>>(
        &mut self,
        cols: ColRange,
    ) -> MatrixSliceMut<T, R, ColRange::Size, S::RStride, S::CStride> {
        self.slice_range_mut(.., cols)
    }
}

impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim>
    From<MatrixSliceMut<'a, T, R, C, RStride, CStride>>
    for MatrixSlice<'a, T, R, C, RStride, CStride>
{
    fn from(slice_mut: MatrixSliceMut<'a, T, R, C, RStride, CStride>) -> Self {
        let data = SliceStorage {
            ptr: slice_mut.data.ptr,
            shape: slice_mut.data.shape,
            strides: slice_mut.data.strides,
            _phantoms: PhantomData,
        };

        Matrix::from_data(data)
    }
}
