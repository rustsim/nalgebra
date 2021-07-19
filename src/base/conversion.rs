use std::borrow::{Borrow, BorrowMut};
use std::convert::{AsMut, AsRef, From, Into};
use std::mem::{self, ManuallyDrop, MaybeUninit};

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;
use simba::scalar::{SubsetOf, SupersetOf};

use simba::simd::{PrimitiveSimdValue, SimdValue};

use crate::base::allocator::{Allocator, SameShapeAllocator};
use crate::base::constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
#[cfg(any(feature = "std", feature = "alloc"))]
use crate::base::dimension::Dynamic;
use crate::base::dimension::{
    Const, Dim, DimName, U1, U10, U11, U12, U13, U14, U15, U16, U2, U3, U4, U5, U6, U7, U8, U9,
};
use crate::base::iter::{MatrixIter, MatrixIterMut};
use crate::base::storage::{ContiguousStorage, ContiguousStorageMut, Storage, StorageMut};
use crate::base::{
    ArrayStorage, DVectorSlice, DVectorSliceMut, DefaultAllocator, Matrix, MatrixSlice,
    MatrixSliceMut, OMatrix, Scalar,
};
#[cfg(any(feature = "std", feature = "alloc"))]
use crate::base::{DVector, VecStorage};
use crate::base::{SliceStorage, SliceStorageMut};
use crate::constraint::DimEq;
use crate::{IsNotStaticOne, RowSVector, SMatrix, SVector};

// TODO: too bad this won't work for slice conversions.
impl<T1, T2, R1: Dim, C1: Dim, R2: Dim, C2: Dim> SubsetOf<OMatrix<T2, R2, C2>>
    for OMatrix<T1, R1, C1>
where
    T2: SupersetOf<T1>,
    DefaultAllocator:
        Allocator<T2, R2, C2> + Allocator<T1, R1, C1> + SameShapeAllocator<T1, R1, C1, R2, C2>,
    ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
{
    #[inline]
    fn to_superset(&self) -> OMatrix<T2, R2, C2> {
        let (nrows, ncols) = self.shape();
        let nrows2 = R2::from_usize(nrows);
        let ncols2 = C2::from_usize(ncols);

        let mut res = Matrix::new_uninitialized_generic(nrows2, ncols2);

        for i in 0..nrows {
            for j in 0..ncols {
                unsafe {
                    *res.get_unchecked_mut((i, j)) =
                        MaybeUninit::new(T2::from_subset(self.get_unchecked((i, j))));
                }
            }
        }

        // Safety: all entries have been initialized.
        unsafe { res.assume_init() }
    }

    #[inline]
    fn is_in_subset(m: &OMatrix<T2, R2, C2>) -> bool {
        m.iter().all(|e| e.is_in_subset())
    }

    #[inline]
    fn from_superset_unchecked(m: &OMatrix<T2, R2, C2>) -> Self {
        let (nrows2, ncols2) = m.shape();
        let nrows = R1::from_usize(nrows2);
        let ncols = C1::from_usize(ncols2);

        let mut res = OMatrix::new_uninitialized_generic(nrows, ncols);
        for i in 0..nrows2 {
            for j in 0..ncols2 {
                unsafe {
                    *res.get_unchecked_mut((i, j)) =
                        MaybeUninit::new(m.get_unchecked((i, j)).to_subset_unchecked());
                }
            }
        }

        // Safety: all entries have been initialized.
        unsafe { res.assume_init() }
    }
}

impl<'a, T, R: Dim, C: Dim, S: Storage<T, R, C>> IntoIterator for &'a Matrix<T, R, C, S> {
    type Item = &'a T;
    type IntoIter = MatrixIter<'a, T, R, C, S>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, R: Dim, C: Dim, S: StorageMut<T, R, C>> IntoIterator for &'a mut Matrix<T, R, C, S> {
    type Item = &'a mut T;
    type IntoIter = MatrixIterMut<'a, T, R, C, S>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, const D: usize> From<[T; D]> for SVector<T, D> {
    #[inline]
    fn from(arr: [T; D]) -> Self {
        Self::from_data(ArrayStorage([arr; 1]))
    }
}

impl<T, const D: usize> From<SVector<T, D>> for [T; D] {
    #[inline]
    fn from(vec: SVector<T, D>) -> Self {
        let data = ManuallyDrop::new(vec.data.0);
        // Safety: [[T; D]; 1] always has the same data layout as [T; D].
        let res = unsafe { (data.as_ptr() as *const [_; D]).read() };
        mem::forget(data);
        res
    }
}

impl<T: Clone, const D: usize> From<[T; D]> for RowSVector<T, D>
where
    Const<D>: IsNotStaticOne,
{
    #[inline]
    fn from(arr: [T; D]) -> Self {
        SVector::<T, D>::from(arr).transpose_into()
    }
}

impl<T: Clone, const D: usize> From<RowSVector<T, D>> for [T; D]
where
    Const<D>: IsNotStaticOne,
{
    #[inline]
    fn from(vec: RowSVector<T, D>) -> [T; D] {
        vec.transpose().into()
    }
}

macro_rules! impl_from_into_asref_1D(
    ($(($NRows: ident, $NCols: ident) => $SZ: expr);* $(;)*) => {$(
        impl<T, S> AsRef<[T; $SZ]> for Matrix<T, $NRows, $NCols, S>
        where
              S: ContiguousStorage<T, $NRows, $NCols> {
            #[inline]
            fn as_ref(&self) -> &[T; $SZ] {
                unsafe {
                    &*(self.data.ptr() as *const [T; $SZ])
                }
            }
        }

        impl<T, S> AsMut<[T; $SZ]> for Matrix<T, $NRows, $NCols, S>
        where
              S: ContiguousStorageMut<T, $NRows, $NCols> {
            #[inline]
            fn as_mut(&mut self) -> &mut [T; $SZ] {
                unsafe {
                    &mut *(self.data.ptr_mut() as *mut [T; $SZ])
                }
            }
        }
    )*}
);

// Implement for vectors of dimension 1 .. 16.
impl_from_into_asref_1D!(
    // Row vectors.
    (U1, U1 ) => 1;  (U1, U2 ) => 2;  (U1, U3 ) => 3;  (U1, U4 ) => 4;
    (U1, U5 ) => 5;  (U1, U6 ) => 6;  (U1, U7 ) => 7;  (U1, U8 ) => 8;
    (U1, U9 ) => 9;  (U1, U10) => 10; (U1, U11) => 11; (U1, U12) => 12;
    (U1, U13) => 13; (U1, U14) => 14; (U1, U15) => 15; (U1, U16) => 16;

    // Column vectors.
                     (U2 , U1) => 2;  (U3 , U1) => 3;  (U4 , U1) => 4;
    (U5 , U1) => 5;  (U6 , U1) => 6;  (U7 , U1) => 7;  (U8 , U1) => 8;
    (U9 , U1) => 9;  (U10, U1) => 10; (U11, U1) => 11; (U12, U1) => 12;
    (U13, U1) => 13; (U14, U1) => 14; (U15, U1) => 15; (U16, U1) => 16;
);

impl<T, const R: usize, const C: usize> From<[[T; R]; C]> for SMatrix<T, R, C> {
    #[inline]
    fn from(arr: [[T; R]; C]) -> Self {
        Self::from_data(ArrayStorage(arr))
    }
}

impl<T, const R: usize, const C: usize> From<SMatrix<T, R, C>> for [[T; R]; C] {
    #[inline]
    fn from(vec: SMatrix<T, R, C>) -> Self {
        vec.data.0
    }
}

macro_rules! impl_from_into_asref_borrow_2D(

    //does the impls on one case for either AsRef/AsMut and Borrow/BorrowMut
    (
        ($NRows: ty, $NCols: ty) => ($SZRows: expr, $SZCols: expr);
        $Ref:ident.$ref:ident(), $Mut:ident.$mut:ident()
    ) => {
        impl<T, S> $Ref<[[T; $SZRows]; $SZCols]> for Matrix<T, $NRows, $NCols, S>
        where S: ContiguousStorage<T, $NRows, $NCols> {
            #[inline]
            fn $ref(&self) -> &[[T; $SZRows]; $SZCols] {
                unsafe {
                    &*(self.data.ptr() as *const [[T; $SZRows]; $SZCols])
                }
            }
        }

        impl<T, S> $Mut<[[T; $SZRows]; $SZCols]> for Matrix<T, $NRows, $NCols, S>
        where S: ContiguousStorageMut<T, $NRows, $NCols> {
            #[inline]
            fn $mut(&mut self) -> &mut [[T; $SZRows]; $SZCols] {
                unsafe {
                    &mut *(self.data.ptr_mut() as *mut [[T; $SZRows]; $SZCols])
                }
            }
        }
    };

    //collects the mappings from typenum pairs to consts
    ($(($NRows: ty, $NCols: ty) => ($SZRows: expr, $SZCols: expr));* $(;)*) => {$(
        impl_from_into_asref_borrow_2D!(
            ($NRows, $NCols) => ($SZRows, $SZCols); AsRef.as_ref(), AsMut.as_mut()
        );
        impl_from_into_asref_borrow_2D!(
            ($NRows, $NCols) => ($SZRows, $SZCols); Borrow.borrow(), BorrowMut.borrow_mut()
        );
    )*}
);

// Implement for matrices with shape 2x2 .. 6x6.
impl_from_into_asref_borrow_2D!(
    (U2, U2) => (2, 2); (U2, U3) => (2, 3); (U2, U4) => (2, 4); (U2, U5) => (2, 5); (U2, U6) => (2, 6);
    (U3, U2) => (3, 2); (U3, U3) => (3, 3); (U3, U4) => (3, 4); (U3, U5) => (3, 5); (U3, U6) => (3, 6);
    (U4, U2) => (4, 2); (U4, U3) => (4, 3); (U4, U4) => (4, 4); (U4, U5) => (4, 5); (U4, U6) => (4, 6);
    (U5, U2) => (5, 2); (U5, U3) => (5, 3); (U5, U4) => (5, 4); (U5, U5) => (5, 5); (U5, U6) => (5, 6);
    (U6, U2) => (6, 2); (U6, U3) => (6, 3); (U6, U4) => (6, 4); (U6, U5) => (6, 5); (U6, U6) => (6, 6);
);

impl<'a, T: Clone, RStride: Dim, CStride: Dim, const R: usize, const C: usize>
    From<MatrixSlice<'a, T, Const<R>, Const<C>, RStride, CStride>>
    for Matrix<T, Const<R>, Const<C>, ArrayStorage<T, R, C>>
{
    fn from(matrix_slice: MatrixSlice<'a, T, Const<R>, Const<C>, RStride, CStride>) -> Self {
        matrix_slice.into_owned()
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<'a, T: Clone, C: Dim, RStride: Dim, CStride: Dim>
    From<MatrixSlice<'a, T, Dynamic, C, RStride, CStride>>
    for Matrix<T, Dynamic, C, VecStorage<T, Dynamic, C>>
{
    fn from(matrix_slice: MatrixSlice<'a, T, Dynamic, C, RStride, CStride>) -> Self {
        matrix_slice.into_owned()
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<'a, T: Clone, R: DimName, RStride: Dim, CStride: Dim>
    From<MatrixSlice<'a, T, R, Dynamic, RStride, CStride>>
    for Matrix<T, R, Dynamic, VecStorage<T, R, Dynamic>>
{
    fn from(matrix_slice: MatrixSlice<'a, T, R, Dynamic, RStride, CStride>) -> Self {
        matrix_slice.into_owned()
    }
}

impl<'a, T: Clone, RStride: Dim, CStride: Dim, const R: usize, const C: usize>
    From<MatrixSliceMut<'a, T, Const<R>, Const<C>, RStride, CStride>>
    for Matrix<T, Const<R>, Const<C>, ArrayStorage<T, R, C>>
{
    fn from(matrix_slice: MatrixSliceMut<'a, T, Const<R>, Const<C>, RStride, CStride>) -> Self {
        matrix_slice.into_owned()
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<'a, T: Clone, C: Dim, RStride: Dim, CStride: Dim>
    From<MatrixSliceMut<'a, T, Dynamic, C, RStride, CStride>>
    for Matrix<T, Dynamic, C, VecStorage<T, Dynamic, C>>
{
    fn from(matrix_slice: MatrixSliceMut<'a, T, Dynamic, C, RStride, CStride>) -> Self {
        matrix_slice.into_owned()
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<'a, T: Clone, R: DimName, RStride: Dim, CStride: Dim>
    From<MatrixSliceMut<'a, T, R, Dynamic, RStride, CStride>>
    for Matrix<T, R, Dynamic, VecStorage<T, R, Dynamic>>
{
    fn from(matrix_slice: MatrixSliceMut<'a, T, R, Dynamic, RStride, CStride>) -> Self {
        matrix_slice.into_owned()
    }
}

impl<'a, T, R: Dim, C: Dim, RSlice: Dim, CSlice: Dim, RStride: Dim, CStride: Dim, S>
    From<&'a Matrix<T, R, C, S>> for MatrixSlice<'a, T, RSlice, CSlice, RStride, CStride>
where
    S: Storage<T, R, C>,
    ShapeConstraint: DimEq<R, RSlice>
        + DimEq<C, CSlice>
        + DimEq<RStride, S::RStride>
        + DimEq<CStride, S::CStride>,
{
    fn from(m: &'a Matrix<T, R, C, S>) -> Self {
        let (row, col) = m.data.shape();
        let row_slice = RSlice::from_usize(row.value());
        let col_slice = CSlice::from_usize(col.value());

        let (rstride, cstride) = m.strides();

        let rstride_slice = RStride::from_usize(rstride);
        let cstride_slice = CStride::from_usize(cstride);

        unsafe {
            let data = SliceStorage::from_raw_parts(
                m.data.ptr(),
                (row_slice, col_slice),
                (rstride_slice, cstride_slice),
            );

            Self::from_data(data)
        }
    }
}

impl<'a, T, R: Dim, C: Dim, RSlice: Dim, CSlice: Dim, RStride: Dim, CStride: Dim, S>
    From<&'a mut Matrix<T, R, C, S>> for MatrixSlice<'a, T, RSlice, CSlice, RStride, CStride>
where
    S: Storage<T, R, C>,
    ShapeConstraint: DimEq<R, RSlice>
        + DimEq<C, CSlice>
        + DimEq<RStride, S::RStride>
        + DimEq<CStride, S::CStride>,
{
    fn from(m: &'a mut Matrix<T, R, C, S>) -> Self {
        let (row, col) = m.data.shape();
        let row_slice = RSlice::from_usize(row.value());
        let col_slice = CSlice::from_usize(col.value());

        let (rstride, cstride) = m.strides();

        let rstride_slice = RStride::from_usize(rstride);
        let cstride_slice = CStride::from_usize(cstride);

        unsafe {
            let data = SliceStorage::from_raw_parts(
                m.data.ptr(),
                (row_slice, col_slice),
                (rstride_slice, cstride_slice),
            );

            Matrix::from_data(data)
        }
    }
}

impl<'a, T, R: Dim, C: Dim, RSlice: Dim, CSlice: Dim, RStride: Dim, CStride: Dim, S>
    From<&'a mut Matrix<T, R, C, S>> for MatrixSliceMut<'a, T, RSlice, CSlice, RStride, CStride>
where
    S: StorageMut<T, R, C>,
    ShapeConstraint: DimEq<R, RSlice>
        + DimEq<C, CSlice>
        + DimEq<RStride, S::RStride>
        + DimEq<CStride, S::CStride>,
{
    fn from(m: &'a mut Matrix<T, R, C, S>) -> Self {
        let (row, col) = m.data.shape();
        let row_slice = RSlice::from_usize(row.value());
        let col_slice = CSlice::from_usize(col.value());

        let (rstride, cstride) = m.strides();

        let rstride_slice = RStride::from_usize(rstride);
        let cstride_slice = CStride::from_usize(cstride);

        unsafe {
            let data = SliceStorageMut::from_raw_parts(
                m.data.ptr_mut(),
                (row_slice, col_slice),
                (rstride_slice, cstride_slice),
            );

            Matrix::from_data(data)
        }
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<'a, T> From<Vec<T>> for DVector<T> {
    #[inline]
    fn from(vec: Vec<T>) -> Self {
        Self::from_vec(vec)
    }
}

impl<'a, T, R: Dim, C: Dim, S: ContiguousStorage<T, R, C>> From<&'a Matrix<T, R, C, S>>
    for &'a [T]
{
    #[inline]
    fn from(matrix: &'a Matrix<T, R, C, S>) -> Self {
        matrix.as_slice()
    }
}

impl<'a, T, R: Dim, C: Dim, S: ContiguousStorageMut<T, R, C>> From<&'a mut Matrix<T, R, C, S>>
    for &'a mut [T]
{
    #[inline]
    fn from(matrix: &'a mut Matrix<T, R, C, S>) -> Self {
        matrix.as_mut_slice()
    }
}

impl<'a, T> From<&'a [T]> for DVectorSlice<'a, T> {
    #[inline]
    fn from(slice: &'a [T]) -> Self {
        Self::from_slice(slice, slice.len())
    }
}

impl<'a, T> From<DVectorSlice<'a, T>> for &'a [T] {
    fn from(vec: DVectorSlice<'a, T>) -> &'a [T] {
        vec.data.into_slice()
    }
}

impl<'a, T> From<&'a mut [T]> for DVectorSliceMut<'a, T> {
    #[inline]
    fn from(slice: &'a mut [T]) -> Self {
        Self::from_slice(slice, slice.len())
    }
}

impl<'a, T> From<DVectorSliceMut<'a, T>> for &'a mut [T] {
    fn from(vec: DVectorSliceMut<'a, T>) -> &'a mut [T] {
        vec.data.into_slice_mut()
    }
}

impl<T: Scalar + PrimitiveSimdValue, R: Dim, C: Dim> From<[OMatrix<T::Element, R, C>; 2]>
    for OMatrix<T, R, C>
where
    T: From<[<T as SimdValue>::Element; 2]>,
    T::Element: Scalar + SimdValue,
    DefaultAllocator: Allocator<T, R, C> + Allocator<T::Element, R, C>,
{
    #[inline]
    fn from(arr: [OMatrix<T::Element, R, C>; 2]) -> Self {
        let (nrows, ncols) = arr[0].data.shape();

        Self::from_fn_generic(nrows, ncols, |i, j| {
            [
                arr[0][(i, j)].inlined_clone(),
                arr[1][(i, j)].inlined_clone(),
            ]
            .into()
        })
    }
}

impl<T: Scalar + PrimitiveSimdValue, R: Dim, C: Dim> From<[OMatrix<T::Element, R, C>; 4]>
    for OMatrix<T, R, C>
where
    T: From<[<T as SimdValue>::Element; 4]>,
    T::Element: Scalar + SimdValue,
    DefaultAllocator: Allocator<T, R, C> + Allocator<T::Element, R, C>,
{
    #[inline]
    fn from(arr: [OMatrix<T::Element, R, C>; 4]) -> Self {
        let (nrows, ncols) = arr[0].data.shape();

        Self::from_fn_generic(nrows, ncols, |i, j| {
            [
                arr[0][(i, j)].inlined_clone(),
                arr[1][(i, j)].inlined_clone(),
                arr[2][(i, j)].inlined_clone(),
                arr[3][(i, j)].inlined_clone(),
            ]
            .into()
        })
    }
}

impl<T: Scalar + PrimitiveSimdValue, R: Dim, C: Dim> From<[OMatrix<T::Element, R, C>; 8]>
    for OMatrix<T, R, C>
where
    T: From<[<T as SimdValue>::Element; 8]>,
    T::Element: Scalar + SimdValue,
    DefaultAllocator: Allocator<T, R, C> + Allocator<T::Element, R, C>,
{
    #[inline]
    fn from(arr: [OMatrix<T::Element, R, C>; 8]) -> Self {
        let (nrows, ncols) = arr[0].data.shape();

        Self::from_fn_generic(nrows, ncols, |i, j| {
            [
                arr[0][(i, j)].inlined_clone(),
                arr[1][(i, j)].inlined_clone(),
                arr[2][(i, j)].inlined_clone(),
                arr[3][(i, j)].inlined_clone(),
                arr[4][(i, j)].inlined_clone(),
                arr[5][(i, j)].inlined_clone(),
                arr[6][(i, j)].inlined_clone(),
                arr[7][(i, j)].inlined_clone(),
            ]
            .into()
        })
    }
}

impl<T: Scalar + PrimitiveSimdValue, R: Dim, C: Dim> From<[OMatrix<T::Element, R, C>; 16]>
    for OMatrix<T, R, C>
where
    T: From<[<T as SimdValue>::Element; 16]>,
    T::Element: Scalar + SimdValue,
    DefaultAllocator: Allocator<T, R, C> + Allocator<T::Element, R, C>,
{
    fn from(arr: [OMatrix<T::Element, R, C>; 16]) -> Self {
        let (nrows, ncols) = arr[0].data.shape();

        Self::from_fn_generic(nrows, ncols, |i, j| {
            [
                arr[0][(i, j)].inlined_clone(),
                arr[1][(i, j)].inlined_clone(),
                arr[2][(i, j)].inlined_clone(),
                arr[3][(i, j)].inlined_clone(),
                arr[4][(i, j)].inlined_clone(),
                arr[5][(i, j)].inlined_clone(),
                arr[6][(i, j)].inlined_clone(),
                arr[7][(i, j)].inlined_clone(),
                arr[8][(i, j)].inlined_clone(),
                arr[9][(i, j)].inlined_clone(),
                arr[10][(i, j)].inlined_clone(),
                arr[11][(i, j)].inlined_clone(),
                arr[12][(i, j)].inlined_clone(),
                arr[13][(i, j)].inlined_clone(),
                arr[14][(i, j)].inlined_clone(),
                arr[15][(i, j)].inlined_clone(),
            ]
            .into()
        })
    }
}
