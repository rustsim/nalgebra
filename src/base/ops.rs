use num::{One, Zero};
use std::iter;
use std::mem::MaybeUninit;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use simba::scalar::{ClosedAdd, ClosedDiv, ClosedMul, ClosedNeg, ClosedSub};

use crate::base::allocator::{
    Allocator, InnerAllocator, SameShapeAllocator, SameShapeC, SameShapeR,
};
use crate::base::constraint::{
    AreMultipliable, DimEq, SameNumberOfColumns, SameNumberOfRows, ShapeConstraint,
};
use crate::base::dimension::{Dim, DimMul, DimName, DimProd, Dynamic};
use crate::base::storage::{ContiguousStorageMut, Storage, StorageMut};
use crate::base::{DefaultAllocator, Matrix, MatrixSum, OMatrix, Scalar, VectorSlice};
use crate::storage::InnerOwned;
use crate::{MatrixSliceMut, SimdComplexField};

/*
 *
 * Indexing.
 *
 */
impl<T, R: Dim, C: Dim, S: Storage<T, R, C>> Index<usize> for Matrix<T, R, C, S> {
    type Output = T;

    #[inline]
    fn index(&self, i: usize) -> &Self::Output {
        let ij = self.vector_to_matrix_index(i);
        &self[ij]
    }
}

impl<T, R: Dim, C: Dim, S> Index<(usize, usize)> for Matrix<T, R, C, S>
where
    S: Storage<T, R, C>,
{
    type Output = T;

    #[inline]
    fn index(&self, ij: (usize, usize)) -> &Self::Output {
        let shape = self.shape();
        assert!(
            ij.0 < shape.0 && ij.1 < shape.1,
            "Matrix index out of bounds."
        );

        unsafe { self.get_unchecked((ij.0, ij.1)) }
    }
}

// Mutable versions.
impl<T, R: Dim, C: Dim, S: StorageMut<T, R, C>> IndexMut<usize> for Matrix<T, R, C, S> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        let ij = self.vector_to_matrix_index(i);
        &mut self[ij]
    }
}

impl<T, R: Dim, C: Dim, S> IndexMut<(usize, usize)> for Matrix<T, R, C, S>
where
    S: StorageMut<T, R, C>,
{
    #[inline]
    fn index_mut(&mut self, ij: (usize, usize)) -> &mut T {
        let shape = self.shape();
        assert!(
            ij.0 < shape.0 && ij.1 < shape.1,
            "Matrix index out of bounds."
        );

        unsafe { self.get_unchecked_mut((ij.0, ij.1)) }
    }
}

/*
 *
 * Neg
 *
 */
impl<T, R: Dim, C: Dim, S> Neg for Matrix<T, R, C, S>
where
    T: Scalar + ClosedNeg,
    S: Storage<T, R, C>,
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = OMatrix<T, R, C>;

    #[inline]
    fn neg(self) -> Self::Output {
        let mut res = self.into_owned();
        res.neg_mut();
        res
    }
}

impl<'a, T, R: Dim, C: Dim, S> Neg for &'a Matrix<T, R, C, S>
where
    T: Scalar + ClosedNeg,
    S: Storage<T, R, C>,
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = OMatrix<T, R, C>;

    #[inline]
    fn neg(self) -> Self::Output {
        -self.clone_owned()
    }
}

impl<T, R: Dim, C: Dim, S> Matrix<T, R, C, S>
where
    T: Scalar + ClosedNeg,
    S: StorageMut<T, R, C>,
{
    /// Negates `self` in-place.
    #[inline]
    pub fn neg_mut(&mut self) {
        for e in self.iter_mut() {
            *e = -e.inlined_clone()
        }
    }
}

/*
 *
 * Addition & Subtraction
 *
 */

macro_rules! componentwise_binop_impl(
    ($Trait: ident, $method: ident, $bound: ident;
     $TraitAssign: ident, $method_assign: ident, $method_assign_statically_unchecked: ident,
     $method_assign_statically_unchecked_rhs: ident;
     $method_to: ident, $method_to_statically_unchecked: ident) => {
        impl<T, R1: Dim, C1: Dim, SA: Storage<T, R1, C1>> Matrix<T, R1, C1, SA>
        where
            T: Scalar + $bound
        {
            /*
             *
             * Methods without dimension checking at compile-time.
             * This is useful for code reuse because the sum representative system does not play
             * nicely with static checks.
             *
             */
            #[inline]
            fn $method_to_statically_unchecked<R2: Dim, C2: Dim, SB, R3: Dim, C3: Dim, SC>(
                &self, rhs: &Matrix<T, R2, C2, SB>, out: &mut Matrix<MaybeUninit<T>, R3, C3, SC>
            ) where
                SB: Storage<T, R2, C2>,
                SC: StorageMut<MaybeUninit<T>, R3, C3>
            {
                assert_eq!(self.shape(), rhs.shape(), "Matrix addition/subtraction dimensions mismatch.");
                assert_eq!(self.shape(), out.shape(), "Matrix addition/subtraction output dimensions mismatch.");

                // This is the most common case and should be deduced at compile-time.
                // TODO: use specialization instead?
                unsafe {
                    if self.data.is_contiguous() && rhs.data.is_contiguous() && out.data.is_contiguous() {
                        let arr1 = self.data.as_slice_unchecked();
                        let arr2 = rhs.data.as_slice_unchecked();
                        let out = out.data.as_mut_slice_unchecked();
                        for i in 0..arr1.len() {
                            *out.get_unchecked_mut(i) = MaybeUninit::new(
                                arr1.get_unchecked(i).inlined_clone().$method(arr2.get_unchecked(i).inlined_clone()
                            ));
                        }
                    } else {
                        for j in 0..self.ncols() {
                            for i in 0..self.nrows() {
                                *out.get_unchecked_mut((i, j)) = MaybeUninit::new(
                                    self.get_unchecked((i, j)).inlined_clone().$method(rhs.get_unchecked((i, j)).inlined_clone())
                                );
                            }
                        }
                    }
                }
            }

            #[inline]
            fn $method_assign_statically_unchecked<R2: Dim, C2: Dim, SB>(
                &mut self, rhs: &Matrix<T, R2, C2, SB>
            ) where
                SA: StorageMut<T, R1, C1>,
                SB: Storage<T, R2, C2>
            {
                assert_eq!(self.shape(), rhs.shape(), "Matrix addition/subtraction dimensions mismatch.");

                // This is the most common case and should be deduced at compile-time.
                // TODO: use specialization instead?
                unsafe {
                    if self.data.is_contiguous() && rhs.data.is_contiguous() {
                        let arr1 = self.data.as_mut_slice_unchecked();
                        let arr2 = rhs.data.as_slice_unchecked();

                        for i in 0 .. arr2.len() {
                            arr1.get_unchecked_mut(i).$method_assign(arr2.get_unchecked(i).inlined_clone());
                        }
                    } else {
                        for j in 0 .. rhs.ncols() {
                            for i in 0 .. rhs.nrows() {
                                self.get_unchecked_mut((i, j)).$method_assign(rhs.get_unchecked((i, j)).inlined_clone())
                            }
                        }
                    }
                }
            }

            #[inline]
            fn $method_assign_statically_unchecked_rhs<R2: Dim, C2: Dim, SB>(
                &self, rhs: &mut Matrix<T, R2, C2, SB>
            ) where
                SB: StorageMut<T, R2, C2>
            {
                assert_eq!(self.shape(), rhs.shape(), "Matrix addition/subtraction dimensions mismatch.");

                // This is the most common case and should be deduced at compile-time.
                // TODO: use specialization instead?
                unsafe {
                    if self.data.is_contiguous() && rhs.data.is_contiguous() {
                        let arr1 = self.data.as_slice_unchecked();
                        let arr2 = rhs.data.as_mut_slice_unchecked();

                        for i in 0 .. arr1.len() {
                            let res = arr1.get_unchecked(i).inlined_clone().$method(arr2.get_unchecked(i).inlined_clone());
                            *arr2.get_unchecked_mut(i) = res;
                        }
                    } else {
                        for j in 0 .. self.ncols() {
                            for i in 0 .. self.nrows() {
                                let r = rhs.get_unchecked_mut((i, j));
                                *r = self.get_unchecked((i, j)).inlined_clone().$method(r.inlined_clone())
                            }
                        }
                    }
                }
            }


            /*
             *
             * Methods without dimension checking at compile-time.
             * This is useful for code reuse because the sum representative system does not plays
             * easily with static checks.
             *
             */
            /// Equivalent to `self + rhs` but stores the result into `out` to avoid allocations.
            #[inline]
            pub fn $method_to<R2: Dim, C2: Dim, SB, R3: Dim, C3: Dim, SC>(
                &self,
                rhs: &Matrix<T, R2, C2, SB>,
                out: &mut Matrix<MaybeUninit<T>, R3, C3, SC>
            ) where
                SB: Storage<T, R2, C2>,
                SC: StorageMut<MaybeUninit<T>, R3, C3>,
                ShapeConstraint:
                    SameNumberOfRows<R1, R2> +
                    SameNumberOfColumns<C1, C2> +
                    SameNumberOfRows<R1, R3> +
                    SameNumberOfColumns<C1, C3>
            {
                self.$method_to_statically_unchecked(rhs, out)
            }
        }

        impl<'b, T, R1, C1, R2, C2, SA, SB> $Trait<&'b Matrix<T, R2, C2, SB>> for Matrix<T, R1, C1, SA>
            where R1: Dim, C1: Dim, R2: Dim, C2: Dim,
                  T: Scalar + $bound,
                  SA: Storage<T, R1, C1>,
                  SB: Storage<T, R2, C2>,
                  DefaultAllocator: SameShapeAllocator<T, R1, C1, R2, C2>,
                  ShapeConstraint:  SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2> {
            type Output = MatrixSum<T, R1, C1, R2, C2>;

            #[inline]
            fn $method(self, rhs: &'b Matrix<T, R2, C2, SB>) -> Self::Output {
                assert_eq!(self.shape(), rhs.shape(), "Matrix addition/subtraction dimensions mismatch.");
                let mut res = self.into_owned_sum::<R2, C2>();
                res.$method_assign_statically_unchecked(rhs);
                res
            }
        }

        impl<'a, T, R1: Dim, C1: Dim, R2: Dim, C2: Dim, SA, SB> $Trait<Matrix<T, R2, C2, SB>> for &'a Matrix<T, R1, C1, SA>
        where
            T: Scalar + $bound,
            SA: Storage<T, R1, C1>,
            SB: Storage<T, R2, C2>,
            DefaultAllocator: SameShapeAllocator<T, R2, C2, R1, C1>,
            ShapeConstraint:  SameNumberOfRows<R2, R1> + SameNumberOfColumns<C2, C1>
        {
            type Output = MatrixSum<T, R2, C2, R1, C1>;

            #[inline]
            fn $method(self, rhs: Matrix<T, R2, C2, SB>) -> Self::Output {
                let mut rhs = rhs.into_owned_sum::<R1, C1>();
                assert_eq!(self.shape(), rhs.shape(), "Matrix addition/subtraction dimensions mismatch.");
                self.$method_assign_statically_unchecked_rhs(&mut rhs);
                rhs
            }
        }

        impl<T, R1: Dim, C1: Dim, R2: Dim, C2: Dim, SA, SB> $Trait<Matrix<T, R2, C2, SB>> for Matrix<T, R1, C1, SA>
        where
            T: Scalar + $bound,
            SA: Storage<T, R1, C1>,
            SB: Storage<T, R2, C2>,
            DefaultAllocator: SameShapeAllocator<T, R1, C1, R2, C2>,
            ShapeConstraint:  SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>
        {
            type Output = MatrixSum<T, R1, C1, R2, C2>;

            #[inline]
            fn $method(self, rhs: Matrix<T, R2, C2, SB>) -> Self::Output {
                self.$method(&rhs)
            }
        }

        impl<'a, 'b, T, R1: Dim, C1: Dim, R2: Dim, C2: Dim, SA, SB> $Trait<&'b Matrix<T, R2, C2, SB>> for &'a Matrix<T, R1, C1, SA>
        where
            T: Scalar + $bound,
            SA: Storage<T, R1, C1>,
            SB: Storage<T, R2, C2>,
            DefaultAllocator: SameShapeAllocator<T, R1, C1, R2, C2>,
            ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>
        {
            type Output = MatrixSum<T, R1, C1, R2, C2>;

            #[inline]
            fn $method(self, rhs: &'b Matrix<T, R2, C2, SB>) -> Self::Output {
                let (nrows, ncols) = self.shape();
                let nrows: SameShapeR<R1, R2> = Dim::from_usize(nrows);
                let ncols: SameShapeC<C1, C2> = Dim::from_usize(ncols);
                let mut res = Matrix::new_uninitialized_generic(nrows, ncols);

                self.$method_to_statically_unchecked(rhs, &mut res);
                unsafe { res.assume_init() }
            }
        }

        impl<'b, T, R1: Dim, C1: Dim, R2: Dim, C2: Dim, SA, SB> $TraitAssign<&'b Matrix<T, R2, C2, SB>> for Matrix<T, R1, C1, SA>
        where
            T: Scalar + $bound,
            SA: StorageMut<T, R1, C1>,
            SB: Storage<T, R2, C2>,
            ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>
        {
            #[inline]
            fn $method_assign(&mut self, rhs: &'b Matrix<T, R2, C2, SB>) {
                self.$method_assign_statically_unchecked(rhs)
            }
        }

        impl<T, R1: Dim, C1: Dim, R2: Dim, C2: Dim, SA, SB> $TraitAssign<Matrix<T, R2, C2, SB>> for Matrix<T, R1, C1, SA>
        where
            T: Scalar + $bound,
            SA: StorageMut<T, R1, C1>,
            SB: Storage<T, R2, C2>,
            ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>
        {
            #[inline]
            fn $method_assign(&mut self, rhs: Matrix<T, R2, C2, SB>) {
                self.$method_assign(&rhs)
            }
        }
    }
);

componentwise_binop_impl!(Add, add, ClosedAdd;
                          AddAssign, add_assign, add_assign_statically_unchecked, add_assign_statically_unchecked_mut;
                          add_to, add_to_statically_unchecked);
componentwise_binop_impl!(Sub, sub, ClosedSub;
                          SubAssign, sub_assign, sub_assign_statically_unchecked, sub_assign_statically_unchecked_mut;
                          sub_to, sub_to_statically_unchecked);

impl<T, R: DimName, C: DimName> iter::Sum for OMatrix<T, R, C>
where
    T: Scalar + ClosedAdd + Zero,
    DefaultAllocator: Allocator<T, R, C>,
{
    fn sum<I: Iterator<Item = OMatrix<T, R, C>>>(iter: I) -> OMatrix<T, R, C> {
        iter.fold(Matrix::zero(), |acc, x| acc + x)
    }
}

impl<T, C: Dim> iter::Sum for OMatrix<T, Dynamic, C>
where
    T: Scalar + ClosedAdd + Zero,
    DefaultAllocator: Allocator<T, Dynamic, C>,
{
    /// # Example
    /// ```
    /// # use nalgebra::DVector;
    /// assert_eq!(vec![DVector::repeat(3, 1.0f64),
    ///                 DVector::repeat(3, 1.0f64),
    ///                 DVector::repeat(3, 1.0f64)].into_iter().sum::<DVector<f64>>(),
    ///            DVector::repeat(3, 1.0f64) + DVector::repeat(3, 1.0f64) + DVector::repeat(3, 1.0f64));
    /// ```
    ///
    /// # Panics
    /// Panics if the iterator is empty:
    /// ```should_panic
    /// # use std::iter;
    /// # use nalgebra::DMatrix;
    /// iter::empty::<DMatrix<f64>>().sum::<DMatrix<f64>>(); // panics!
    /// ```
    fn sum<I: Iterator<Item = OMatrix<T, Dynamic, C>>>(mut iter: I) -> OMatrix<T, Dynamic, C> {
        if let Some(first) = iter.next() {
            iter.fold(first, |acc, x| acc + x)
        } else {
            panic!("Cannot compute `sum` of empty iterator.")
        }
    }
}

impl<'a, T, R: DimName, C: DimName> iter::Sum<&'a OMatrix<T, R, C>> for OMatrix<T, R, C>
where
    T: Scalar + ClosedAdd + Zero,
    DefaultAllocator: Allocator<T, R, C>,
{
    fn sum<I: Iterator<Item = &'a OMatrix<T, R, C>>>(iter: I) -> OMatrix<T, R, C> {
        iter.fold(Matrix::zero(), |acc, x| acc + x)
    }
}

impl<'a, T, C: Dim> iter::Sum<&'a OMatrix<T, Dynamic, C>> for OMatrix<T, Dynamic, C>
where
    T: Scalar + ClosedAdd + Zero,
    DefaultAllocator: Allocator<T, Dynamic, C>,

    // TODO: we should take out this trait bound, as T: Clone should suffice.
    // The brute way to do it would be how it was already done: by adding this
    // trait bound on the associated type itself.
    InnerOwned<T, Dynamic, C>: Clone,
{
    /// # Example
    /// ```
    /// # use nalgebra::DVector;
    /// let v = &DVector::repeat(3, 1.0f64);
    ///
    /// assert_eq!(vec![v, v, v].into_iter().sum::<DVector<f64>>(),
    ///            v + v + v);
    /// ```
    ///
    /// # Panics
    /// Panics if the iterator is empty:
    /// ```should_panic
    /// # use std::iter;
    /// # use nalgebra::DMatrix;
    /// iter::empty::<&DMatrix<f64>>().sum::<DMatrix<f64>>(); // panics!
    /// ```
    fn sum<I: Iterator<Item = &'a OMatrix<T, Dynamic, C>>>(mut iter: I) -> OMatrix<T, Dynamic, C> {
        if let Some(first) = iter.next() {
            iter.fold(first.clone(), |acc, x| acc + x)
        } else {
            panic!("Cannot compute `sum` of empty iterator.")
        }
    }
}

/*
 *
 * Multiplication
 *
 */

// Matrix × Scalar
// Matrix / Scalar
macro_rules! componentwise_scalarop_impl(
    ($Trait: ident, $method: ident, $bound: ident;
     $TraitAssign: ident, $method_assign: ident) => {
        impl<T, R: Dim, C: Dim, S> $Trait<T> for Matrix<T, R, C, S>
            where T: Scalar + $bound,
                  S: Storage<T, R, C>,
                  DefaultAllocator: Allocator<T, R, C> {
            type Output = OMatrix<T, R, C>;

            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                let mut res = self.into_owned();

                // XXX: optimize our iterator!
                //
                // Using our own iterator prevents loop unrolling, which breaks some optimization
                // (like SIMD). On the other hand, using the slice iterator is 4x faster.

                // for left in res.iter_mut() {
                for left in res.as_mut_slice().iter_mut() {
                    *left = left.inlined_clone().$method(rhs.inlined_clone())
                }

                res
            }
        }

        impl<'a, T, R: Dim, C: Dim, S> $Trait<T> for &'a Matrix<T, R, C, S>
            where T: Scalar + $bound,
                  S: Storage<T, R, C>,
                  DefaultAllocator: Allocator<T, R, C> {
            type Output = OMatrix<T, R, C>;

            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                self.clone_owned().$method(rhs)
            }
        }

        impl<T, R: Dim, C: Dim, S> $TraitAssign<T> for Matrix<T, R, C, S>
            where T: Scalar + $bound,
                  S: StorageMut<T, R, C> {
            #[inline]
            fn $method_assign(&mut self, rhs: T) {
                for j in 0 .. self.ncols() {
                    for i in 0 .. self.nrows() {
                        unsafe { self.get_unchecked_mut((i, j)).$method_assign(rhs.inlined_clone()) };
                    }
                }
            }
        }
    }
);

componentwise_scalarop_impl!(Mul, mul, ClosedMul; MulAssign, mul_assign);
componentwise_scalarop_impl!(Div, div, ClosedDiv; DivAssign, div_assign);

macro_rules! left_scalar_mul_impl(
    ($($T: ty),* $(,)*) => {$(
        impl<R: Dim, C: Dim, S: Storage<$T, R, C>> Mul<Matrix<$T, R, C, S>> for $T
            where DefaultAllocator: Allocator<$T, R, C> {
            type Output = OMatrix<$T, R, C>;

            #[inline]
            fn mul(self, rhs: Matrix<$T, R, C, S>) -> Self::Output {
                let mut res = rhs.into_owned();

                // XXX: optimize our iterator!
                //
                // Using our own iterator prevents loop unrolling, which breaks some optimization
                // (like SIMD). On the other hand, using the slice iterator is 4x faster.

                // for rhs in res.iter_mut() {
                for rhs in res.as_mut_slice().iter_mut() {
                    *rhs *= self
                }

                res
            }
        }

        impl<'b, R: Dim, C: Dim, S: Storage<$T, R, C>> Mul<&'b Matrix<$T, R, C, S>> for $T
            where DefaultAllocator: Allocator<$T, R, C> {
            type Output = OMatrix<$T, R, C>;

            #[inline]
            fn mul(self, rhs: &'b Matrix<$T, R, C, S>) -> Self::Output {
                self * rhs.clone_owned()
            }
        }
    )*}
);

left_scalar_mul_impl!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);

// Matrix × Matrix
impl<'a, 'b, T, R1: Dim, C1: Dim, R2: Dim, C2: Dim, SA, SB> Mul<&'b Matrix<T, R2, C2, SB>>
    for &'a Matrix<T, R1, C1, SA>
where
    T: Scalar + Zero + One + ClosedAdd + ClosedMul,
    SA: Storage<T, R1, C1>,
    SB: Storage<T, R2, C2>,
    DefaultAllocator: Allocator<T, R1, C2>,
    ShapeConstraint: AreMultipliable<R1, C1, R2, C2>,
{
    type Output = OMatrix<T, R1, C2>;

    #[inline]
    fn mul(self, rhs: &'b Matrix<T, R2, C2, SB>) -> Self::Output {
        let mut res = Matrix::new_uninitialized_generic(self.data.shape().0, rhs.data.shape().1);
        let _ = self.mul_to(rhs, &mut res);
        unsafe { res.assume_init() }
    }
}

impl<'a, T, R1: Dim, C1: Dim, R2: Dim, C2: Dim, SA, SB> Mul<Matrix<T, R2, C2, SB>>
    for &'a Matrix<T, R1, C1, SA>
where
    T: Scalar + Zero + One + ClosedAdd + ClosedMul,
    SB: Storage<T, R2, C2>,
    SA: Storage<T, R1, C1>,
    DefaultAllocator: Allocator<T, R1, C2>,
    ShapeConstraint: AreMultipliable<R1, C1, R2, C2>,
{
    type Output = OMatrix<T, R1, C2>;

    #[inline]
    fn mul(self, rhs: Matrix<T, R2, C2, SB>) -> Self::Output {
        self * &rhs
    }
}

impl<'b, T, R1: Dim, C1: Dim, R2: Dim, C2: Dim, SA, SB> Mul<&'b Matrix<T, R2, C2, SB>>
    for Matrix<T, R1, C1, SA>
where
    T: Scalar + Zero + One + ClosedAdd + ClosedMul,
    SB: Storage<T, R2, C2>,
    SA: Storage<T, R1, C1>,
    DefaultAllocator: Allocator<T, R1, C2>,
    ShapeConstraint: AreMultipliable<R1, C1, R2, C2>,
{
    type Output = OMatrix<T, R1, C2>;

    #[inline]
    fn mul(self, rhs: &'b Matrix<T, R2, C2, SB>) -> Self::Output {
        &self * rhs
    }
}

impl<T, R1: Dim, C1: Dim, R2: Dim, C2: Dim, SA, SB> Mul<Matrix<T, R2, C2, SB>>
    for Matrix<T, R1, C1, SA>
where
    T: Scalar + Zero + One + ClosedAdd + ClosedMul,
    SB: Storage<T, R2, C2>,
    SA: Storage<T, R1, C1>,
    DefaultAllocator: Allocator<T, R1, C2>,
    ShapeConstraint: AreMultipliable<R1, C1, R2, C2>,
{
    type Output = OMatrix<T, R1, C2>;

    #[inline]
    fn mul(self, rhs: Matrix<T, R2, C2, SB>) -> Self::Output {
        &self * &rhs
    }
}

// TODO: this is too restrictive:
//    − we can't use `a *= b` when `a` is a mutable slice.
//    − we can't use `a *= b` when C2 is not equal to C1.
impl<T, R1: Dim, C1: Dim, R2: Dim, SA, SB> MulAssign<Matrix<T, R2, C1, SB>>
    for Matrix<T, R1, C1, SA>
where
    T: Scalar + Zero + One + ClosedAdd + ClosedMul,
    SB: Storage<T, R2, C1>,
    SA: ContiguousStorageMut<T, R1, C1>,
    ShapeConstraint: AreMultipliable<R1, C1, R2, C1>,
    DefaultAllocator: Allocator<T, R1, C1> + InnerAllocator<T, R1, C1, Buffer = SA>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Matrix<T, R2, C1, SB>) {
        *self = &*self * rhs
    }
}

impl<'b, T, R1: Dim, C1: Dim, R2: Dim, SA, SB> MulAssign<&'b Matrix<T, R2, C1, SB>>
    for Matrix<T, R1, C1, SA>
where
    T: Scalar + Zero + One + ClosedAdd + ClosedMul,
    SB: Storage<T, R2, C1>,
    SA: ContiguousStorageMut<T, R1, C1>,
    ShapeConstraint: AreMultipliable<R1, C1, R2, C1>,
    // TODO: this is too restrictive. See comments for the non-ref version.
    DefaultAllocator: Allocator<T, R1, C1> + InnerAllocator<T, R1, C1, Buffer = SA>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: &'b Matrix<T, R2, C1, SB>) {
        *self = &*self * rhs
    }
}

/// # Special multiplications.
impl<T, R1: Dim, C1: Dim, SA> Matrix<T, R1, C1, SA>
where
    T: Scalar + Zero + One + ClosedAdd + ClosedMul,
    SA: Storage<T, R1, C1>,
{
    /// Equivalent to `self.transpose() * rhs`.
    #[inline]
    #[must_use]
    pub fn tr_mul<R2: Dim, C2: Dim, SB>(&self, rhs: &Matrix<T, R2, C2, SB>) -> OMatrix<T, C1, C2>
    where
        SB: Storage<T, R2, C2>,
        DefaultAllocator: Allocator<T, C1, C2>,
        ShapeConstraint: SameNumberOfRows<R1, R2>,
    {
        let mut res = Matrix::new_uninitialized_generic(self.data.shape().1, rhs.data.shape().1);
        self.tr_mul_to(rhs, &mut res);
        unsafe { res.assume_init() }
    }

    /// Equivalent to `self.adjoint() * rhs`.
    #[inline]
    #[must_use]
    pub fn ad_mul<R2: Dim, C2: Dim, SB>(&self, rhs: &Matrix<T, R2, C2, SB>) -> OMatrix<T, C1, C2>
    where
        T: SimdComplexField,
        SB: Storage<T, R2, C2>,
        DefaultAllocator: Allocator<T, C1, C2>,
        ShapeConstraint: SameNumberOfRows<R1, R2>,
    {
        let mut res = Matrix::new_uninitialized_generic(self.data.shape().1, rhs.data.shape().1);
        self.ad_mul_to(rhs, &mut res);
        unsafe { res.assume_init() }
    }

    #[inline(always)]
    fn xx_mul_to<R2: Dim, C2: Dim, SB, R3: Dim, C3: Dim, SC>(
        &self,
        rhs: &Matrix<T, R2, C2, SB>,
        out: &mut Matrix<MaybeUninit<T>, R3, C3, SC>,
        dot: impl Fn(
            &VectorSlice<T, R1, SA::RStride, SA::CStride>,
            &VectorSlice<T, R2, SB::RStride, SB::CStride>,
        ) -> T,
    ) where
        SB: Storage<T, R2, C2>,
        SC: StorageMut<MaybeUninit<T>, R3, C3>,
        ShapeConstraint: SameNumberOfRows<R1, R2> + DimEq<C1, R3> + DimEq<C2, C3>,
    {
        let (nrows1, ncols1) = self.shape();
        let (nrows2, ncols2) = rhs.shape();
        let (nrows3, ncols3) = out.shape();

        assert!(
            nrows1 == nrows2,
            "Matrix multiplication dimensions mismatch {:?} and {:?}: left rows != right rows.",
            self.shape(),
            rhs.shape()
        );
        assert!(
            ncols1 == nrows3,
            "Matrix multiplication output dimensions mismatch {:?} and {:?}: left cols != right rows.",
            self.shape(),
            out.shape()
        );
        assert!(
            ncols2 == ncols3,
            "Matrix multiplication output dimensions mismatch {:?} and {:?}: left cols != right cols",
            rhs.shape(),
            out.shape()
        );

        for i in 0..ncols1 {
            for j in 0..ncols2 {
                let dot = dot(&self.column(i), &rhs.column(j));
                unsafe {
                    *out.get_unchecked_mut((i, j)) = MaybeUninit::new(dot);
                }
            }
        }
    }

    /// Equivalent to `self.transpose() * rhs` but stores the result into `out` to avoid
    /// allocations.
    #[inline]
    pub fn tr_mul_to<R2: Dim, C2: Dim, SB, R3: Dim, C3: Dim, SC>(
        &self,
        rhs: &Matrix<T, R2, C2, SB>,
        out: &mut Matrix<MaybeUninit<T>, R3, C3, SC>,
    ) where
        SB: Storage<T, R2, C2>,
        SC: StorageMut<MaybeUninit<T>, R3, C3>,
        ShapeConstraint: SameNumberOfRows<R1, R2> + DimEq<C1, R3> + DimEq<C2, C3>,
    {
        self.xx_mul_to(rhs, out, |a, b| a.dot(b))
    }

    /// Equivalent to `self.adjoint() * rhs` but stores the result into `out` to avoid
    /// allocations.
    #[inline]
    pub fn ad_mul_to<R2: Dim, C2: Dim, SB, R3: Dim, C3: Dim, SC>(
        &self,
        rhs: &Matrix<T, R2, C2, SB>,
        out: &mut Matrix<MaybeUninit<T>, R3, C3, SC>,
    ) where
        T: SimdComplexField,
        SB: Storage<T, R2, C2>,
        SC: StorageMut<MaybeUninit<T>, R3, C3>,
        ShapeConstraint: SameNumberOfRows<R1, R2> + DimEq<C1, R3> + DimEq<C2, C3>,
    {
        self.xx_mul_to(rhs, out, |a, b| a.dotc(b))
    }

    /// Equivalent to `self * rhs` but stores the result into `out` to avoid allocations.
    #[inline]
    pub fn mul_to<'a, R2: Dim, C2: Dim, SB, R3: Dim, C3: Dim, SC>(
        &self,
        rhs: &Matrix<T, R2, C2, SB>,
        out: &'a mut Matrix<MaybeUninit<T>, R3, C3, SC>,
    ) -> MatrixSliceMut<'a, T, R3, C3, SC::RStride, SC::CStride>
    where
        SB: Storage<T, R2, C2>,
        SC: StorageMut<MaybeUninit<T>, R3, C3>,
        ShapeConstraint: SameNumberOfRows<R3, R1>
            + SameNumberOfColumns<C3, C2>
            + AreMultipliable<R1, C1, R2, C2>,
    {
        out.gemm_z(T::one(), self, rhs)
    }

    /// The kronecker product of two matrices (aka. tensor product of the corresponding linear
    /// maps).
    #[must_use]
    pub fn kronecker<R2: Dim, C2: Dim, SB>(
        &self,
        rhs: &Matrix<T, R2, C2, SB>,
    ) -> OMatrix<T, DimProd<R1, R2>, DimProd<C1, C2>>
    where
        T: ClosedMul,
        R1: DimMul<R2>,
        C1: DimMul<C2>,
        SB: Storage<T, R2, C2>,
        DefaultAllocator: Allocator<T, DimProd<R1, R2>, DimProd<C1, C2>>,
    {
        let (nrows1, ncols1) = self.data.shape();
        let (nrows2, ncols2) = rhs.data.shape();

        let mut res = Matrix::new_uninitialized_generic(nrows1.mul(nrows2), ncols1.mul(ncols2));

        {
            let mut data_res = res.data.ptr_mut();

            for j1 in 0..ncols1.value() {
                for j2 in 0..ncols2.value() {
                    for i1 in 0..nrows1.value() {
                        unsafe {
                            let coeff = self.get_unchecked((i1, j1)).inlined_clone();

                            for i2 in 0..nrows2.value() {
                                *data_res = MaybeUninit::new(
                                    coeff.inlined_clone()
                                        * rhs.get_unchecked((i2, j2)).inlined_clone(),
                                );
                                data_res = data_res.offset(1);
                            }
                        }
                    }
                }
            }
        }

        unsafe { res.assume_init() }
    }
}

impl<T, D: DimName> iter::Product for OMatrix<T, D, D>
where
    T: Scalar + Zero + One + ClosedMul + ClosedAdd,
    DefaultAllocator: Allocator<T, D, D>,
{
    fn product<I: Iterator<Item = OMatrix<T, D, D>>>(iter: I) -> OMatrix<T, D, D> {
        iter.fold(Matrix::one(), |acc, x| acc * x)
    }
}

impl<'a, T, D: DimName> iter::Product<&'a OMatrix<T, D, D>> for OMatrix<T, D, D>
where
    T: Scalar + Zero + One + ClosedMul + ClosedAdd,
    DefaultAllocator: Allocator<T, D, D>,
{
    fn product<I: Iterator<Item = &'a OMatrix<T, D, D>>>(iter: I) -> OMatrix<T, D, D> {
        iter.fold(Matrix::one(), |acc, x| acc * x)
    }
}
