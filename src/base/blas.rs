//! Implements a subset of the Basic Linear Algebra Subprograms (BLAS), a
//! standard and highly optimized set of basic vector and matrix operations.
//!
//! To avoid unsoundness due to mishandling of uninitialized data, we divide our
//! methods into two groups: those that take in a `&mut` to a matrix, and those
//! that return an owned matrix that would otherwise result from setting a
//! parameter to zero in the other methods.

use crate::{MatrixSliceMut, SimdComplexField, VectorSliceMut};
#[cfg(feature = "std")]
use matrixmultiply;
use num::{One, Zero};
use simba::scalar::{ClosedAdd, ClosedMul};
#[cfg(feature = "std")]
use std::mem;
use std::mem::MaybeUninit;

use crate::base::allocator::Allocator;
use crate::base::constraint::{
    AreMultipliable, DimEq, SameNumberOfColumns, SameNumberOfRows, ShapeConstraint,
};
use crate::base::dimension::{Const, Dim, Dynamic, U1, U2, U3, U4};
use crate::base::storage::{Storage, StorageMut};
use crate::base::{
    DVectorSlice, DefaultAllocator, Matrix, Scalar, SquareMatrix, Vector, VectorSlice,
};

/// # Dot/scalar product
impl<T, R: Dim, C: Dim, S: Storage<T, R, C>> Matrix<T, R, C, S>
where
    T: Scalar + Zero + ClosedAdd + ClosedMul,
{
    #[inline(always)]
    fn dotx<R2: Dim, C2: Dim, SB>(
        &self,
        rhs: &Matrix<T, R2, C2, SB>,
        conjugate: impl Fn(T) -> T,
    ) -> T
    where
        SB: Storage<T, R2, C2>,
        ShapeConstraint: DimEq<R, R2> + DimEq<C, C2>,
    {
        assert!(
            self.nrows() == rhs.nrows(),
            "Dot product dimensions mismatch for shapes {:?} and {:?}: left rows != right rows.",
            self.shape(),
            rhs.shape(),
        );

        assert!(
            self.ncols() == rhs.ncols(),
            "Dot product dimensions mismatch for shapes {:?} and {:?}: left cols != right cols.",
            self.shape(),
            rhs.shape(),
        );

        // So we do some special cases for common fixed-size vectors of dimension lower than 8
        // because the `for` loop below won't be very efficient on those.
        if (R::is::<U2>() || R2::is::<U2>()) && (C::is::<U1>() || C2::is::<U1>()) {
            unsafe {
                let a = conjugate(self.get_unchecked((0, 0)).inlined_clone())
                    * rhs.get_unchecked((0, 0)).inlined_clone();
                let b = conjugate(self.get_unchecked((1, 0)).inlined_clone())
                    * rhs.get_unchecked((1, 0)).inlined_clone();

                return a + b;
            }
        }
        if (R::is::<U3>() || R2::is::<U3>()) && (C::is::<U1>() || C2::is::<U1>()) {
            unsafe {
                let a = conjugate(self.get_unchecked((0, 0)).inlined_clone())
                    * rhs.get_unchecked((0, 0)).inlined_clone();
                let b = conjugate(self.get_unchecked((1, 0)).inlined_clone())
                    * rhs.get_unchecked((1, 0)).inlined_clone();
                let c = conjugate(self.get_unchecked((2, 0)).inlined_clone())
                    * rhs.get_unchecked((2, 0)).inlined_clone();

                return a + b + c;
            }
        }
        if (R::is::<U4>() || R2::is::<U4>()) && (C::is::<U1>() || C2::is::<U1>()) {
            unsafe {
                let mut a = conjugate(self.get_unchecked((0, 0)).inlined_clone())
                    * rhs.get_unchecked((0, 0)).inlined_clone();
                let mut b = conjugate(self.get_unchecked((1, 0)).inlined_clone())
                    * rhs.get_unchecked((1, 0)).inlined_clone();
                let c = conjugate(self.get_unchecked((2, 0)).inlined_clone())
                    * rhs.get_unchecked((2, 0)).inlined_clone();
                let d = conjugate(self.get_unchecked((3, 0)).inlined_clone())
                    * rhs.get_unchecked((3, 0)).inlined_clone();

                a += c;
                b += d;

                return a + b;
            }
        }

        // All this is inspired from the "unrolled version" discussed in:
        // https://blog.theincredibleholk.org/blog/2012/12/10/optimizing-dot-product/
        //
        // And this comment from bluss:
        // https://users.rust-lang.org/t/how-to-zip-two-slices-efficiently/2048/12
        let mut res = T::zero();

        // We have to define them outside of the loop (and not inside at first assignment)
        // otherwise vectorization won't kick in for some reason.
        let mut acc0;
        let mut acc1;
        let mut acc2;
        let mut acc3;
        let mut acc4;
        let mut acc5;
        let mut acc6;
        let mut acc7;

        for j in 0..self.ncols() {
            let mut i = 0;

            acc0 = T::zero();
            acc1 = T::zero();
            acc2 = T::zero();
            acc3 = T::zero();
            acc4 = T::zero();
            acc5 = T::zero();
            acc6 = T::zero();
            acc7 = T::zero();

            while self.nrows() - i >= 8 {
                acc0 += unsafe {
                    conjugate(self.get_unchecked((i, j)).inlined_clone())
                        * rhs.get_unchecked((i, j)).inlined_clone()
                };
                acc1 += unsafe {
                    conjugate(self.get_unchecked((i + 1, j)).inlined_clone())
                        * rhs.get_unchecked((i + 1, j)).inlined_clone()
                };
                acc2 += unsafe {
                    conjugate(self.get_unchecked((i + 2, j)).inlined_clone())
                        * rhs.get_unchecked((i + 2, j)).inlined_clone()
                };
                acc3 += unsafe {
                    conjugate(self.get_unchecked((i + 3, j)).inlined_clone())
                        * rhs.get_unchecked((i + 3, j)).inlined_clone()
                };
                acc4 += unsafe {
                    conjugate(self.get_unchecked((i + 4, j)).inlined_clone())
                        * rhs.get_unchecked((i + 4, j)).inlined_clone()
                };
                acc5 += unsafe {
                    conjugate(self.get_unchecked((i + 5, j)).inlined_clone())
                        * rhs.get_unchecked((i + 5, j)).inlined_clone()
                };
                acc6 += unsafe {
                    conjugate(self.get_unchecked((i + 6, j)).inlined_clone())
                        * rhs.get_unchecked((i + 6, j)).inlined_clone()
                };
                acc7 += unsafe {
                    conjugate(self.get_unchecked((i + 7, j)).inlined_clone())
                        * rhs.get_unchecked((i + 7, j)).inlined_clone()
                };
                i += 8;
            }

            res += acc0 + acc4;
            res += acc1 + acc5;
            res += acc2 + acc6;
            res += acc3 + acc7;

            for k in i..self.nrows() {
                res += unsafe {
                    conjugate(self.get_unchecked((k, j)).inlined_clone())
                        * rhs.get_unchecked((k, j)).inlined_clone()
                }
            }
        }

        res
    }

    /// The dot product between two vectors or matrices (seen as vectors).
    ///
    /// This is equal to `self.transpose() * rhs`. For the sesquilinear complex dot product, use
    /// `self.dotc(rhs)`.
    ///
    /// Note that this is **not** the matrix multiplication as in, e.g., numpy. For matrix
    /// multiplication, use one of: `.gemm`, `.mul_to`, `.mul`, the `*` operator.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Vector3, Matrix2x3};
    /// let vec1 = Vector3::new(1.0, 2.0, 3.0);
    /// let vec2 = Vector3::new(0.1, 0.2, 0.3);
    /// assert_eq!(vec1.dot(&vec2), 1.4);
    ///
    /// let mat1 = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                           4.0, 5.0, 6.0);
    /// let mat2 = Matrix2x3::new(0.1, 0.2, 0.3,
    ///                           0.4, 0.5, 0.6);
    /// assert_eq!(mat1.dot(&mat2), 9.1);
    /// ```
    ///
    #[inline]
    #[must_use]
    pub fn dot<R2: Dim, C2: Dim, SB>(&self, rhs: &Matrix<T, R2, C2, SB>) -> T
    where
        SB: Storage<T, R2, C2>,
        ShapeConstraint: DimEq<R, R2> + DimEq<C, C2>,
    {
        self.dotx(rhs, |e| e)
    }

    /// The conjugate-linear dot product between two vectors or matrices (seen as vectors).
    ///
    /// This is equal to `self.adjoint() * rhs`.
    /// For real vectors, this is identical to `self.dot(&rhs)`.
    /// Note that this is **not** the matrix multiplication as in, e.g., numpy. For matrix
    /// multiplication, use one of: `.gemm`, `.mul_to`, `.mul`, the `*` operator.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Vector2, Complex};
    /// let vec1 = Vector2::new(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0));
    /// let vec2 = Vector2::new(Complex::new(0.4, 0.3), Complex::new(0.2, 0.1));
    /// assert_eq!(vec1.dotc(&vec2), Complex::new(2.0, -1.0));
    ///
    /// // Note that for complex vectors, we generally have:
    /// // vec1.dotc(&vec2) != vec2.dot(&vec2)
    /// assert_ne!(vec1.dotc(&vec2), vec1.dot(&vec2));
    /// ```
    #[inline]
    #[must_use]
    pub fn dotc<R2: Dim, C2: Dim, SB>(&self, rhs: &Matrix<T, R2, C2, SB>) -> T
    where
        T: SimdComplexField,
        SB: Storage<T, R2, C2>,
        ShapeConstraint: DimEq<R, R2> + DimEq<C, C2>,
    {
        self.dotx(rhs, T::simd_conjugate)
    }

    /// The dot product between the transpose of `self` and `rhs`.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Vector3, RowVector3, Matrix2x3, Matrix3x2};
    /// let vec1 = Vector3::new(1.0, 2.0, 3.0);
    /// let vec2 = RowVector3::new(0.1, 0.2, 0.3);
    /// assert_eq!(vec1.tr_dot(&vec2), 1.4);
    ///
    /// let mat1 = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                           4.0, 5.0, 6.0);
    /// let mat2 = Matrix3x2::new(0.1, 0.4,
    ///                           0.2, 0.5,
    ///                           0.3, 0.6);
    /// assert_eq!(mat1.tr_dot(&mat2), 9.1);
    /// ```
    #[inline]
    #[must_use]
    pub fn tr_dot<R2: Dim, C2: Dim, SB>(&self, rhs: &Matrix<T, R2, C2, SB>) -> T
    where
        SB: Storage<T, R2, C2>,
        ShapeConstraint: DimEq<C, R2> + DimEq<R, C2>,
    {
        let (nrows, ncols) = self.shape();
        assert_eq!(
            (ncols, nrows),
            rhs.shape(),
            "Transposed dot product dimension mismatch."
        );

        let mut res = T::zero();

        for j in 0..self.nrows() {
            for i in 0..self.ncols() {
                res += unsafe {
                    self.get_unchecked((j, i)).inlined_clone()
                        * rhs.get_unchecked((i, j)).inlined_clone()
                }
            }
        }

        res
    }
}

/// # BLAS functions
impl<T, D: Dim, S> Vector<T, D, S>
where
    T: Scalar + Zero + ClosedAdd + ClosedMul,
    S: StorageMut<T, D>,
{
    /// Computes `self = a * x * c + b * self`, where `a`, `b`, `c` are scalars,
    /// and `x` is a vector of the same size as `self`.
    ///
    /// For commutative scalars, this is equivalent to an [`axpy`] call.
    ///
    /// If `b` is zero, `self` is never read from.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let mut vec1 = Vector3::new(1.0, 2.0, 3.0);
    /// let vec2 = Vector3::new(0.1, 0.2, 0.3);
    /// vec1.axcpy(5.0, &vec2, 2.0, 5.0);
    /// assert_eq!(vec1, Vector3::new(6.0, 12.0, 18.0));
    /// ```
    #[inline]
    #[allow(clippy::many_single_char_names)]
    pub fn axcpy<D2: Dim, SB>(&mut self, a: T, x: &Vector<T, D2, SB>, c: T, b: T)
    where
        SB: Storage<T, D2>,
        ShapeConstraint: DimEq<D, D2>,
    {
        assert_eq!(self.nrows(), x.nrows(), "Axcpy: mismatched vector shapes.");

        let rstride1 = self.strides().0;
        let rstride2 = x.strides().0;

        unsafe {
            // SAFETY: the conversion to slices is OK because we access the
            //         elements taking the strides into account.
            let y = self.data.as_mut_slice_unchecked();
            let x = x.data.as_slice_unchecked();

            if !b.is_zero() {
                for i in 0..x.len() {
                    let y = y.get_unchecked_mut(i * rstride1);
                    *y = a.inlined_clone()
                        * x.get_unchecked(i * rstride2).inlined_clone()
                        * c.inlined_clone()
                        + b.inlined_clone() * y.inlined_clone();
                }
            } else {
                for i in 0..x.len() {
                    let y = y.get_unchecked_mut(i * rstride1);
                    *y = a.inlined_clone()
                        * x.get_unchecked(i * rstride2).inlined_clone()
                        * c.inlined_clone();
                }
            }
        }
    }

    /// Computes `self = a * x + b * self`.
    ///
    /// If `b` is zero, `self` is never read from.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::Vector3;
    /// let mut vec1 = Vector3::new(1.0, 2.0, 3.0);
    /// let vec2 = Vector3::new(0.1, 0.2, 0.3);
    /// vec1.axpy(10.0, &vec2, 5.0);
    /// assert_eq!(vec1, Vector3::new(6.0, 12.0, 18.0));
    /// ```
    #[inline]
    pub fn axpy<D2: Dim, SB>(&mut self, a: T, x: &Vector<T, D2, SB>, b: T)
    where
        T: One,
        SB: Storage<T, D2>,
        ShapeConstraint: DimEq<D, D2>,
    {
        assert_eq!(self.nrows(), x.nrows(), "Axpy: mismatched vector shapes.");
        self.axcpy(a, x, T::one(), b)
    }

    /// Computes `self = alpha * a * x + beta * self`, where `a` is a matrix, `x` a vector, and
    /// `alpha, beta` two scalars.
    ///
    /// If `beta` is zero, `self` is never read.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// let mut vec1 = Vector2::new(1.0, 2.0);
    /// let vec2 = Vector2::new(0.1, 0.2);
    /// let mat = Matrix2::new(1.0, 2.0,
    ///                        3.0, 4.0);
    /// vec1.gemv(10.0, &mat, &vec2, 5.0);
    /// assert_eq!(vec1, Vector2::new(10.0, 21.0));
    /// ```
    #[inline]
    pub fn gemv<R2: Dim, C2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &Matrix<T, R2, C2, SB>,
        x: &Vector<T, D3, SC>,
        beta: T,
    ) where
        T: One,
        SB: Storage<T, R2, C2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<D, R2> + AreMultipliable<R2, C2, D3, U1>,
    {
        let dim1 = self.nrows();
        let (nrows2, ncols2) = a.shape();
        let dim3 = x.nrows();

        assert!(
            ncols2 == dim3 && dim1 == nrows2,
            "Gemv: dimensions mismatch."
        );

        if ncols2 == 0 {
            // NOTE: we can't just always multiply by beta
            // because we documented the guaranty that `self` is
            // never read if `beta` is zero.
            if beta.is_zero() {
                self.fill(T::zero());
            } else {
                *self *= beta;
            }
            return;
        }

        // TODO: avoid bound checks.
        let col2 = a.column(0);
        let val = unsafe { x.vget_unchecked(0).inlined_clone() };
        self.axcpy(alpha.inlined_clone(), &col2, val, beta);

        for j in 1..ncols2 {
            let col2 = a.column(j);
            let val = unsafe { x.vget_unchecked(j).inlined_clone() };

            self.axcpy(alpha.inlined_clone(), &col2, val, T::one());
        }
    }

    #[inline(always)]
    fn xxgemv<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &SquareMatrix<T, D2, SB>,
        x: &Vector<T, D3, SC>,
        beta: T,
        dot: impl Fn(
            &DVectorSlice<T, SB::RStride, SB::CStride>,
            &DVectorSlice<T, SC::RStride, SC::CStride>,
        ) -> T,
    ) where
        T: One,
        SB: Storage<T, D2, D2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<D, D2> + AreMultipliable<D2, D2, D3, U1>,
    {
        let dim1 = self.nrows();
        let dim2 = a.nrows();
        let dim3 = x.nrows();

        assert!(
            a.is_square(),
            "Symmetric cgemv: the input matrix must be square."
        );
        assert!(
            dim2 == dim3 && dim1 == dim2,
            "Symmetric cgemv: dimensions mismatch."
        );

        if dim2 == 0 {
            return;
        }

        // TODO: avoid bound checks.
        let col2 = a.column(0);
        let val = unsafe { x.vget_unchecked(0).inlined_clone() };
        self.axpy(alpha.inlined_clone() * val, &col2, beta);
        self[0] += alpha.inlined_clone() * dot(&a.slice_range(1.., 0), &x.rows_range(1..));

        for j in 1..dim2 {
            let col2 = a.column(j);
            let dot = dot(&col2.rows_range(j..), &x.rows_range(j..));

            let val;
            unsafe {
                val = x.vget_unchecked(j).inlined_clone();
                *self.vget_unchecked_mut(j) += alpha.inlined_clone() * dot;
            }
            self.rows_range_mut(j + 1..).axpy(
                alpha.inlined_clone() * val,
                &col2.rows_range(j + 1..),
                T::one(),
            );
        }
    }

    /// Computes `self = alpha * a * x + beta * self`, where `a` is a **symmetric** matrix, `x` a
    /// vector, and `alpha, beta` two scalars. DEPRECATED: use `sygemv` instead.
    #[inline]
    #[deprecated(note = "This is renamed `sygemv` to match the original BLAS terminology.")]
    pub fn gemv_symm<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &SquareMatrix<T, D2, SB>,
        x: &Vector<T, D3, SC>,
        beta: T,
    ) where
        T: One,
        SB: Storage<T, D2, D2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<D, D2> + AreMultipliable<D2, D2, D3, U1>,
    {
        self.sygemv(alpha, a, x, beta)
    }

    /// Computes `self = alpha * a * x + beta * self`, where `a` is a **symmetric** matrix, `x` a
    /// vector, and `alpha, beta` two scalars.
    ///
    /// For hermitian matrices, use `.hegemv` instead.
    /// If `beta` is zero, `self` is never read. If `self` is read, only its lower-triangular part
    /// (including the diagonal) is actually read.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// let mat = Matrix2::new(1.0, 2.0,
    ///                        2.0, 4.0);
    /// let mut vec1 = Vector2::new(1.0, 2.0);
    /// let vec2 = Vector2::new(0.1, 0.2);
    /// vec1.sygemv(10.0, &mat, &vec2, 5.0);
    /// assert_eq!(vec1, Vector2::new(10.0, 20.0));
    ///
    ///
    /// // The matrix upper-triangular elements can be garbage because it is never
    /// // read by this method. Therefore, it is not necessary for the caller to
    /// // fill the matrix struct upper-triangle.
    /// let mat = Matrix2::new(1.0, 9999999.9999999,
    ///                        2.0, 4.0);
    /// let mut vec1 = Vector2::new(1.0, 2.0);
    /// vec1.sygemv(10.0, &mat, &vec2, 5.0);
    /// assert_eq!(vec1, Vector2::new(10.0, 20.0));
    /// ```
    #[inline]
    pub fn sygemv<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &SquareMatrix<T, D2, SB>,
        x: &Vector<T, D3, SC>,
        beta: T,
    ) where
        T: One,
        SB: Storage<T, D2, D2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<D, D2> + AreMultipliable<D2, D2, D3, U1>,
    {
        self.xxgemv(alpha, a, x, beta, |a, b| a.dot(b))
    }

    /// Computes `self = alpha * a * x + beta * self`, where `a` is an **hermitian** matrix, `x` a
    /// vector, and `alpha, beta` two scalars.
    ///
    /// If `beta` is zero, `self` is never read. If `self` is read, only its lower-triangular part
    /// (including the diagonal) is actually read.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Vector2, Complex};
    /// let mat = Matrix2::new(Complex::new(1.0, 0.0), Complex::new(2.0, -0.1),
    ///                        Complex::new(2.0, 1.0), Complex::new(4.0, 0.0));
    /// let mut vec1 = Vector2::new(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0));
    /// let vec2 = Vector2::new(Complex::new(0.1, 0.2), Complex::new(0.3, 0.4));
    /// vec1.sygemv(Complex::new(10.0, 20.0), &mat, &vec2, Complex::new(5.0, 15.0));
    /// assert_eq!(vec1, Vector2::new(Complex::new(-48.0, 44.0), Complex::new(-75.0, 110.0)));
    ///
    ///
    /// // The matrix upper-triangular elements can be garbage because it is never
    /// // read by this method. Therefore, it is not necessary for the caller to
    /// // fill the matrix struct upper-triangle.
    ///
    /// let mat = Matrix2::new(Complex::new(1.0, 0.0), Complex::new(99999999.9, 999999999.9),
    ///                        Complex::new(2.0, 1.0), Complex::new(4.0, 0.0));
    /// let mut vec1 = Vector2::new(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0));
    /// let vec2 = Vector2::new(Complex::new(0.1, 0.2), Complex::new(0.3, 0.4));
    /// vec1.sygemv(Complex::new(10.0, 20.0), &mat, &vec2, Complex::new(5.0, 15.0));
    /// assert_eq!(vec1, Vector2::new(Complex::new(-48.0, 44.0), Complex::new(-75.0, 110.0)));
    /// ```
    #[inline]
    pub fn hegemv<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &SquareMatrix<T, D2, SB>,
        x: &Vector<T, D3, SC>,
        beta: T,
    ) where
        T: SimdComplexField,
        SB: Storage<T, D2, D2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<D, D2> + AreMultipliable<D2, D2, D3, U1>,
    {
        self.xxgemv(alpha, a, x, beta, |a, b| a.dotc(b))
    }

    #[inline(always)]
    fn gemv_xx<R2: Dim, C2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &Matrix<T, R2, C2, SB>,
        x: &Vector<T, D3, SC>,
        beta: T,
        dot: impl Fn(&VectorSlice<T, R2, SB::RStride, SB::CStride>, &Vector<T, D3, SC>) -> T,
    ) where
        T: One,
        SB: Storage<T, R2, C2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<D, C2> + AreMultipliable<C2, R2, D3, U1>,
    {
        let dim1 = self.nrows();
        let (nrows2, ncols2) = a.shape();
        let dim3 = x.nrows();

        assert!(
            nrows2 == dim3 && dim1 == ncols2,
            "Gemv: dimensions mismatch."
        );

        if ncols2 == 0 {
            return;
        }

        if beta.is_zero() {
            for j in 0..ncols2 {
                let val = unsafe { self.vget_unchecked_mut(j) };
                *val = alpha.inlined_clone() * dot(&a.column(j), x)
            }
        } else {
            for j in 0..ncols2 {
                let val = unsafe { self.vget_unchecked_mut(j) };
                *val = alpha.inlined_clone() * dot(&a.column(j), x)
                    + beta.inlined_clone() * val.inlined_clone();
            }
        }
    }

    /// Computes `self = alpha * a.transpose() * x + beta * self`, where `a` is a matrix, `x` a vector, and
    /// `alpha, beta` two scalars.
    ///
    /// If `beta` is zero, `self` is never read.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// let mat = Matrix2::new(1.0, 3.0,
    ///                        2.0, 4.0);
    /// let mut vec1 = Vector2::new(1.0, 2.0);
    /// let vec2 = Vector2::new(0.1, 0.2);
    /// let expected = mat.transpose() * vec2 * 10.0 + vec1 * 5.0;
    ///
    /// vec1.gemv_tr(10.0, &mat, &vec2, 5.0);
    /// assert_eq!(vec1, expected);
    /// ```
    #[inline]
    pub fn gemv_tr<R2: Dim, C2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &Matrix<T, R2, C2, SB>,
        x: &Vector<T, D3, SC>,
        beta: T,
    ) where
        T: One,
        SB: Storage<T, R2, C2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<D, C2> + AreMultipliable<C2, R2, D3, U1>,
    {
        self.gemv_xx(alpha, a, x, beta, |a, b| a.dot(b))
    }

    /// Computes `self = alpha * a.adjoint() * x + beta * self`, where `a` is a matrix, `x` a vector, and
    /// `alpha, beta` two scalars.
    ///
    /// For real matrices, this is the same as `.gemv_tr`.
    /// If `beta` is zero, `self` is never read.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Vector2, Complex};
    /// let mat = Matrix2::new(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0),
    ///                        Complex::new(5.0, 6.0), Complex::new(7.0, 8.0));
    /// let mut vec1 = Vector2::new(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0));
    /// let vec2 = Vector2::new(Complex::new(0.1, 0.2), Complex::new(0.3, 0.4));
    /// let expected = mat.adjoint() * vec2 * Complex::new(10.0, 20.0) + vec1 * Complex::new(5.0, 15.0);
    ///
    /// vec1.gemv_ad(Complex::new(10.0, 20.0), &mat, &vec2, Complex::new(5.0, 15.0));
    /// assert_eq!(vec1, expected);
    /// ```
    #[inline]
    pub fn gemv_ad<R2: Dim, C2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &Matrix<T, R2, C2, SB>,
        x: &Vector<T, D3, SC>,
        beta: T,
    ) where
        T: SimdComplexField,
        SB: Storage<T, R2, C2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<D, C2> + AreMultipliable<C2, R2, D3, U1>,
    {
        self.gemv_xx(alpha, a, x, beta, |a, b| a.dotc(b))
    }
}

impl<T, D: Dim, S> Vector<MaybeUninit<T>, D, S>
where
    T: Scalar + Zero + ClosedAdd + ClosedMul,
    S: StorageMut<MaybeUninit<T>, D>,
{
    /// Computes `alpha * a * x`, where `a` is a matrix, `x` a vector, and
    /// `alpha` is a scalar.
    ///
    /// `self` must be completely uninitialized, or data leaks will occur. After
    /// this method is called, all entries in `self` will be initialized.
    #[inline]
    pub fn axc<D2: Dim, S2>(
        &mut self,
        a: T,
        x: &Vector<T, D2, S2>,
        c: T,
    ) -> VectorSliceMut<T, D, S::RStride, S::CStride>
    where
        S2: Storage<T, D2>,
        ShapeConstraint: DimEq<D, D2>,
    {
        let rstride1 = self.strides().0;
        let rstride2 = x.strides().0;

        // Safety: see each individual remark.
        unsafe {
            // We don't mind `x` and `y` not being contiguous, as we'll only
            // access the elements we're allowed to. (TODO: double check this)
            let y = self.data.as_mut_slice_unchecked();
            let x = x.data.as_slice_unchecked();

            // The indices are within range, and only access elements that belong
            // to `x` and `y` themselves.
            for i in 0..y.len() {
                *y.get_unchecked_mut(i * rstride1) = MaybeUninit::new(
                    a.inlined_clone()
                        * x.get_unchecked(i * rstride2).inlined_clone()
                        * c.inlined_clone(),
                );
            }

            // We've initialized all elements.
            self.assume_init_mut()
        }
    }

    /// Computes `alpha * a * x`, where `a` is a matrix, `x` a vector, and
    /// `alpha` is a scalar.
    ///
    /// `self` must be completely uninitialized, or data leaks will occur. After
    /// the method is called, `self` will be completely initialized. We return
    /// an initialized mutable vector slice to `self` for convenience.
    #[inline]
    pub fn gemv_z<R2: Dim, C2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &Matrix<T, R2, C2, SB>,
        x: &Vector<T, D3, SC>,
    ) -> VectorSliceMut<T, D, S::RStride, S::CStride>
    where
        T: One,
        SB: Storage<T, R2, C2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<D, R2> + AreMultipliable<R2, C2, D3, U1>,
    {
        let dim1 = self.nrows();
        let (nrows2, ncols2) = a.shape();
        let dim3 = x.nrows();

        assert!(
            ncols2 == dim3 && dim1 == nrows2,
            "Gemv: dimensions mismatch."
        );

        if ncols2 == 0 {
            self.fill_fn(|| MaybeUninit::new(T::zero()));

            // Safety: all entries have just been initialized.
            unsafe {
                return self.assume_init_mut();
            }
        }

        // TODO: avoid bound checks.
        let col2 = a.column(0);
        let val = unsafe { x.vget_unchecked(0).inlined_clone() };
        let mut init = self.axc(alpha.inlined_clone(), &col2, val);

        // Safety: all indices are within range.
        unsafe {
            for j in 1..ncols2 {
                let col2 = a.column(j);
                let val = x.vget_unchecked(j).inlined_clone();
                init.axcpy(alpha.inlined_clone(), &col2, val, T::one());
            }
        }

        init
    }

    #[inline(always)]
    fn xxgemv_z<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &SquareMatrix<T, D2, SB>,
        x: &Vector<T, D3, SC>,
        dot: impl Fn(
            &DVectorSlice<T, SB::RStride, SB::CStride>,
            &DVectorSlice<T, SC::RStride, SC::CStride>,
        ) -> T,
    ) where
        T: One,
        SB: Storage<T, D2, D2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<D, D2> + AreMultipliable<D2, D2, D3, U1>,
    {
        let dim1 = self.nrows();
        let dim2 = a.nrows();
        let dim3 = x.nrows();

        assert!(
            a.is_square(),
            "Symmetric cgemv: the input matrix must be square."
        );
        assert!(
            dim2 == dim3 && dim1 == dim2,
            "Symmetric cgemv: dimensions mismatch."
        );

        if dim2 == 0 {
            return;
        }

        // TODO: avoid bound checks.
        let col2 = a.column(0);
        let val = unsafe { x.vget_unchecked(0).inlined_clone() };
        let mut res = self.axc(alpha.inlined_clone(), &col2, val);

        res[0] += alpha.inlined_clone() * dot(&a.slice_range(1.., 0), &x.rows_range(1..));

        for j in 1..dim2 {
            let col2 = a.column(j);
            let dot = dot(&col2.rows_range(j..), &x.rows_range(j..));

            let val;
            unsafe {
                val = x.vget_unchecked(j).inlined_clone();
                *res.vget_unchecked_mut(j) += alpha.inlined_clone() * dot;
            }
            res.rows_range_mut(j + 1..).axpy(
                alpha.inlined_clone() * val,
                &col2.rows_range(j + 1..),
                T::one(),
            );
        }
    }

    /// Computes `self = alpha * a * x`, where `a` is an **hermitian** matrix, `x` a
    /// vector, and `alpha, beta` two scalars.
    pub fn hegemv_z<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &SquareMatrix<T, D2, SB>,
        x: &Vector<T, D3, SC>,
    ) where
        T: SimdComplexField,
        SB: Storage<T, D2, D2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<D, D2> + AreMultipliable<D2, D2, D3, U1>,
    {
        self.xxgemv_z(alpha, a, x, |a, b| a.dotc(b))
    }
}

impl<T, R1: Dim, C1: Dim, S: StorageMut<MaybeUninit<T>, R1, C1>> Matrix<MaybeUninit<T>, R1, C1, S>
where
    T: Scalar + Zero + One + ClosedAdd + ClosedMul,
    // DefaultAllocator: Allocator<T, R1, C1>,
{
    /// Computes `alpha * a * b`, where `a` and `b` are matrices, and `alpha` is
    /// a scalar.
    ///
    /// # Examples:
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix2x3, Matrix3x4, Matrix2x4};
    /// let mut mat1 = Matrix2x4::identity();
    /// let mat2 = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                           4.0, 5.0, 6.0);
    /// let mat3 = Matrix3x4::new(0.1, 0.2, 0.3, 0.4,
    ///                           0.5, 0.6, 0.7, 0.8,
    ///                           0.9, 1.0, 1.1, 1.2);
    /// let expected = mat2 * mat3 * 10.0 + mat1 * 5.0;
    ///
    /// mat1.gemm(10.0, &mat2, &mat3, 5.0);
    /// assert_relative_eq!(mat1, expected);
    /// ```
    #[inline]
    pub fn gemm_z<R2: Dim, C2: Dim, R3: Dim, C3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &Matrix<T, R2, C2, SB>,
        b: &Matrix<T, R3, C3, SC>,
    ) -> MatrixSliceMut<T, R1, C1, S::RStride, S::CStride>
    where
        SB: Storage<T, R2, C2>,
        SC: Storage<T, R3, C3>,
        ShapeConstraint: SameNumberOfRows<R1, R2>
            + SameNumberOfColumns<C1, C3>
            + AreMultipliable<R2, C2, R3, C3>,
    {
        let ncols1 = self.ncols();

        #[cfg(feature = "std")]
        {
            // We assume large matrices will be Dynamic but small matrices static.
            // We could use matrixmultiply for large statically-sized matrices but the performance
            // threshold to activate it would be different from SMALL_DIM because our code optimizes
            // better for statically-sized matrices.
            if R1::is::<Dynamic>()
                || C1::is::<Dynamic>()
                || R2::is::<Dynamic>()
                || C2::is::<Dynamic>()
                || R3::is::<Dynamic>()
                || C3::is::<Dynamic>()
            {
                // matrixmultiply can be used only if the std feature is available.
                let nrows1 = self.nrows();
                let (nrows2, ncols2) = a.shape();
                let (nrows3, ncols3) = b.shape();

                // Threshold determined empirically.
                const SMALL_DIM: usize = 5;

                if nrows1 > SMALL_DIM
                    && ncols1 > SMALL_DIM
                    && nrows2 > SMALL_DIM
                    && ncols2 > SMALL_DIM
                {
                    assert_eq!(
                        ncols2, nrows3,
                        "gemm: dimensions mismatch for multiplication."
                    );
                    assert_eq!(
                        (nrows1, ncols1),
                        (nrows2, ncols3),
                        "gemm: dimensions mismatch for addition."
                    );

                    // NOTE: this case should never happen because we enter this
                    // codepath only when ncols2 > SMALL_DIM. Though we keep this
                    // here just in case if in the future we change the conditions to
                    // enter this codepath.
                    if ncols1 == 0 {
                        self.fill_fn(|| MaybeUninit::new(T::zero()));

                        // Safety: there's no (uninitialized) values.
                        return unsafe { self.assume_init_mut() };
                    }

                    let (rsa, csa) = a.strides();
                    let (rsb, csb) = b.strides();
                    let (rsc, csc) = self.strides();

                    if T::is::<f32>() {
                        unsafe {
                            matrixmultiply::sgemm(
                                nrows2,
                                ncols2,
                                ncols3,
                                mem::transmute_copy(&alpha),
                                a.data.ptr() as *const f32,
                                rsa as isize,
                                csa as isize,
                                b.data.ptr() as *const f32,
                                rsb as isize,
                                csb as isize,
                                0.0,
                                self.data.ptr_mut() as *mut f32,
                                rsc as isize,
                                csc as isize,
                            );
                        }
                    } else if T::is::<f64>() {
                        unsafe {
                            matrixmultiply::dgemm(
                                nrows2,
                                ncols2,
                                ncols3,
                                mem::transmute_copy(&alpha),
                                a.data.ptr() as *const f64,
                                rsa as isize,
                                csa as isize,
                                b.data.ptr() as *const f64,
                                rsb as isize,
                                csb as isize,
                                0.0,
                                self.data.ptr_mut() as *mut f64,
                                rsc as isize,
                                csc as isize,
                            );
                        }
                    }

                    // Safety: all entries have been initialized.
                    unsafe {
                        return self.assume_init_mut();
                    }
                }
            }
        }

        for j1 in 0..ncols1 {
            // TODO: avoid bound checks.
            let _ = self
                .column_mut(j1)
                .gemv_z(alpha.inlined_clone(), a, &b.column(j1));
        }

        // Safety: all entries have been initialized.
        unsafe { self.assume_init_mut() }
    }
}

impl<T, R1: Dim, C1: Dim, S: StorageMut<T, R1, C1>> Matrix<T, R1, C1, S>
where
    T: Scalar + Zero + ClosedAdd + ClosedMul,
{
    #[inline(always)]
    fn gerx<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        x: &Vector<T, D2, SB>,
        y: &Vector<T, D3, SC>,
        beta: T,
        conjugate: impl Fn(T) -> T,
    ) where
        T: One,
        SB: Storage<T, D2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<R1, D2> + DimEq<C1, D3>,
    {
        let (nrows1, ncols1) = self.shape();
        let dim2 = x.nrows();
        let dim3 = y.nrows();

        assert!(
            nrows1 == dim2 && ncols1 == dim3,
            "ger: dimensions mismatch."
        );

        for j in 0..ncols1 {
            // TODO: avoid bound checks.
            let val = unsafe { conjugate(y.vget_unchecked(j).inlined_clone()) };
            self.column_mut(j)
                .axpy(alpha.inlined_clone() * val, x, beta.inlined_clone());
        }
    }

    /// Computes `self = alpha * x * y.transpose() + beta * self`.
    ///
    /// If `beta` is zero, `self` is never read.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Matrix2x3, Vector2, Vector3};
    /// let mut mat = Matrix2x3::repeat(4.0);
    /// let vec1 = Vector2::new(1.0, 2.0);
    /// let vec2 = Vector3::new(0.1, 0.2, 0.3);
    /// let expected = vec1 * vec2.transpose() * 10.0 + mat * 5.0;
    ///
    /// mat.ger(10.0, &vec1, &vec2, 5.0);
    /// assert_eq!(mat, expected);
    /// ```
    #[inline]
    pub fn ger<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        x: &Vector<T, D2, SB>,
        y: &Vector<T, D3, SC>,
        beta: T,
    ) where
        T: One,
        SB: Storage<T, D2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<R1, D2> + DimEq<C1, D3>,
    {
        self.gerx(alpha, x, y, beta, |e| e)
    }

    /// Computes `self = alpha * x * y.adjoint() + beta * self`.
    ///
    /// If `beta` is zero, `self` is never read.
    ///
    /// # Examples:
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix2x3, Vector2, Vector3, Complex};
    /// let mut mat = Matrix2x3::repeat(Complex::new(4.0, 5.0));
    /// let vec1 = Vector2::new(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0));
    /// let vec2 = Vector3::new(Complex::new(0.6, 0.5), Complex::new(0.4, 0.5), Complex::new(0.2, 0.1));
    /// let expected = vec1 * vec2.adjoint() * Complex::new(10.0, 20.0) + mat * Complex::new(5.0, 15.0);
    ///
    /// mat.gerc(Complex::new(10.0, 20.0), &vec1, &vec2, Complex::new(5.0, 15.0));
    /// assert_eq!(mat, expected);
    /// ```
    #[inline]
    pub fn gerc<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        x: &Vector<T, D2, SB>,
        y: &Vector<T, D3, SC>,
        beta: T,
    ) where
        T: SimdComplexField,
        SB: Storage<T, D2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<R1, D2> + DimEq<C1, D3>,
    {
        self.gerx(alpha, x, y, beta, SimdComplexField::simd_conjugate)
    }

    /// Computes `self = alpha * a * b + beta * self`, where `a, b, self` are matrices.
    /// `alpha` and `beta` are scalar.
    ///
    /// If `beta` is zero, `self` is never read.
    ///
    /// # Examples:
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix2x3, Matrix3x4, Matrix2x4};
    /// let mut mat1 = Matrix2x4::identity();
    /// let mat2 = Matrix2x3::new(1.0, 2.0, 3.0,
    ///                           4.0, 5.0, 6.0);
    /// let mat3 = Matrix3x4::new(0.1, 0.2, 0.3, 0.4,
    ///                           0.5, 0.6, 0.7, 0.8,
    ///                           0.9, 1.0, 1.1, 1.2);
    /// let expected = mat2 * mat3 * 10.0 + mat1 * 5.0;
    ///
    /// mat1.gemm(10.0, &mat2, &mat3, 5.0);
    /// assert_relative_eq!(mat1, expected);
    /// ```
    #[inline]
    pub fn gemm<R2: Dim, C2: Dim, R3: Dim, C3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &Matrix<T, R2, C2, SB>,
        b: &Matrix<T, R3, C3, SC>,
        beta: T,
    ) where
        T: One,
        SB: Storage<T, R2, C2>,
        SC: Storage<T, R3, C3>,
        ShapeConstraint: SameNumberOfRows<R1, R2>
            + SameNumberOfColumns<C1, C3>
            + AreMultipliable<R2, C2, R3, C3>,
    {
        let ncols1 = self.ncols();

        #[cfg(feature = "std")]
        {
            // We assume large matrices will be Dynamic but small matrices static.
            // We could use matrixmultiply for large statically-sized matrices but the performance
            // threshold to activate it would be different from SMALL_DIM because our code optimizes
            // better for statically-sized matrices.
            if R1::is::<Dynamic>()
                || C1::is::<Dynamic>()
                || R2::is::<Dynamic>()
                || C2::is::<Dynamic>()
                || R3::is::<Dynamic>()
                || C3::is::<Dynamic>()
            {
                // matrixmultiply can be used only if the std feature is available.
                let nrows1 = self.nrows();
                let (nrows2, ncols2) = a.shape();
                let (nrows3, ncols3) = b.shape();

                // Threshold determined empirically.
                const SMALL_DIM: usize = 5;

                if nrows1 > SMALL_DIM
                    && ncols1 > SMALL_DIM
                    && nrows2 > SMALL_DIM
                    && ncols2 > SMALL_DIM
                {
                    assert_eq!(
                        ncols2, nrows3,
                        "gemm: dimensions mismatch for multiplication."
                    );
                    assert_eq!(
                        (nrows1, ncols1),
                        (nrows2, ncols3),
                        "gemm: dimensions mismatch for addition."
                    );

                    // NOTE: this case should never happen because we enter this
                    // codepath only when ncols2 > SMALL_DIM. Though we keep this
                    // here just in case if in the future we change the conditions to
                    // enter this codepath.
                    if ncols2 == 0 {
                        // NOTE: we can't just always multiply by beta
                        // because we documented the guaranty that `self` is
                        // never read if `beta` is zero.
                        if beta.is_zero() {
                            self.fill(T::zero());
                        } else {
                            *self *= beta;
                        }
                        return;
                    }

                    if T::is::<f32>() {
                        let (rsa, csa) = a.strides();
                        let (rsb, csb) = b.strides();
                        let (rsc, csc) = self.strides();

                        unsafe {
                            matrixmultiply::sgemm(
                                nrows2,
                                ncols2,
                                ncols3,
                                mem::transmute_copy(&alpha),
                                a.data.ptr() as *const f32,
                                rsa as isize,
                                csa as isize,
                                b.data.ptr() as *const f32,
                                rsb as isize,
                                csb as isize,
                                mem::transmute_copy(&beta),
                                self.data.ptr_mut() as *mut f32,
                                rsc as isize,
                                csc as isize,
                            );
                        }
                        return;
                    } else if T::is::<f64>() {
                        let (rsa, csa) = a.strides();
                        let (rsb, csb) = b.strides();
                        let (rsc, csc) = self.strides();

                        unsafe {
                            matrixmultiply::dgemm(
                                nrows2,
                                ncols2,
                                ncols3,
                                mem::transmute_copy(&alpha),
                                a.data.ptr() as *const f64,
                                rsa as isize,
                                csa as isize,
                                b.data.ptr() as *const f64,
                                rsb as isize,
                                csb as isize,
                                mem::transmute_copy(&beta),
                                self.data.ptr_mut() as *mut f64,
                                rsc as isize,
                                csc as isize,
                            );
                        }
                        return;
                    }
                }
            }
        }

        for j1 in 0..ncols1 {
            // TODO: avoid bound checks.
            self.column_mut(j1).gemv(
                alpha.inlined_clone(),
                a,
                &b.column(j1),
                beta.inlined_clone(),
            );
        }
    }

    /// Computes `self = alpha * a.transpose() * b + beta * self`, where `a, b, self` are matrices.
    /// `alpha` and `beta` are scalar.
    ///
    /// If `beta` is zero, `self` is never read.
    ///
    /// # Examples:
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix3x2, Matrix3x4, Matrix2x4};
    /// let mut mat1 = Matrix2x4::identity();
    /// let mat2 = Matrix3x2::new(1.0, 4.0,
    ///                           2.0, 5.0,
    ///                           3.0, 6.0);
    /// let mat3 = Matrix3x4::new(0.1, 0.2, 0.3, 0.4,
    ///                           0.5, 0.6, 0.7, 0.8,
    ///                           0.9, 1.0, 1.1, 1.2);
    /// let expected = mat2.transpose() * mat3 * 10.0 + mat1 * 5.0;
    ///
    /// mat1.gemm_tr(10.0, &mat2, &mat3, 5.0);
    /// assert_eq!(mat1, expected);
    /// ```
    #[inline]
    pub fn gemm_tr<R2: Dim, C2: Dim, R3: Dim, C3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &Matrix<T, R2, C2, SB>,
        b: &Matrix<T, R3, C3, SC>,
        beta: T,
    ) where
        T: One,
        SB: Storage<T, R2, C2>,
        SC: Storage<T, R3, C3>,
        ShapeConstraint: SameNumberOfRows<R1, C2>
            + SameNumberOfColumns<C1, C3>
            + AreMultipliable<C2, R2, R3, C3>,
    {
        let (nrows1, ncols1) = self.shape();
        let (nrows2, ncols2) = a.shape();
        let (nrows3, ncols3) = b.shape();

        assert_eq!(
            nrows2, nrows3,
            "gemm: dimensions mismatch for multiplication."
        );
        assert_eq!(
            (nrows1, ncols1),
            (ncols2, ncols3),
            "gemm: dimensions mismatch for addition."
        );

        for j1 in 0..ncols1 {
            // TODO: avoid bound checks.
            self.column_mut(j1).gemv_tr(
                alpha.inlined_clone(),
                a,
                &b.column(j1),
                beta.inlined_clone(),
            );
        }
    }

    /// Computes `self = alpha * a.adjoint() * b + beta * self`, where `a, b, self` are matrices.
    /// `alpha` and `beta` are scalar.
    ///
    /// If `beta` is zero, `self` is never read.
    ///
    /// # Examples:
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{Matrix3x2, Matrix3x4, Matrix2x4, Complex};
    /// let mut mat1 = Matrix2x4::identity();
    /// let mat2 = Matrix3x2::new(Complex::new(1.0, 4.0), Complex::new(7.0, 8.0),
    ///                           Complex::new(2.0, 5.0), Complex::new(9.0, 10.0),
    ///                           Complex::new(3.0, 6.0), Complex::new(11.0, 12.0));
    /// let mat3 = Matrix3x4::new(Complex::new(0.1, 1.3), Complex::new(0.2, 1.4), Complex::new(0.3, 1.5), Complex::new(0.4, 1.6),
    ///                           Complex::new(0.5, 1.7), Complex::new(0.6, 1.8), Complex::new(0.7, 1.9), Complex::new(0.8, 2.0),
    ///                           Complex::new(0.9, 2.1), Complex::new(1.0, 2.2), Complex::new(1.1, 2.3), Complex::new(1.2, 2.4));
    /// let expected = mat2.adjoint() * mat3 * Complex::new(10.0, 20.0) + mat1 * Complex::new(5.0, 15.0);
    ///
    /// mat1.gemm_ad(Complex::new(10.0, 20.0), &mat2, &mat3, Complex::new(5.0, 15.0));
    /// assert_eq!(mat1, expected);
    /// ```
    #[inline]
    pub fn gemm_ad<R2: Dim, C2: Dim, R3: Dim, C3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        a: &Matrix<T, R2, C2, SB>,
        b: &Matrix<T, R3, C3, SC>,
        beta: T,
    ) where
        T: SimdComplexField,
        SB: Storage<T, R2, C2>,
        SC: Storage<T, R3, C3>,
        ShapeConstraint: SameNumberOfRows<R1, C2>
            + SameNumberOfColumns<C1, C3>
            + AreMultipliable<C2, R2, R3, C3>,
    {
        let (nrows1, ncols1) = self.shape();
        let (nrows2, ncols2) = a.shape();
        let (nrows3, ncols3) = b.shape();

        assert_eq!(
            nrows2, nrows3,
            "gemm: dimensions mismatch for multiplication."
        );
        assert_eq!(
            (nrows1, ncols1),
            (ncols2, ncols3),
            "gemm: dimensions mismatch for addition."
        );

        for j1 in 0..ncols1 {
            // TODO: avoid bound checks.
            self.column_mut(j1).gemv_ad(alpha, a, &b.column(j1), beta);
        }
    }
}

impl<T, R1: Dim, C1: Dim, S: StorageMut<T, R1, C1>> Matrix<T, R1, C1, S>
where
    T: Scalar + Zero + ClosedAdd + ClosedMul,
{
    #[inline(always)]
    fn xxgerx<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        x: &Vector<T, D2, SB>,
        y: &Vector<T, D3, SC>,
        beta: T,
        conjugate: impl Fn(T) -> T,
    ) where
        T: One,
        SB: Storage<T, D2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<R1, D2> + DimEq<C1, D3>,
    {
        let dim1 = self.nrows();
        let dim2 = x.nrows();
        let dim3 = y.nrows();

        assert!(
            self.is_square(),
            "Symmetric ger: the input matrix must be square."
        );
        assert!(dim1 == dim2 && dim1 == dim3, "ger: dimensions mismatch.");

        for j in 0..dim1 {
            let val = unsafe { conjugate(y.vget_unchecked(j).inlined_clone()) };
            let subdim = Dynamic::new(dim1 - j);
            // TODO: avoid bound checks.
            self.generic_slice_mut((j, j), (subdim, Const::<1>)).axpy(
                alpha.inlined_clone() * val,
                &x.rows_range(j..),
                beta.inlined_clone(),
            );
        }
    }

    /// Computes `self = alpha * x * y.transpose() + beta * self`, where `self` is a **symmetric**
    /// matrix.
    ///
    /// If `beta` is zero, `self` is never read. The result is symmetric. Only the lower-triangular
    /// (including the diagonal) part of `self` is read/written.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// let mut mat = Matrix2::identity();
    /// let vec1 = Vector2::new(1.0, 2.0);
    /// let vec2 = Vector2::new(0.1, 0.2);
    /// let expected = vec1 * vec2.transpose() * 10.0 + mat * 5.0;
    /// mat.m12 = 99999.99999; // This component is on the upper-triangular part and will not be read/written.
    ///
    /// mat.ger_symm(10.0, &vec1, &vec2, 5.0);
    /// assert_eq!(mat.lower_triangle(), expected.lower_triangle());
    /// assert_eq!(mat.m12, 99999.99999); // This was untouched.
    #[inline]
    #[deprecated(note = "This is renamed `syger` to match the original BLAS terminology.")]
    pub fn ger_symm<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        x: &Vector<T, D2, SB>,
        y: &Vector<T, D3, SC>,
        beta: T,
    ) where
        T: One,
        SB: Storage<T, D2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<R1, D2> + DimEq<C1, D3>,
    {
        self.syger(alpha, x, y, beta)
    }

    /// Computes `self = alpha * x * y.transpose() + beta * self`, where `self` is a **symmetric**
    /// matrix.
    ///
    /// For hermitian complex matrices, use `.hegerc` instead.
    /// If `beta` is zero, `self` is never read. The result is symmetric. Only the lower-triangular
    /// (including the diagonal) part of `self` is read/written.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Vector2};
    /// let mut mat = Matrix2::identity();
    /// let vec1 = Vector2::new(1.0, 2.0);
    /// let vec2 = Vector2::new(0.1, 0.2);
    /// let expected = vec1 * vec2.transpose() * 10.0 + mat * 5.0;
    /// mat.m12 = 99999.99999; // This component is on the upper-triangular part and will not be read/written.
    ///
    /// mat.syger(10.0, &vec1, &vec2, 5.0);
    /// assert_eq!(mat.lower_triangle(), expected.lower_triangle());
    /// assert_eq!(mat.m12, 99999.99999); // This was untouched.
    #[inline]
    pub fn syger<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        x: &Vector<T, D2, SB>,
        y: &Vector<T, D3, SC>,
        beta: T,
    ) where
        T: One,
        SB: Storage<T, D2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<R1, D2> + DimEq<C1, D3>,
    {
        self.xxgerx(alpha, x, y, beta, |e| e)
    }

    /// Computes `self = alpha * x * y.adjoint() + beta * self`, where `self` is an **hermitian**
    /// matrix.
    ///
    /// If `beta` is zero, `self` is never read. The result is symmetric. Only the lower-triangular
    /// (including the diagonal) part of `self` is read/written.
    ///
    /// # Examples:
    ///
    /// ```
    /// # use nalgebra::{Matrix2, Vector2, Complex};
    /// let mut mat = Matrix2::identity();
    /// let vec1 = Vector2::new(Complex::new(1.0, 3.0), Complex::new(2.0, 4.0));
    /// let vec2 = Vector2::new(Complex::new(0.2, 0.4), Complex::new(0.1, 0.3));
    /// let expected = vec1 * vec2.adjoint() * Complex::new(10.0, 20.0) + mat * Complex::new(5.0, 15.0);
    /// mat.m12 = Complex::new(99999.99999, 88888.88888); // This component is on the upper-triangular part and will not be read/written.
    ///
    /// mat.hegerc(Complex::new(10.0, 20.0), &vec1, &vec2, Complex::new(5.0, 15.0));
    /// assert_eq!(mat.lower_triangle(), expected.lower_triangle());
    /// assert_eq!(mat.m12, Complex::new(99999.99999, 88888.88888)); // This was untouched.
    #[inline]
    pub fn hegerc<D2: Dim, D3: Dim, SB, SC>(
        &mut self,
        alpha: T,
        x: &Vector<T, D2, SB>,
        y: &Vector<T, D3, SC>,
        beta: T,
    ) where
        T: SimdComplexField,
        SB: Storage<T, D2>,
        SC: Storage<T, D3>,
        ShapeConstraint: DimEq<R1, D2> + DimEq<C1, D3>,
    {
        self.xxgerx(alpha, x, y, beta, SimdComplexField::simd_conjugate)
    }
}

impl<T, D1: Dim, S: StorageMut<T, D1, D1>> SquareMatrix<T, D1, S>
where
    T: Scalar + Zero + One + ClosedAdd + ClosedMul,
{
    /// Computes the quadratic form `self = alpha * lhs * mid * lhs.transpose() + beta * self`.
    ///
    /// This uses the provided workspace `work` to avoid allocations for intermediate results.
    ///
    /// # Examples:
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{DMatrix, DVector};
    /// // Note that all those would also work with statically-sized matrices.
    /// // We use DMatrix/DVector since that's the only case where pre-allocating the
    /// // workspace is actually useful (assuming the same workspace is re-used for
    /// // several computations) because it avoids repeated dynamic allocations.
    /// let mut mat = DMatrix::identity(2, 2);
    /// let lhs = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0,
    ///                                           4.0, 5.0, 6.0]);
    /// let mid = DMatrix::from_row_slice(3, 3, &[0.1, 0.2, 0.3,
    ///                                           0.5, 0.6, 0.7,
    ///                                           0.9, 1.0, 1.1]);
    ///
    /// let expected = &lhs * &mid * lhs.transpose() * 10.0 + &mat * 5.0;
    ///
    /// mat.quadform_tr(10.0, &lhs, &mid, 5.0);
    /// assert_relative_eq!(mat, expected);
    pub fn quadform_tr<R3: Dim, C3: Dim, S3, D4: Dim, S4>(
        &mut self,
        alpha: T,
        lhs: &Matrix<T, R3, C3, S3>,
        mid: &SquareMatrix<T, D4, S4>,
        beta: T,
    ) where
        S3: Storage<T, R3, C3>,
        S4: Storage<T, D4, D4>,
        ShapeConstraint: DimEq<D1, R3> + DimEq<C3, D4>,
        DefaultAllocator: Allocator<T, R3>,
    {
        let mut work =
            Matrix::new_uninitialized_generic(R3::from_usize(self.shape().0), Const::<1>);
        let mut work = work.gemv_z(T::one(), lhs, &mid.column(0));

        self.ger(alpha.inlined_clone(), &work, &lhs.column(0), beta);

        for j in 1..mid.ncols() {
            work.gemv(T::one(), lhs, &mid.column(j), T::zero());
            self.ger(alpha.inlined_clone(), &work, &lhs.column(j), T::one());
        }
    }

    /// Computes the quadratic form `self = alpha * rhs.transpose() * mid * rhs + beta * self`.
    ///
    /// This uses the provided workspace `work` to avoid allocations for intermediate results.
    ///
    /// ```
    /// # #[macro_use] extern crate approx;
    /// # use nalgebra::{DMatrix, DVector};
    /// // Note that all those would also work with statically-sized matrices.
    /// // We use DMatrix/DVector since that's the only case where pre-allocating the
    /// // workspace is actually useful (assuming the same workspace is re-used for
    /// // several computations) because it avoids repeated dynamic allocations.
    /// let mut mat = DMatrix::identity(2, 2);
    /// let rhs = DMatrix::from_row_slice(3, 2, &[1.0, 2.0,
    ///                                           3.0, 4.0,
    ///                                           5.0, 6.0]);
    /// let mid = DMatrix::from_row_slice(3, 3, &[0.1, 0.2, 0.3,
    ///                                           0.5, 0.6, 0.7,
    ///                                           0.9, 1.0, 1.1]);
    ///
    /// let expected = rhs.transpose() * &mid * &rhs * 10.0 + &mat * 5.0;
    ///
    /// mat.quadform(10.0, &mid, &rhs, 5.0);
    /// assert_relative_eq!(mat, expected);
    pub fn quadform<D3: Dim, S3, R4: Dim, C4: Dim, S4>(
        &mut self,
        alpha: T,
        mid: &SquareMatrix<T, D3, S3>,
        rhs: &Matrix<T, R4, C4, S4>,
        beta: T,
    ) where
        S3: Storage<T, D3, D3>,
        S4: Storage<T, R4, C4>,
        ShapeConstraint: DimEq<D3, R4> + DimEq<R4, D3> + DimEq<D1, C4>,
        DefaultAllocator: Allocator<T, D3>,
    {
        // TODO: figure out why type inference isn't doing its job.
        let mut work = Matrix::new_uninitialized_generic(D3::from_usize(mid.shape().0), Const::<1>);
        let mut work = work.gemv_z::<D3, _, _, _, _>(T::one(), mid, &rhs.column(0));

        self.column_mut(0)
            .gemv_tr(alpha.inlined_clone(), rhs, &work, beta.inlined_clone());

        for j in 1..rhs.ncols() {
            work.gemv::<D3, D3, R4, S3, _>(T::one(), mid, &rhs.column(j), T::zero());
            self.column_mut(j)
                .gemv_tr(alpha.inlined_clone(), rhs, &work, beta.inlined_clone());
        }
    }
}
