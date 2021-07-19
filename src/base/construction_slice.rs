use crate::base::dimension::{Const, Dim, DimName, Dynamic};
use crate::base::matrix_slice::{SliceStorage, SliceStorageMut};
use crate::base::{MatrixSlice, MatrixSliceMutMN};

use num_rational::Ratio;

/// # Creating matrix slices from `&[T]`
impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> MatrixSlice<'a, T, R, C, RStride, CStride> {
    /// Creates, without bound-checking, a matrix slice from an array and with dimensions and strides specified by generic types instances.
    ///
    /// # Safety
    /// This method is unsafe because the input data array is not checked to contain enough elements.
    /// The generic types `R`, `C`, `RStride`, `CStride` can either be type-level integers or integers wrapped with `Dynamic::new()`.
    #[inline]
    pub unsafe fn from_slice_with_strides_generic_unchecked(
        data: &'a [T],
        start: usize,
        nrows: R,
        ncols: C,
        rstride: RStride,
        cstride: CStride,
    ) -> Self {
        let data = SliceStorage::from_raw_parts(
            data.as_ptr().add(start),
            (nrows, ncols),
            (rstride, cstride),
        );
        Self::from_data(data)
    }

    /// Creates a matrix slice from an array and with dimensions and strides specified by generic types instances.
    ///
    /// Panics if the input data array dose not contain enough elements.
    /// The generic types `R`, `C`, `RStride`, `CStride` can either be type-level integers or integers wrapped with `Dynamic::new()`.
    #[inline]
    pub fn from_slice_with_strides_generic(
        data: &'a [T],
        nrows: R,
        ncols: C,
        rstride: RStride,
        cstride: CStride,
    ) -> Self {
        // NOTE: The assertion implements the following formula, but without subtractions to avoid
        // underflow panics:
        //      len >= (ncols - 1) * cstride + (nrows - 1) * rstride + 1
        assert!(
            data.len() + cstride.value() + rstride.value()
                >= ncols.value() * cstride.value() + nrows.value() * rstride.value() + 1,
            "Matrix slice: input data buffer to small."
        );

        unsafe {
            Self::from_slice_with_strides_generic_unchecked(data, 0, nrows, ncols, rstride, cstride)
        }
    }
}

impl<'a, T, R: Dim, C: Dim> MatrixSlice<'a, T, R, C> {
    /// Creates, without bound-checking, a matrix slice from an array and with dimensions specified by generic types instances.
    ///
    /// # Safety
    /// This method is unsafe because the input data array is not checked to contain enough elements.
    /// The generic types `R` and `C` can either be type-level integers or integers wrapped with `Dynamic::new()`.
    #[inline]
    pub unsafe fn from_slice_generic_unchecked(
        data: &'a [T],
        start: usize,
        nrows: R,
        ncols: C,
    ) -> Self {
        Self::from_slice_with_strides_generic_unchecked(
            data, start, nrows, ncols, Const::<1>, nrows,
        )
    }

    /// Creates a matrix slice from an array and with dimensions and strides specified by generic types instances.
    ///
    /// Panics if the input data array dose not contain enough elements.
    /// The generic types `R` and `C` can either be type-level integers or integers wrapped with `Dynamic::new()`.
    #[inline]
    pub fn from_slice_generic(data: &'a [T], nrows: R, ncols: C) -> Self {
        Self::from_slice_with_strides_generic(data, nrows, ncols, Const::<1>, nrows)
    }
}

macro_rules! impl_constructors(
    ($($Dims: ty),*; $(=> $DimIdent: ident: $DimBound: ident),*; $($gargs: expr),*; $($args: ident),*) => {
        impl<'a, T, $($DimIdent: $DimBound),*> MatrixSlice<'a, T, $($Dims),*> {
            /// Creates a new matrix slice from the given data array.
            ///
            /// Panics if `data` does not contain enough elements.
            #[inline]
            pub fn from_slice(data: &'a [T], $($args: usize),*) -> Self {
                Self::from_slice_generic(data, $($gargs),*)
            }

            /// Creates, without bound checking, a new matrix slice from the given data array.
            #[inline]
            pub unsafe fn from_slice_unchecked(data: &'a [T], start: usize, $($args: usize),*) -> Self {
                Self::from_slice_generic_unchecked(data, start, $($gargs),*)
            }
        }

        impl<'a, T, $($DimIdent: $DimBound, )*> MatrixSlice<'a, T, $($Dims,)* Dynamic, Dynamic> {
            /// Creates a new matrix slice with the specified strides from the given data array.
            ///
            /// Panics if `data` does not contain enough elements.
            #[inline]
            pub fn from_slice_with_strides(data: &'a [T], $($args: usize,)* rstride: usize, cstride: usize) -> Self {
                Self::from_slice_with_strides_generic(data, $($gargs,)* Dynamic::new(rstride), Dynamic::new(cstride))
            }

            /// Creates, without bound checking, a new matrix slice with the specified strides from the given data array.
            #[inline]
            pub unsafe fn from_slice_with_strides_unchecked(data: &'a [T], start: usize, $($args: usize,)* rstride: usize, cstride: usize) -> Self {
                Self::from_slice_with_strides_generic_unchecked(data, start, $($gargs,)* Dynamic::new(rstride), Dynamic::new(cstride))
            }
        }
    }
);

// TODO: this is not very pretty. We could find a better call syntax.
impl_constructors!(R, C;                         // Arguments for Matrix<T, ..., S>
=> R: DimName, => C: DimName; // Type parameters for impl<T, ..., S>
R::name(), C::name();         // Arguments for `_generic` constructors.
); // Arguments for non-generic constructors.

impl_constructors!(R, Dynamic;
                   => R: DimName;
                   R::name(), Dynamic::new(ncols);
                   ncols);

impl_constructors!(Dynamic, C;
                   => C: DimName;
                   Dynamic::new(nrows), C::name();
                   nrows);

impl_constructors!(Dynamic, Dynamic;
                   ;
                   Dynamic::new(nrows), Dynamic::new(ncols);
                   nrows, ncols);

/// # Creating mutable matrix slices from `&mut [T]`
impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim>
    MatrixSliceMutMN<'a, T, R, C, RStride, CStride>
{
    /// Creates, without bound-checking, a mutable matrix slice from an array and with dimensions and strides specified by generic types instances.
    ///
    /// # Safety
    /// This method is unsafe because the input data array is not checked to contain enough elements.
    /// The generic types `R`, `C`, `RStride`, `CStride` can either be type-level integers or integers wrapped with `Dynamic::new()`.
    #[inline]
    pub unsafe fn from_slice_with_strides_generic_unchecked(
        data: &'a mut [T],
        start: usize,
        nrows: R,
        ncols: C,
        rstride: RStride,
        cstride: CStride,
    ) -> Self {
        let data = SliceStorageMut::from_raw_parts(
            data.as_mut_ptr().add(start),
            (nrows, ncols),
            (rstride, cstride),
        );
        Self::from_data(data)
    }

    /// Creates a mutable matrix slice from an array and with dimensions and strides specified by generic types instances.
    ///
    /// Panics if the input data array dose not contain enough elements.
    /// The generic types `R`, `C`, `RStride`, `CStride` can either be type-level integers or integers wrapped with `Dynamic::new()`.
    #[inline]
    pub fn from_slice_with_strides_generic(
        data: &'a mut [T],
        nrows: R,
        ncols: C,
        rstride: RStride,
        cstride: CStride,
    ) -> Self {
        // NOTE: The assertion implements the following formula, but without subtractions to avoid
        // underflow panics:
        //      len >= (ncols - 1) * cstride + (nrows - 1) * rstride + 1
        assert!(
            data.len() + cstride.value() + rstride.value()
                >= ncols.value() * cstride.value() + nrows.value() * rstride.value() + 1,
            "Matrix slice: input data buffer to small."
        );

        assert!(
            {
                let nrows = nrows.value();
                let ncols = ncols.value();
                let rstride = rstride.value();
                let cstride = cstride.value();

                nrows * ncols <= 1
                    || match (rstride, cstride) {
                        (0, 0) => false,      // otherwise: matrix[(0, 0)] == index[(nrows - 1, ncols - 1)],
                        (0, _) => nrows <= 1, // otherwise: matrix[(0, 0)] == index[(nrows - 1, 0)],
                        (_, 0) => ncols <= 1, // otherwise: matrix[(0, 0)] == index[(0, ncols - 1)],
                        (_, _) => {
                            // otherwise: matrix[(0, numer)] == index[(denom, 0)]
                            let ratio = Ratio::new(rstride, cstride);
                            nrows <= *ratio.denom() || ncols <= *ratio.numer()
                        }
                    }
            },
            "Matrix slice: dimensions and strides result in aliased indices."
        );

        unsafe {
            Self::from_slice_with_strides_generic_unchecked(data, 0, nrows, ncols, rstride, cstride)
        }
    }
}

impl<'a, T, R: Dim, C: Dim> MatrixSliceMutMN<'a, T, R, C> {
    /// Creates, without bound-checking, a mutable matrix slice from an array and with dimensions specified by generic types instances.
    ///
    /// # Safety
    /// This method is unsafe because the input data array is not checked to contain enough elements.
    /// The generic types `R` and `C` can either be type-level integers or integers wrapped with `Dynamic::new()`.
    #[inline]
    pub unsafe fn from_slice_generic_unchecked(
        data: &'a mut [T],
        start: usize,
        nrows: R,
        ncols: C,
    ) -> Self {
        Self::from_slice_with_strides_generic_unchecked(
            data, start, nrows, ncols, Const::<1>, nrows,
        )
    }

    /// Creates a mutable matrix slice from an array and with dimensions and strides specified by generic types instances.
    ///
    /// Panics if the input data array dose not contain enough elements.
    /// The generic types `R` and `C` can either be type-level integers or integers wrapped with `Dynamic::new()`.
    #[inline]
    pub fn from_slice_generic(data: &'a mut [T], nrows: R, ncols: C) -> Self {
        Self::from_slice_with_strides_generic(data, nrows, ncols, Const::<1>, nrows)
    }
}

macro_rules! impl_constructors_mut(
    ($($Dims: ty),*; $(=> $DimIdent: ident: $DimBound: ident),*; $($gargs: expr),*; $($args: ident),*) => {
        impl<'a, T, $($DimIdent: $DimBound),*> MatrixSliceMutMN<'a, T, $($Dims),*> {
            /// Creates a new mutable matrix slice from the given data array.
            ///
            /// Panics if `data` does not contain enough elements.
            #[inline]
            pub fn from_slice(data: &'a mut [T], $($args: usize),*) -> Self {
                Self::from_slice_generic(data, $($gargs),*)
            }

            /// Creates, without bound checking, a new mutable matrix slice from the given data array.
            #[inline]
            pub unsafe fn from_slice_unchecked(data: &'a mut [T], start: usize, $($args: usize),*) -> Self {
                Self::from_slice_generic_unchecked(data, start, $($gargs),*)
            }
        }

        impl<'a, T, $($DimIdent: $DimBound, )*> MatrixSliceMutMN<'a, T, $($Dims,)* Dynamic, Dynamic> {
            /// Creates a new mutable matrix slice with the specified strides from the given data array.
            ///
            /// Panics if `data` does not contain enough elements.
            #[inline]
            pub fn from_slice_with_strides_mut(data: &'a mut [T], $($args: usize,)* rstride: usize, cstride: usize) -> Self {
                Self::from_slice_with_strides_generic(
                    data, $($gargs,)* Dynamic::new(rstride), Dynamic::new(cstride))
            }

            /// Creates, without bound checking, a new mutable matrix slice with the specified strides from the given data array.
            #[inline]
            pub unsafe fn from_slice_with_strides_unchecked(data: &'a mut [T], start: usize, $($args: usize,)* rstride: usize, cstride: usize) -> Self {
                Self::from_slice_with_strides_generic_unchecked(
                    data, start, $($gargs,)* Dynamic::new(rstride), Dynamic::new(cstride))
            }
        }
    }
);

// TODO: this is not very pretty. We could find a better call syntax.
impl_constructors_mut!(R, C;                         // Arguments for Matrix<T, ..., S>
=> R: DimName, => C: DimName; // Type parameters for impl<T, ..., S>
R::name(), C::name();         // Arguments for `_generic` constructors.
); // Arguments for non-generic constructors.

impl_constructors_mut!(R, Dynamic;
                       => R: DimName;
                       R::name(), Dynamic::new(ncols);
                       ncols);

impl_constructors_mut!(Dynamic, C;
                       => C: DimName;
                       Dynamic::new(nrows), C::name();
                       nrows);

impl_constructors_mut!(Dynamic, Dynamic;
                       ;
                       Dynamic::new(nrows), Dynamic::new(ncols);
                       nrows, ncols);
