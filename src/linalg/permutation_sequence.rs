use std::fmt;
use std::mem::MaybeUninit;

#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

use num::One;
use simba::scalar::ClosedNeg;

use crate::allocator::Allocator;
use crate::base::{DefaultAllocator, Matrix, OVector, Scalar};
#[cfg(any(feature = "std", feature = "alloc"))]
use crate::dimension::Dynamic;
use crate::dimension::{Dim, DimName};
use crate::iter::MatrixIter;
use crate::storage::{InnerOwned, StorageMut};
use crate::{Const, U1};

/// A sequence of row or column permutations.
#[cfg_attr(feature = "serde-serialize-no-std", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(serialize = "DefaultAllocator: Allocator<(usize, usize), D>,
         OVector<(usize, usize), D>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(deserialize = "DefaultAllocator: Allocator<(usize, usize), D>,
         OVector<(usize, usize), D>: Deserialize<'de>"))
)]
pub struct PermutationSequence<D: Dim>
where
    DefaultAllocator: Allocator<(usize, usize), D>,
{
    len: usize,
    ipiv: OVector<MaybeUninit<(usize, usize)>, D>,
}

impl<D: Dim> Copy for PermutationSequence<D>
where
    DefaultAllocator: Allocator<(usize, usize), D>,
    OVector<MaybeUninit<(usize, usize)>, D>: Copy,
{
}

impl<D: Dim> Clone for PermutationSequence<D>
where
    DefaultAllocator: Allocator<(usize, usize), D>,
    OVector<MaybeUninit<(usize, usize)>, D>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            len: self.len,
            ipiv: self.ipiv.clone(),
        }
    }
}

impl<D: Dim> fmt::Debug for PermutationSequence<D>
where
    DefaultAllocator: Allocator<(usize, usize), D>,
    OVector<MaybeUninit<(usize, usize)>, D>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("PermutationSequence")
            .field("len", &self.len)
            .field("ipiv", &self.ipiv)
            .finish()
    }
}

impl<D: DimName> PermutationSequence<D>
where
    DefaultAllocator: Allocator<(usize, usize), D>,
{
    /// Creates a new statically-allocated sequence of `D` identity permutations.
    #[inline]
    pub fn identity() -> Self {
        Self::identity_generic(D::name())
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl PermutationSequence<Dynamic>
where
    DefaultAllocator: Allocator<(usize, usize), Dynamic>,
{
    /// Creates a new dynamically-allocated sequence of `n` identity permutations.
    #[inline]
    pub fn identity(n: usize) -> Self {
        Self::identity_generic(Dynamic::new(n))
    }
}

impl<D: Dim> PermutationSequence<D>
where
    DefaultAllocator: Allocator<(usize, usize), D>,
{
    /// Creates a new sequence of D identity permutations.
    #[inline]
    pub fn identity_generic(dim: D) -> Self {
        Self {
            len: 0,
            ipiv: OVector::new_uninitialized_generic(dim, Const::<1>),
        }
    }

    /// Adds the interchange of the row (or column) `i` with the row (or column) `i2` to this
    /// sequence of permutations.
    #[inline]
    pub fn append_permutation(&mut self, i: usize, i2: usize) {
        if i != i2 {
            assert!(
                self.len < self.ipiv.len(),
                "Maximum number of permutations exceeded."
            );
            self.ipiv[self.len] = MaybeUninit::new((i, i2));
            self.len += 1;
        }
    }

    /// Applies this sequence of permutations to the rows of `rhs`.
    #[inline]
    pub fn permute_rows<T: Scalar, R2: Dim, C2: Dim, S2>(&self, rhs: &mut Matrix<T, R2, C2, S2>)
    where
        S2: StorageMut<T, R2, C2>,
    {
        for perm in self.iter() {
            rhs.swap_rows(perm.0, perm.1)
        }
    }

    /// Applies this sequence of permutations in reverse to the rows of `rhs`.
    #[inline]
    pub fn inv_permute_rows<T: Scalar, R2: Dim, C2: Dim, S2>(&self, rhs: &mut Matrix<T, R2, C2, S2>)
    where
        S2: StorageMut<T, R2, C2>,
    {
        for perm in self.iter().rev() {
            let (i1, i2) = perm;
            rhs.swap_rows(i1, i2)
        }
    }

    /// Applies this sequence of permutations to the columns of `rhs`.
    #[inline]
    pub fn permute_columns<T: Scalar, R2: Dim, C2: Dim, S2>(&self, rhs: &mut Matrix<T, R2, C2, S2>)
    where
        S2: StorageMut<T, R2, C2>,
    {
        for perm in self.iter() {
            rhs.swap_columns(perm.0, perm.1)
        }
    }

    /// Applies this sequence of permutations in reverse to the columns of `rhs`.
    #[inline]
    pub fn inv_permute_columns<T: Scalar, R2: Dim, C2: Dim, S2>(
        &self,
        rhs: &mut Matrix<T, R2, C2, S2>,
    ) where
        S2: StorageMut<T, R2, C2>,
    {
        for perm in self.iter().rev() {
            let (i1, i2) = perm;
            rhs.swap_columns(i1, i2)
        }
    }

    /// The number of non-identity permutations applied by this sequence.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the permutation sequence contains no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The determinant of the matrix corresponding to this permutation.
    #[inline]
    #[must_use]
    pub fn determinant<T: One + ClosedNeg>(&self) -> T {
        if self.len % 2 == 0 {
            T::one()
        } else {
            -T::one()
        }
    }

    /// Iterates over the permutations that have been initialized.
    pub fn iter(
        &self,
    ) -> std::iter::Map<
        std::iter::Copied<
            std::iter::Take<
                MatrixIter<
                    MaybeUninit<(usize, usize)>,
                    D,
                    U1,
                    InnerOwned<MaybeUninit<(usize, usize)>, D, U1>,
                >,
            >,
        >,
        impl FnMut(MaybeUninit<(usize, usize)>) -> (usize, usize),
    > {
        self.ipiv
            .iter()
            .take(self.len)
            .copied()
            .map(|e| unsafe { e.assume_init() })
    }
}
