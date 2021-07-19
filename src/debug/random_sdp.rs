use std::fmt;

#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use crate::base::allocator::Allocator;
use crate::base::dimension::{Dim, Dynamic};
use crate::base::{DefaultAllocator, OMatrix, Owned};
use simba::scalar::ComplexField;

use crate::debug::RandomOrthogonal;

/// A random, well-conditioned, symmetric definite-positive matrix.
pub struct RandomSDP<T, D: Dim = Dynamic>
where
    DefaultAllocator: Allocator<T, D, D>,
{
    m: OMatrix<T, D, D>,
}

impl<T: Copy, D: Dim> Copy for RandomSDP<T, D>
where
    DefaultAllocator: Allocator<T, D, D>,
    Owned<T, D, D>: Copy,
{
}

impl<T: Clone, D: Dim> Clone for RandomSDP<T, D>
where
    DefaultAllocator: Allocator<T, D, D>,
    Owned<T, D, D>: Clone,
{
    fn clone(&self) -> Self {
        Self { m: self.m.clone() }
    }
}

impl<T: fmt::Debug, D: Dim> fmt::Debug for RandomSDP<T, D>
where
    DefaultAllocator: Allocator<T, D, D>,
    Owned<T, D, D>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("RandomSDP").field("m", &self.m).finish()
    }
}

impl<T: ComplexField, D: Dim> RandomSDP<T, D>
where
    DefaultAllocator: Allocator<T, D, D>,
{
    /// Retrieve the generated matrix.
    pub fn unwrap(self) -> OMatrix<T, D, D> {
        self.m
    }

    /// Creates a new well conditioned symmetric definite-positive matrix from its dimension and a
    /// random reals generators.
    pub fn new<Rand: FnMut() -> T>(dim: D, mut rand: Rand) -> Self {
        let mut m = RandomOrthogonal::new(dim, || rand()).unwrap();
        let mt = m.adjoint();

        for i in 0..dim.value() {
            let mut col = m.column_mut(i);
            let eigenval = T::one() + T::from_real(rand().modulus());
            col *= eigenval;
        }

        RandomSDP { m: m * mt }
    }
}

#[cfg(feature = "arbitrary")]
impl<T: ComplexField + Arbitrary + Send, D: Dim> Arbitrary for RandomSDP<T, D>
where
    DefaultAllocator: Allocator<T, D, D>,
    Owned<T, D, D>: Clone + Send,
{
    fn arbitrary(g: &mut Gen) -> Self {
        let dim = D::try_to_usize().unwrap_or(1 + usize::arbitrary(g) % 50);
        Self::new(D::from_usize(dim), || T::arbitrary(g))
    }
}
