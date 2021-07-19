use std::fmt;

#[cfg(feature = "arbitrary")]
use quickcheck::{Arbitrary, Gen};

use crate::base::allocator::Allocator;
use crate::base::dimension::{Dim, DimName, Dynamic};
use crate::base::{DefaultAllocator, OMatrix};
use crate::linalg::givens::GivensRotation;
use crate::storage::Owned;
use simba::scalar::ComplexField;

/// A random orthogonal matrix.
pub struct RandomOrthogonal<T, D: Dim = Dynamic>
where
    DefaultAllocator: Allocator<T, D, D>,
{
    m: OMatrix<T, D, D>,
}

impl<T: Copy, D: DimName> Copy for RandomOrthogonal<T, D>
where
    DefaultAllocator: Allocator<T, D, D>,
    Owned<T, D, D>: Copy,
{
}

impl<T: Clone, D: Dim> Clone for RandomOrthogonal<T, D>
where
    DefaultAllocator: Allocator<T, D, D>,
    Owned<T, D, D>: Clone,
{
    fn clone(&self) -> Self {
        Self { m: self.m.clone() }
    }
}

impl<T: fmt::Debug, D: Dim> fmt::Debug for RandomOrthogonal<T, D>
where
    DefaultAllocator: Allocator<T, D, D>,
    Owned<T, D, D>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("RandomOrthogonal")
            .field("m", &self.m)
            .finish()
    }
}

impl<T: ComplexField, D: Dim> RandomOrthogonal<T, D>
where
    DefaultAllocator: Allocator<T, D, D>,
{
    /// Retrieve the generated matrix.
    pub fn unwrap(self) -> OMatrix<T, D, D> {
        self.m
    }

    /// Creates a new random orthogonal matrix from its dimension and a random reals generators.
    pub fn new<Rand: FnMut() -> T>(dim: D, mut rand: Rand) -> Self {
        let mut res = OMatrix::identity_generic(dim, dim);

        // Create an orthogonal matrix by composing random Givens rotations rotations.
        for i in 0..dim.value() - 1 {
            let rot = GivensRotation::new(rand(), rand()).0;
            rot.rotate(&mut res.fixed_rows_mut::<2>(i));
        }

        RandomOrthogonal { m: res }
    }
}

#[cfg(feature = "arbitrary")]
impl<T: ComplexField + Arbitrary + Send, D: Dim> Arbitrary for RandomOrthogonal<T, D>
where
    DefaultAllocator: Allocator<T, D, D>,
    Owned<T, D, D>: Clone + Send,
{
    fn arbitrary(g: &mut Gen) -> Self {
        let dim = D::try_to_usize().unwrap_or(1 + usize::arbitrary(g) % 50);
        Self::new(D::from_usize(dim), || T::arbitrary(g))
    }
}
