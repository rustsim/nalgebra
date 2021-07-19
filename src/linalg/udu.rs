use std::fmt;

#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Serialize};

use crate::allocator::Allocator;
use crate::base::{Const, DefaultAllocator, OMatrix, OVector};
use crate::dimension::{Dim, DimName};
use crate::storage::{InnerOwned, Storage};
use simba::scalar::RealField;

/// UDU factorization.
#[cfg_attr(feature = "serde-serialize-no-std", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(serialize = "OVector<T, D>: Serialize, OMatrix<T, D, D>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize-no-std",
    serde(bound(
        deserialize = "OVector<T, D>: Deserialize<'de>, OMatrix<T, D, D>: Deserialize<'de>"
    ))
)]
pub struct UDU<T, D: Dim>
where
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    /// The upper triangular matrix resulting from the factorization
    pub u: OMatrix<T, D, D>,
    /// The diagonal matrix resulting from the factorization
    pub d: OVector<T, D>,
}

impl<T: Copy, D: DimName> Copy for UDU<T, D>
where
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
    InnerOwned<T, D>: Copy,
    InnerOwned<T, D, D>: Copy,
{
}

impl<T: Clone, D: Dim> Clone for UDU<T, D>
where
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
    InnerOwned<T, D>: Clone,
    InnerOwned<T, D, D>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            u: self.u.clone(),
            d: self.d.clone(),
        }
    }
}

impl<T: fmt::Debug, D: Dim> fmt::Debug for UDU<T, D>
where
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
    InnerOwned<T, D>: fmt::Debug,
    InnerOwned<T, D, D>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("UDU")
            .field("u", &self.u)
            .field("d", &self.d)
            .finish()
    }
}

impl<T: RealField, D: Dim> UDU<T, D>
where
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    /// Computes the UDU^T factorization.
    ///
    /// The input matrix `p` is assumed to be symmetric and this decomposition will only read
    /// the upper-triangular part of `p`.
    ///
    /// Ref.: "Optimal control and estimation-Dover Publications", Robert F. Stengel, (1994) page 360
    pub fn new(p: OMatrix<T, D, D>) -> Option<Self> {
        let n = p.ncols();
        let n_dim = p.data.shape().1;

        let mut d = OVector::zeros_generic(n_dim, Const::<1>);
        let mut u = OMatrix::zeros_generic(n_dim, n_dim);

        d[n - 1] = p[(n - 1, n - 1)];

        if d[n - 1].is_zero() {
            return None;
        }

        u.column_mut(n - 1)
            .axpy(T::one() / d[n - 1], &p.column(n - 1), T::zero());

        for j in (0..n - 1).rev() {
            let mut d_j = d[j];
            for k in j + 1..n {
                d_j += d[k] * u[(j, k)].powi(2);
            }

            d[j] = p[(j, j)] - d_j;

            if d[j].is_zero() {
                return None;
            }

            for i in (0..=j).rev() {
                let mut u_ij = u[(i, j)];
                for k in j + 1..n {
                    u_ij += d[k] * u[(j, k)] * u[(i, k)];
                }

                u[(i, j)] = (p[(i, j)] - u_ij) / d[j];
            }

            u[(j, j)] = T::one();
        }

        Some(Self { u, d })
    }

    /// Returns the diagonal elements as a matrix
    #[must_use]
    pub fn d_matrix(&self) -> OMatrix<T, D, D> {
        OMatrix::from_diagonal(&self.d)
    }
}
