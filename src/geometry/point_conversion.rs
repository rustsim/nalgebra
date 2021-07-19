use num::{One, Zero};
use simba::scalar::{ClosedDiv, SubsetOf, SupersetOf};
use simba::simd::PrimitiveSimdValue;

use crate::base::allocator::{Allocator, InnerAllocator};
use crate::base::dimension::{DimNameAdd, DimNameSum, U1};
use crate::base::{Const, DefaultAllocator, Matrix, OVector, Scalar};

use crate::geometry::Point;
use crate::{DimName, OPoint};

/*
 * This file provides the following conversions:
 * =============================================
 *
 * Point -> Point
 * Point -> Vector (homogeneous)
 */

impl<T1, T2, D: DimName> SubsetOf<OPoint<T2, D>> for OPoint<T1, D>
where
    T2: SupersetOf<T1>,
    DefaultAllocator: Allocator<T1, D> + Allocator<T2, D>,
{
    #[inline]
    fn to_superset(&self) -> OPoint<T2, D> {
        OPoint::from(self.coords.to_superset())
    }

    #[inline]
    fn is_in_subset(m: &OPoint<T2, D>) -> bool {
        // TODO: is there a way to reuse the `.is_in_subset` from the matrix implementation of
        // SubsetOf?
        m.iter().all(|e| e.is_in_subset())
    }

    #[inline]
    fn from_superset_unchecked(m: &OPoint<T2, D>) -> Self {
        Self::from(Matrix::from_superset_unchecked(&m.coords))
    }
}

impl<T1, T2, D> SubsetOf<OVector<T2, DimNameSum<D, U1>>> for OPoint<T1, D>
where
    D: DimNameAdd<U1>,
    T2: Scalar + Zero + One + ClosedDiv + SupersetOf<T1>,
    DefaultAllocator: Allocator<T1, D>
        + Allocator<T2, D>
        + Allocator<T1, DimNameSum<D, U1>>
        + Allocator<T2, DimNameSum<D, U1>>,
    // + Allocator<T1, D>
    // + Allocator<T2, D>,
{
    #[inline]
    fn to_superset(&self) -> OVector<T2, DimNameSum<D, U1>> {
        let p: OPoint<T2, D> = self.to_superset();
        p.into_homogeneous()
    }

    #[inline]
    fn is_in_subset(v: &OVector<T2, DimNameSum<D, U1>>) -> bool {
        crate::is_convertible::<_, OVector<T1, DimNameSum<D, U1>>>(v) && !v[D::dim()].is_zero()
    }

    #[inline]
    fn from_superset_unchecked(v: &OVector<T2, DimNameSum<D, U1>>) -> Self {
        let coords = v.generic_slice((0, 0), (D::name(), Const::<1>)) / v[D::dim()].clone();
        Self {
            coords: crate::convert_unchecked(coords),
        }
    }
}

impl<T: Zero + One, D: DimName> From<OPoint<T, D>> for OVector<T, DimNameSum<D, U1>>
where
    D: DimNameAdd<U1>,
    DefaultAllocator: Allocator<T, DimNameSum<D, U1>> + Allocator<T, D>,
{
    #[inline]
    fn from(t: OPoint<T, D>) -> Self {
        t.into_homogeneous()
    }
}

impl<T, const D: usize> From<[T; D]> for Point<T, D> {
    #[inline]
    fn from(coords: [T; D]) -> Self {
        Point {
            coords: coords.into(),
        }
    }
}

impl<T, const D: usize> From<Point<T, D>> for [T; D]
where
    T: Clone,
{
    #[inline]
    fn from(p: Point<T, D>) -> Self {
        p.coords.into()
    }
}

impl<T, D: DimName> From<OVector<T, D>> for OPoint<T, D>
where
    DefaultAllocator: InnerAllocator<T, D>,
{
    #[inline]
    fn from(coords: OVector<T, D>) -> Self {
        OPoint { coords }
    }
}

impl<T: Scalar + PrimitiveSimdValue, const D: usize> From<[Point<T::Element, D>; 2]> for Point<T, D>
where
    T: From<[<T as simba::simd::SimdValue>::Element; 2]>,
    T::Element: Scalar,
{
    #[inline]
    fn from(arr: [Point<T::Element, D>; 2]) -> Self {
        Self::from(OVector::from([
            arr[0].coords.clone(),
            arr[1].coords.clone(),
        ]))
    }
}

impl<T: Scalar + PrimitiveSimdValue, const D: usize> From<[Point<T::Element, D>; 4]> for Point<T, D>
where
    T: From<[<T as simba::simd::SimdValue>::Element; 4]>,
    T::Element: Scalar,
{
    #[inline]
    fn from(arr: [Point<T::Element, D>; 4]) -> Self {
        Self::from(OVector::from([
            arr[0].coords.clone(),
            arr[1].coords.clone(),
            arr[2].coords.clone(),
            arr[3].coords.clone(),
        ]))
    }
}

impl<T: Scalar + PrimitiveSimdValue, const D: usize> From<[Point<T::Element, D>; 8]> for Point<T, D>
where
    T: From<[<T as simba::simd::SimdValue>::Element; 8]>,
    T::Element: Scalar,
{
    #[inline]
    fn from(arr: [Point<T::Element, D>; 8]) -> Self {
        Self::from(OVector::from([
            arr[0].coords.clone(),
            arr[1].coords.clone(),
            arr[2].coords.clone(),
            arr[3].coords.clone(),
            arr[4].coords.clone(),
            arr[5].coords.clone(),
            arr[6].coords.clone(),
            arr[7].coords.clone(),
        ]))
    }
}

impl<T: Scalar + PrimitiveSimdValue, const D: usize> From<[Point<T::Element, D>; 16]>
    for Point<T, D>
where
    T: From<[<T as simba::simd::SimdValue>::Element; 16]>,
    T::Element: Scalar,
{
    #[inline]
    fn from(arr: [Point<T::Element, D>; 16]) -> Self {
        Self::from(OVector::from([
            arr[0].coords.clone(),
            arr[1].coords.clone(),
            arr[2].coords.clone(),
            arr[3].coords.clone(),
            arr[4].coords.clone(),
            arr[5].coords.clone(),
            arr[6].coords.clone(),
            arr[7].coords.clone(),
            arr[8].coords.clone(),
            arr[9].coords.clone(),
            arr[10].coords.clone(),
            arr[11].coords.clone(),
            arr[12].coords.clone(),
            arr[13].coords.clone(),
            arr[14].coords.clone(),
            arr[15].coords.clone(),
        ]))
    }
}
