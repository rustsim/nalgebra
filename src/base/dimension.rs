#![allow(missing_docs)]

//! Traits and tags for identifying the dimension of all algebraic entities.

use std::any::TypeId;
use std::cmp;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};
use typenum::{self, Diff, Max, Maximum, Min, Minimum, Prod, Quot, Sum, Unsigned};

#[cfg(feature = "serde-serialize-no-std")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Stores the dimension of dynamically-sized algebraic entities.
#[derive(Clone, Copy, Default, Eq, PartialEq, Debug)]
pub struct Dynamic {
    value: usize,
}

impl Dynamic {
    /// A dynamic size equal to `value`.
    #[inline]
    pub const fn new(value: usize) -> Self {
        Self { value }
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl Serialize for Dynamic {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.value.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<'de> Deserialize<'de> for Dynamic {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        usize::deserialize(deserializer).map(|x| Dynamic { value: x })
    }
}

/// Trait implemented by `Dynamic`.
pub trait IsDynamic {}
/// Trait implemented by `Dynamic` and type-level integers different from `U1`.
pub trait IsNotStaticOne {}

impl IsDynamic for Dynamic {}
impl IsNotStaticOne for Dynamic {}

/// Trait implemented by any type that can be used as a dimension. This includes type-level
/// integers and `Dynamic` (for dimensions not known at compile-time).
pub trait Dim: 'static + Debug + Copy + Default + PartialEq + Send + Sync {
    #[inline(always)]
    fn is<D: Dim>() -> bool {
        TypeId::of::<Self>() == TypeId::of::<D>()
    }

    /// Gets the compile-time value of `Self`. Returns `None` if it is not known, i.e., if `Self =
    /// Dynamic`.
    fn try_to_usize() -> Option<usize>;

    /// Returns whether `Self` has a known compile-time value.
    fn is_static() -> bool {
        Self::try_to_usize().is_some()
    }

    /// Returns whether `Self` does not have a known compile-time value.
    fn is_dynamic() -> bool {
        Self::try_to_usize().is_none()
    }

    /// Gets the run-time value of `self`. For type-level integers, this is the same as
    /// `Self::try_to_usize().unwrap()`.
    fn value(&self) -> usize;

    /// Builds an instance of `Self` from a run-time value. Panics if `Self` is a type-level
    /// integer and `dim != Self::try_to_usize().unwrap()`.
    fn from_usize(dim: usize) -> Self;
}

impl Dim for Dynamic {
    #[inline]
    fn try_to_usize() -> Option<usize> {
        None
    }

    #[inline]
    fn from_usize(dim: usize) -> Self {
        Self::new(dim)
    }

    #[inline]
    fn value(&self) -> usize {
        self.value
    }
}

impl Add<usize> for Dynamic {
    type Output = Dynamic;

    #[inline]
    fn add(self, rhs: usize) -> Self {
        Self::new(self.value + rhs)
    }
}

impl Sub<usize> for Dynamic {
    type Output = Dynamic;

    #[inline]
    fn sub(self, rhs: usize) -> Self {
        Self::new(self.value - rhs)
    }
}

/*
 *
 * Operations.
 *
 */

macro_rules! dim_ops(
    ($($DimOp:    ident, $DimNameOp: ident,
       $Op:       ident, $op: ident, $op_path: path,
       $DimResOp: ident, $DimNameResOp: ident,
       $ResOp: ident);* $(;)*) => {$(
        pub type $DimResOp<D1, D2> = <D1 as $DimOp<D2>>::Output;

        pub trait $DimOp<D: Dim>: Dim {
            type Output: Dim;

            fn $op(self, other: D) -> Self::Output;
        }

        impl<const A: usize, const B: usize> $DimOp<Const<B>> for Const<A>
        where
            Const<A>: ToTypenum,
            Const<B>: ToTypenum,
            <Const<A> as ToTypenum>::Typenum: $Op<<Const<B> as ToTypenum>::Typenum>,
            $ResOp<<Const<A> as ToTypenum>::Typenum, <Const<B> as ToTypenum>::Typenum>: ToConst,
        {
            type Output =
                <$ResOp<<Const<A> as ToTypenum>::Typenum, <Const<B> as ToTypenum>::Typenum> as ToConst>::Const;

            fn $op(self, _: Const<B>) -> Self::Output {
                Self::Output::name()
            }
        }

        impl<D: Dim> $DimOp<D> for Dynamic {
            type Output = Dynamic;

            #[inline]
            fn $op(self, other: D) -> Dynamic {
                Dynamic::new($op_path(self.value, other.value()))
            }
        }

        // TODO: use Const<T> instead of D: DimName?
        impl<D: DimName> $DimOp<Dynamic> for D {
            type Output = Dynamic;

            #[inline]
            fn $op(self, other: Dynamic) -> Dynamic {
                Dynamic::new($op_path(self.value(), other.value))
            }
        }

        pub type $DimNameResOp<D1, D2> = <D1 as $DimNameOp<D2>>::Output;

        pub trait $DimNameOp<D: DimName>: DimName {
            type Output: DimName;

            fn $op(self, other: D) -> Self::Output;
        }

        impl<const A: usize, const B: usize> $DimNameOp<Const<B>> for Const<A>
        where
            Const<A>: ToTypenum,
            Const<B>: ToTypenum,
            <Const<A> as ToTypenum>::Typenum: $Op<<Const<B> as ToTypenum>::Typenum>,
            $ResOp<<Const<A> as ToTypenum>::Typenum, <Const<B> as ToTypenum>::Typenum>: ToConst,
        {
            type Output =
                <$ResOp<<Const<A> as ToTypenum>::Typenum, <Const<B> as ToTypenum>::Typenum> as ToConst>::Const;

            fn $op(self, _: Const<B>) -> Self::Output {
                Self::Output::name()
            }
        }
   )*}
);

dim_ops!(
    DimAdd, DimNameAdd, Add, add, Add::add, DimSum,     DimNameSum,     Sum;
    DimMul, DimNameMul, Mul, mul, Mul::mul, DimProd,    DimNameProd,    Prod;
    DimSub, DimNameSub, Sub, sub, Sub::sub, DimDiff,    DimNameDiff,    Diff;
    DimDiv, DimNameDiv, Div, div, Div::div, DimQuot,    DimNameQuot,    Quot;
    DimMin, DimNameMin, Min, min, cmp::min, DimMinimum, DimNameMinimum, Minimum;
    DimMax, DimNameMax, Max, max, cmp::max, DimMaximum, DimNameMaximum, Maximum;
);

/// A wrapper around const types, which provides the capability of performing
/// type-level arithmetic. This might get removed if const-generics become
/// more powerful in the future.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, Hash)]
pub struct Const<const R: usize>;

/// Trait implemented exclusively by type-level integers.
pub trait DimName: Dim {
    const USIZE: usize;

    /// The name of this dimension, i.e., the singleton `Self`.
    fn name() -> Self;

    // TODO: this is not a very idiomatic name.
    /// The value of this dimension.
    fn dim() -> usize;
}

#[cfg(feature = "serde-serialize-no-std")]
impl<const D: usize> Serialize for Const<D> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ().serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize-no-std")]
impl<'de, const D: usize> Deserialize<'de> for Const<D> {
    fn deserialize<Des>(deserializer: Des) -> Result<Self, Des::Error>
    where
        Des: Deserializer<'de>,
    {
        <()>::deserialize(deserializer).map(|_| Const::<D>)
    }
}

#[cfg(feature = "rkyv-serialize-no-std")]
mod rkyv_impl {
    use super::Const;
    use rkyv::{Archive, Deserialize, Fallible, Serialize};

    impl<const R: usize> Archive for Const<R> {
        type Archived = Self;
        type Resolver = ();

        fn resolve(
            &self,
            _: usize,
            _: Self::Resolver,
            _: &mut core::mem::MaybeUninit<Self::Archived>,
        ) {
        }
    }

    impl<S: Fallible + ?Sized, const R: usize> Serialize<S> for Const<R> {
        fn serialize(&self, _: &mut S) -> Result<Self::Resolver, S::Error> {
            Ok(())
        }
    }

    impl<D: Fallible + ?Sized, const R: usize> Deserialize<Self, D> for Const<R> {
        fn deserialize(&self, _: &mut D) -> Result<Self, D::Error> {
            Ok(Const)
        }
    }
}

pub trait ToConst {
    type Const: DimName;
}

pub trait ToTypenum {
    type Typenum: Unsigned;
}

impl<const T: usize> Dim for Const<T> {
    fn try_to_usize() -> Option<usize> {
        Some(T)
    }

    fn value(&self) -> usize {
        T
    }

    fn from_usize(dim: usize) -> Self {
        assert_eq!(dim, T);
        Self
    }
}

impl<const T: usize> DimName for Const<T> {
    const USIZE: usize = T;

    #[inline]
    fn name() -> Self {
        Self
    }

    #[inline]
    fn dim() -> usize {
        T
    }
}

pub type U1 = Const<1>;

impl ToTypenum for Const<{ typenum::U1::USIZE }> {
    type Typenum = typenum::U1;
}

impl ToConst for typenum::U1 {
    type Const = Const<{ typenum::U1::USIZE }>;
}

macro_rules! from_to_typenum (
    ($($D: ident),* $(,)*) => {$(
        pub type $D = Const<{ typenum::$D::USIZE }>;

        impl ToTypenum for Const<{ typenum::$D::USIZE }> {
            type Typenum = typenum::$D;
        }

        impl ToConst for typenum::$D {
            type Const = Const<{ typenum::$D::USIZE }>;
        }

        impl IsNotStaticOne for $D { }
    )*}
);

from_to_typenum!(
    U0, /*U1,*/ U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18,
    U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37,
    U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56,
    U57, U58, U59, U60, U61, U62, U63, U64, U65, U66, U67, U68, U69, U70, U71, U72, U73, U74, U75,
    U76, U77, U78, U79, U80, U81, U82, U83, U84, U85, U86, U87, U88, U89, U90, U91, U92, U93, U94,
    U95, U96, U97, U98, U99, U100, U101, U102, U103, U104, U105, U106, U107, U108, U109, U110,
    U111, U112, U113, U114, U115, U116, U117, U118, U119, U120, U121, U122, U123, U124, U125, U126,
    U127
);
