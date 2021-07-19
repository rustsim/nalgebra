use std::any::TypeId;
use std::fmt::Debug;

/// The basic scalar trait for all structures of `nalgebra`.
///
/// This is by design a very loose trait, and does not make any assumption on
/// the algebraic properties of `Self`. It has various purposes and objectives:
/// - Enforces simple and future-proof trait bounds.
/// - Enables important optimizations for floating point types via specialization.
/// - Makes debugging generic code possible in most circumstances.
pub trait Scalar: 'static + Clone + Debug {
    #[inline]
    /// Tests whether `Self` is the same as the type `T`.
    ///
    /// Typically used to test of `Self` is an `f32` or an `f64`, which is
    /// important as it allows for specialization and certain optimizations to
    /// be made.
    ///
    // If the need ever arose to get rid of the `'static` requirement, we could
    // merely replace this method by two unsafe associated methods `is_f32` and
    // `is_f64`.
    fn is<T: Scalar>() -> bool {
        TypeId::of::<Self>() == TypeId::of::<T>()
    }

    /// Performance hack: Clone doesn't get inlined for Copy types in debug
    /// mode, so make it inline anyway.
    fn inlined_clone(&self) -> Self {
        self.clone()
    }
}

// Unfortunately, this blanket impl leads to many misleading compiler messages
// telling you to implement Copy, even though Scalar is what's really needed.
impl<T: 'static + Copy + Debug> Scalar for T {
    #[inline(always)]
    fn inlined_clone(&self) -> T {
        *self
    }
}
