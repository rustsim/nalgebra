use std::ops::{Deref, DerefMut};

use simba::simd::SimdValue;

use crate::base::coordinates::IJKW;
use crate::Scalar;

use crate::geometry::Quaternion;

impl<T: Scalar + SimdValue> Deref for Quaternion<T> {
    type Target = IJKW<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        // Safety: Self and IJKW are both stored as contiguous coordinates.
        unsafe { &*(self as *const _ as *const _) }
    }
}

impl<T: Scalar + SimdValue> DerefMut for Quaternion<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self as *mut _ as *mut _) }
    }
}
