use crate::{Quaternion, Scalar, SimdValue, UnitQuaternion};

impl<T> From<mint::Quaternion<T>> for Quaternion<T> {
    fn from(q: mint::Quaternion<T>) -> Self {
        Self::new(q.s, q.v.x, q.v.y, q.v.z)
    }
}

impl<T: Scalar> Into<mint::Quaternion<T>> for Quaternion<T> {
    fn into(self) -> mint::Quaternion<T> {
        mint::Quaternion {
            v: mint::Vector3 {
                x: self[0].inlined_clone(),
                y: self[1].inlined_clone(),
                z: self[2].inlined_clone(),
            },
            s: self[3].inlined_clone(),
        }
    }
}

impl<T: Scalar + SimdValue> Into<mint::Quaternion<T>> for UnitQuaternion<T> {
    fn into(self) -> mint::Quaternion<T> {
        mint::Quaternion {
            v: mint::Vector3 {
                x: self[0].inlined_clone(),
                y: self[1].inlined_clone(),
                z: self[2].inlined_clone(),
            },
            s: self[3].inlined_clone(),
        }
    }
}
