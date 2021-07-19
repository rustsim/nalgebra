use crate::cs::CsMatrix;
use crate::ops::serial::{OperationError, OperationErrorKind};
use crate::ops::Op;
use crate::SparseEntryMut;
use nalgebra::{ClosedAdd, ClosedMul, DMatrixSlice, DMatrixSliceMut, Scalar};
use num_traits::{One, Zero};

fn spmm_cs_unexpected_entry() -> OperationError {
    OperationError::from_kind_and_message(
        OperationErrorKind::InvalidPattern,
        String::from("Found unexpected entry that is not present in `c`."),
    )
}

/// Helper functionality for implementing CSR/CSC SPMM.
///
/// Since CSR/CSC matrices are basically transpositions of each other, which lets us use the same
/// algorithm for the SPMM implementation. The implementation here is written in a CSR-centric
/// manner. This means that when using it for CSC, the order of the matrices needs to be
/// reversed (since transpose(AB) = transpose(B) * transpose(A) and CSC(A) = transpose(CSR(A)).
///
/// We assume here that the matrices have already been verified to be dimensionally compatible.
pub fn spmm_cs_prealloc<T>(
    beta: T,
    c: &mut CsMatrix<T>,
    alpha: T,
    a: &CsMatrix<T>,
    b: &CsMatrix<T>,
) -> Result<(), OperationError>
where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One,
{
    for i in 0..c.pattern().major_dim() {
        let a_lane_i = a.get_lane(i).unwrap();
        let mut c_lane_i = c.get_lane_mut(i).unwrap();
        for c_ij in c_lane_i.values_mut() {
            *c_ij = beta.inlined_clone() * c_ij.inlined_clone();
        }

        for (&k, a_ik) in a_lane_i.minor_indices().iter().zip(a_lane_i.values()) {
            let b_lane_k = b.get_lane(k).unwrap();
            let (mut c_lane_i_cols, mut c_lane_i_values) = c_lane_i.indices_and_values_mut();
            let alpha_aik = alpha.inlined_clone() * a_ik.inlined_clone();
            for (j, b_kj) in b_lane_k.minor_indices().iter().zip(b_lane_k.values()) {
                // Determine the location in C to append the value
                let (c_local_idx, _) = c_lane_i_cols
                    .iter()
                    .enumerate()
                    .find(|(_, c_col)| *c_col == j)
                    .ok_or_else(spmm_cs_unexpected_entry)?;

                c_lane_i_values[c_local_idx] += alpha_aik.inlined_clone() * b_kj.inlined_clone();
                c_lane_i_cols = &c_lane_i_cols[c_local_idx..];
                c_lane_i_values = &mut c_lane_i_values[c_local_idx..];
            }
        }
    }

    Ok(())
}

fn spadd_cs_unexpected_entry() -> OperationError {
    OperationError::from_kind_and_message(
        OperationErrorKind::InvalidPattern,
        String::from("Found entry in `op(a)` that is not present in `c`."),
    )
}

/// Helper functionality for implementing CSR/CSC SPADD.
pub fn spadd_cs_prealloc<T>(
    beta: T,
    c: &mut CsMatrix<T>,
    alpha: T,
    a: Op<&CsMatrix<T>>,
) -> Result<(), OperationError>
where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One + PartialEq,
{
    match a {
        Op::NoOp(a) => {
            for (mut c_lane_i, a_lane_i) in c.lane_iter_mut().zip(a.lane_iter()) {
                if beta != T::one() {
                    for c_ij in c_lane_i.values_mut() {
                        *c_ij *= beta.inlined_clone();
                    }
                }

                let (mut c_minors, mut c_vals) = c_lane_i.indices_and_values_mut();
                let (a_minors, a_vals) = (a_lane_i.minor_indices(), a_lane_i.values());

                for (a_col, a_val) in a_minors.iter().zip(a_vals) {
                    // TODO: Use exponential search instead of linear search.
                    // If C has substantially more entries in the row than A, then a line search
                    // will needlessly visit many entries in C.
                    let (c_idx, _) = c_minors
                        .iter()
                        .enumerate()
                        .find(|(_, c_col)| *c_col == a_col)
                        .ok_or_else(spadd_cs_unexpected_entry)?;
                    c_vals[c_idx] += alpha.inlined_clone() * a_val.inlined_clone();
                    c_minors = &c_minors[c_idx..];
                    c_vals = &mut c_vals[c_idx..];
                }
            }
        }
        Op::Transpose(a) => {
            if beta != T::one() {
                for c_ij in c.values_mut() {
                    *c_ij *= beta.inlined_clone();
                }
            }

            for (i, a_lane_i) in a.lane_iter().enumerate() {
                for (&j, a_val) in a_lane_i.minor_indices().iter().zip(a_lane_i.values()) {
                    let a_val = a_val.inlined_clone();
                    let alpha = alpha.inlined_clone();
                    match c.get_entry_mut(j, i).unwrap() {
                        SparseEntryMut::NonZero(c_ji) => *c_ji += alpha * a_val,
                        SparseEntryMut::Zero => return Err(spadd_cs_unexpected_entry()),
                    }
                }
            }
        }
    }
    Ok(())
}

/// Helper functionality for implementing CSR/CSC SPMM.
///
/// The implementation essentially assumes that `a` is a CSR matrix. To use it with CSC matrices,
/// the transposed operation must be specified for the CSC matrix.
pub fn spmm_cs_dense<T>(
    beta: T,
    mut c: DMatrixSliceMut<T>,
    alpha: T,
    a: Op<&CsMatrix<T>>,
    b: Op<DMatrixSlice<T>>,
) where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One,
{
    match a {
        Op::NoOp(a) => {
            for j in 0..c.ncols() {
                let mut c_col_j = c.column_mut(j);
                for (c_ij, a_row_i) in c_col_j.iter_mut().zip(a.lane_iter()) {
                    let mut dot_ij = T::zero();
                    for (&k, a_ik) in a_row_i.minor_indices().iter().zip(a_row_i.values()) {
                        let b_contrib = match b {
                            Op::NoOp(ref b) => b.index((k, j)),
                            Op::Transpose(ref b) => b.index((j, k)),
                        };
                        dot_ij += a_ik.inlined_clone() * b_contrib.inlined_clone();
                    }
                    *c_ij = beta.inlined_clone() * c_ij.inlined_clone()
                        + alpha.inlined_clone() * dot_ij;
                }
            }
        }
        Op::Transpose(a) => {
            // In this case, we have to pre-multiply C by beta
            c *= beta;

            for k in 0..a.pattern().major_dim() {
                let a_row_k = a.get_lane(k).unwrap();
                for (&i, a_ki) in a_row_k.minor_indices().iter().zip(a_row_k.values()) {
                    let gamma_ki = alpha.inlined_clone() * a_ki.inlined_clone();
                    let mut c_row_i = c.row_mut(i);
                    match b {
                        Op::NoOp(ref b) => {
                            let b_row_k = b.row(k);
                            for (c_ij, b_kj) in c_row_i.iter_mut().zip(b_row_k.iter()) {
                                *c_ij += gamma_ki.inlined_clone() * b_kj.inlined_clone();
                            }
                        }
                        Op::Transpose(ref b) => {
                            let b_col_k = b.column(k);
                            for (c_ij, b_jk) in c_row_i.iter_mut().zip(b_col_k.iter()) {
                                *c_ij += gamma_ki.inlined_clone() * b_jk.inlined_clone();
                            }
                        }
                    }
                }
            }
        }
    }
}
