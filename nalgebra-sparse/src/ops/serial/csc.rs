use crate::csc::CscMatrix;
use crate::ops::serial::cs::{spadd_cs_prealloc, spmm_cs_dense, spmm_cs_prealloc};
use crate::ops::serial::{OperationError, OperationErrorKind};
use crate::ops::Op;
use nalgebra::{ClosedAdd, ClosedMul, DMatrixSlice, DMatrixSliceMut, RealField, Scalar};
use num_traits::{One, Zero};

use std::borrow::Cow;

/// Sparse-dense matrix-matrix multiplication `C <- beta * C + alpha * op(A) * op(B)`.
///
/// # Panics
///
/// Panics if the dimensions of the matrices involved are not compatible with the expression.
pub fn spmm_csc_dense<'a, T>(
    beta: T,
    c: impl Into<DMatrixSliceMut<'a, T>>,
    alpha: T,
    a: Op<&CscMatrix<T>>,
    b: Op<impl Into<DMatrixSlice<'a, T>>>,
) where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One,
{
    let b = b.convert();
    spmm_csc_dense_(beta, c.into(), alpha, a, b)
}

fn spmm_csc_dense_<T>(
    beta: T,
    c: DMatrixSliceMut<T>,
    alpha: T,
    a: Op<&CscMatrix<T>>,
    b: Op<DMatrixSlice<T>>,
) where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One,
{
    assert_compatible_spmm_dims!(c, a, b);
    // Need to interpret matrix as transposed since the spmm_cs_dense function assumes CSR layout
    let a = a.transposed().map_same_op(|a| &a.cs);
    spmm_cs_dense(beta, c, alpha, a, b)
}

/// Sparse matrix addition `C <- beta * C + alpha * op(A)`.
///
/// If the pattern of `c` does not accommodate all the non-zero entries in `a`, an error is
/// returned.
///
/// # Panics
///
/// Panics if the dimensions of the matrices involved are not compatible with the expression.
pub fn spadd_csc_prealloc<T>(
    beta: T,
    c: &mut CscMatrix<T>,
    alpha: T,
    a: Op<&CscMatrix<T>>,
) -> Result<(), OperationError>
where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One+PartialEq,
{
    assert_compatible_spadd_dims!(c, a);
    spadd_cs_prealloc(beta, &mut c.cs, alpha, a.map_same_op(|a| &a.cs))
}

/// Sparse-sparse matrix multiplication, `C <- beta * C + alpha * op(A) * op(B)`.
///
/// # Errors
///
/// If the sparsity pattern of `C` is not able to store the result of the operation,
/// an error is returned.
///
/// # Panics
///
/// Panics if the dimensions of the matrices involved are not compatible with the expression.
pub fn spmm_csc_prealloc<T>(
    beta: T,
    c: &mut CscMatrix<T>,
    alpha: T,
    a: Op<&CscMatrix<T>>,
    b: Op<&CscMatrix<T>>,
) -> Result<(), OperationError>
where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One,
{
    assert_compatible_spmm_dims!(c, a, b);

    use Op::{NoOp, Transpose};

    match (&a, &b) {
        (NoOp(ref a), NoOp(ref b)) => {
            // Note: We have to reverse the order for CSC matrices
            spmm_cs_prealloc(beta, &mut c.cs, alpha, &b.cs, &a.cs)
        }
        _ => {
            // Currently we handle transposition by explicitly precomputing transposed matrices
            // and calling the operation again without transposition
            let a_ref: &CscMatrix<T> = a.inner_ref();
            let b_ref: &CscMatrix<T> = b.inner_ref();
            let (a, b) = {
                use Cow::*;
                match (&a, &b) {
                    (NoOp(_), NoOp(_)) => unreachable!(),
                    (Transpose(ref a), NoOp(_)) => (Owned(a.transpose()), Borrowed(b_ref)),
                    (NoOp(_), Transpose(ref b)) => (Borrowed(a_ref), Owned(b.transpose())),
                    (Transpose(ref a), Transpose(ref b)) => {
                        (Owned(a.transpose()), Owned(b.transpose()))
                    }
                }
            };

            spmm_csc_prealloc(beta, c, alpha, NoOp(a.as_ref()), NoOp(b.as_ref()))
        }
    }
}

/// Solve the lower triangular system `op(L) X = B`.
///
/// Only the lower triangular part of L is read, and the result is stored in B.
///
/// # Errors
///
/// An error is returned if the system can not be solved due to the matrix being singular.
///
/// # Panics
///
/// Panics if `L` is not square, or if `L` and `B` are not dimensionally compatible.
pub fn spsolve_csc_lower_triangular<'a, T: RealField>(
    l: Op<&CscMatrix<T>>,
    b: impl Into<DMatrixSliceMut<'a, T>>,
) -> Result<(), OperationError> {
    let b = b.into();
    let l_matrix = l.into_inner();
    assert_eq!(
        l_matrix.nrows(),
        l_matrix.ncols(),
        "Matrix must be square for triangular solve."
    );
    assert_eq!(
        l_matrix.nrows(),
        b.nrows(),
        "Dimension mismatch in sparse lower triangular solver."
    );
    match l {
        Op::NoOp(a) => spsolve_csc_lower_triangular_no_transpose(a, b),
        Op::Transpose(a) => spsolve_csc_lower_triangular_transpose(a, b),
    }
}

fn spsolve_csc_lower_triangular_no_transpose<T: RealField>(
    l: &CscMatrix<T>,
    b: DMatrixSliceMut<T>,
) -> Result<(), OperationError> {
    let mut x = b;

    // Solve column-by-column
    for j in 0..x.ncols() {
        let mut x_col_j = x.column_mut(j);

        for k in 0..l.ncols() {
            let l_col_k = l.col(k);

            // Skip entries above the diagonal
            // TODO: Can use exponential search here to quickly skip entries
            // (we'd like to avoid using binary search as it's very cache unfriendly
            // and the matrix might actually *be* lower triangular, which would induce
            // a severe penalty)
            let diag_csc_index = l_col_k.row_indices().iter().position(|&i| i == k);
            if let Some(diag_csc_index) = diag_csc_index {
                let l_kk = l_col_k.values()[diag_csc_index];

                if l_kk != T::zero() {
                    // Update entry associated with diagonal
                    x_col_j[k] /= l_kk;
                    // Copy value after updating (so we don't run into the borrow checker)
                    let x_kj = x_col_j[k];

                    let row_indices = &l_col_k.row_indices()[(diag_csc_index + 1)..];
                    let l_values = &l_col_k.values()[(diag_csc_index + 1)..];

                    // Note: The remaining entries are below the diagonal
                    for (&i, l_ik) in row_indices.iter().zip(l_values) {
                        let x_ij = &mut x_col_j[i];
                        *x_ij -= l_ik.inlined_clone() * x_kj;
                    }

                    x_col_j[k] = x_kj;
                } else {
                    return spsolve_encountered_zero_diagonal();
                }
            } else {
                return spsolve_encountered_zero_diagonal();
            }
        }
    }

    Ok(())
}

fn spsolve_encountered_zero_diagonal() -> Result<(), OperationError> {
    let message = "Matrix contains at least one diagonal entry that is zero.";
    Err(OperationError::from_kind_and_message(
        OperationErrorKind::Singular,
        String::from(message),
    ))
}

fn spsolve_csc_lower_triangular_transpose<T: RealField>(
    l: &CscMatrix<T>,
    b: DMatrixSliceMut<T>,
) -> Result<(), OperationError> {
    let mut x = b;

    // Solve column-by-column
    for j in 0..x.ncols() {
        let mut x_col_j = x.column_mut(j);

        // Due to the transposition, we're essentially solving an upper triangular system,
        // and the columns in our matrix become rows

        for i in (0..l.ncols()).rev() {
            let l_col_i = l.col(i);

            // Skip entries above the diagonal
            // TODO: Can use exponential search here to quickly skip entries
            let diag_csc_index = l_col_i.row_indices().iter().position(|&k| i == k);
            if let Some(diag_csc_index) = diag_csc_index {
                let l_ii = l_col_i.values()[diag_csc_index];

                if l_ii != T::zero() {
                    // // Update entry associated with diagonal
                    // x_col_j[k] /= a_kk;

                    // Copy value after updating (so we don't run into the borrow checker)
                    let mut x_ii = x_col_j[i];

                    let row_indices = &l_col_i.row_indices()[(diag_csc_index + 1)..];
                    let a_values = &l_col_i.values()[(diag_csc_index + 1)..];

                    // Note: The remaining entries are below the diagonal
                    for (&k, &l_ki) in row_indices.iter().zip(a_values) {
                        let x_kj = x_col_j[k];
                        x_ii -= l_ki * x_kj;
                    }

                    x_col_j[i] = x_ii / l_ii;
                } else {
                    return spsolve_encountered_zero_diagonal();
                }
            } else {
                return spsolve_encountered_zero_diagonal();
            }
        }
    }

    Ok(())
}
