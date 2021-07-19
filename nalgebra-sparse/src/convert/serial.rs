//! Serial routines for converting between matrix formats.
//!
//! All routines in this module are single-threaded. At present these routines offer no
//! advantage over using the [`From`] trait, but future changes to the API might offer more
//! control to the user.
use std::ops::Add;

use num_traits::Zero;

use nalgebra::storage::Storage;
use nalgebra::{ClosedAdd, DMatrix, Dim, Matrix, Scalar};

use crate::coo::CooMatrix;
use crate::cs;
use crate::csc::CscMatrix;
use crate::csr::CsrMatrix;

/// Converts a dense matrix to [`CooMatrix`].
pub fn convert_dense_coo<T, R: Dim, C: Dim, S>(dense: &Matrix<T, R, C, S>) -> CooMatrix<T>
where
    T: Scalar + Zero + PartialEq,
    S: Storage<T, R, C>,
{
    let mut coo = CooMatrix::new(dense.nrows(), dense.ncols());

    for (index, v) in dense.iter().enumerate() {
        if v != &T::zero() {
            // We use the fact that matrix iteration is guaranteed to be column-major
            let i = index % dense.nrows();
            let j = index / dense.nrows();
            coo.push(i, j, v.inlined_clone());
        }
    }

    coo
}

/// Converts a [`CooMatrix`] to a dense matrix.
pub fn convert_coo_dense<T>(coo: &CooMatrix<T>) -> DMatrix<T>
where
    T: Scalar + Zero + ClosedAdd,
{
    let mut output = DMatrix::repeat(coo.nrows(), coo.ncols(), T::zero());
    for (i, j, v) in coo.triplet_iter() {
        output[(i, j)] += v.inlined_clone();
    }
    output
}

/// Converts a [`CooMatrix`] to a [`CsrMatrix`].
pub fn convert_coo_csr<T>(coo: &CooMatrix<T>) -> CsrMatrix<T>
where
    T: Scalar + Zero,
{
    let (offsets, indices, values) = convert_coo_cs(
        coo.nrows(),
        coo.row_indices(),
        coo.col_indices(),
        coo.values(),
    );

    // TODO: Avoid "try_from" since it validates the data? (requires unsafe, should benchmark
    // to see if it can be justified for performance reasons)
    CsrMatrix::try_from_csr_data(coo.nrows(), coo.ncols(), offsets, indices, values)
        .expect("Internal error: Invalid CSR data during COO->CSR conversion")
}

/// Converts a [`CsrMatrix`] to a [`CooMatrix`].
pub fn convert_csr_coo<T: Scalar>(csr: &CsrMatrix<T>) -> CooMatrix<T> {
    let mut result = CooMatrix::new(csr.nrows(), csr.ncols());
    for (i, j, v) in csr.triplet_iter() {
        result.push(i, j, v.inlined_clone());
    }
    result
}

/// Converts a [`CsrMatrix`] to a dense matrix.
pub fn convert_csr_dense<T>(csr: &CsrMatrix<T>) -> DMatrix<T>
where
    T: Scalar + ClosedAdd + Zero,
{
    let mut output = DMatrix::zeros(csr.nrows(), csr.ncols());

    for (i, j, v) in csr.triplet_iter() {
        output[(i, j)] += v.inlined_clone();
    }

    output
}

/// Converts a dense matrix to a [`CsrMatrix`].
pub fn convert_dense_csr<T, R, C, S>(dense: &Matrix<T, R, C, S>) -> CsrMatrix<T>
where
    T: Scalar + Zero + PartialEq,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
{
    let mut row_offsets = Vec::with_capacity(dense.nrows() + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    // We have to iterate row-by-row to build the CSR matrix, which is at odds with
    // nalgebra's column-major storage. The alternative would be to perform an initial sweep
    // to count number of non-zeros per row.
    row_offsets.push(0);
    for i in 0..dense.nrows() {
        for j in 0..dense.ncols() {
            let v = dense.index((i, j));
            if v != &T::zero() {
                col_idx.push(j);
                values.push(v.inlined_clone());
            }
        }
        row_offsets.push(col_idx.len());
    }

    // TODO: Consider circumventing the data validity check here
    // (would require unsafe, should benchmark)
    CsrMatrix::try_from_csr_data(dense.nrows(), dense.ncols(), row_offsets, col_idx, values)
        .expect("Internal error: Invalid CsrMatrix format during dense-> CSR conversion")
}

/// Converts a [`CooMatrix`] to a [`CscMatrix`].
pub fn convert_coo_csc<T>(coo: &CooMatrix<T>) -> CscMatrix<T>
where
    T: Scalar + Zero,
{
    let (offsets, indices, values) = convert_coo_cs(
        coo.ncols(),
        coo.col_indices(),
        coo.row_indices(),
        coo.values(),
    );

    // TODO: Avoid "try_from" since it validates the data? (requires unsafe, should benchmark
    // to see if it can be justified for performance reasons)
    CscMatrix::try_from_csc_data(coo.nrows(), coo.ncols(), offsets, indices, values)
        .expect("Internal error: Invalid CSC data during COO->CSC conversion")
}

/// Converts a [`CscMatrix`] to a [`CooMatrix`].
pub fn convert_csc_coo<T>(csc: &CscMatrix<T>) -> CooMatrix<T>
where
    T: Scalar,
{
    let mut coo = CooMatrix::new(csc.nrows(), csc.ncols());
    for (i, j, v) in csc.triplet_iter() {
        coo.push(i, j, v.inlined_clone());
    }
    coo
}

/// Converts a [`CscMatrix`] to a dense matrix.
pub fn convert_csc_dense<T>(csc: &CscMatrix<T>) -> DMatrix<T>
where
    T: Scalar + ClosedAdd + Zero,
{
    let mut output = DMatrix::zeros(csc.nrows(), csc.ncols());

    for (i, j, v) in csc.triplet_iter() {
        output[(i, j)] += v.inlined_clone();
    }

    output
}

/// Converts a dense matrix to a [`CscMatrix`].
pub fn convert_dense_csc<T, R, C, S>(dense: &Matrix<T, R, C, S>) -> CscMatrix<T>
where
    T: Scalar + Zero + PartialEq,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
{
    let mut col_offsets = Vec::with_capacity(dense.ncols() + 1);
    let mut row_idx = Vec::new();
    let mut values = Vec::new();

    col_offsets.push(0);
    for j in 0..dense.ncols() {
        for i in 0..dense.nrows() {
            let v = dense.index((i, j));
            if v != &T::zero() {
                row_idx.push(i);
                values.push(v.inlined_clone());
            }
        }
        col_offsets.push(row_idx.len());
    }

    // TODO: Consider circumventing the data validity check here
    // (would require unsafe, should benchmark)
    CscMatrix::try_from_csc_data(dense.nrows(), dense.ncols(), col_offsets, row_idx, values)
        .expect("Internal error: Invalid CscMatrix format during dense-> CSC conversion")
}

/// Converts a [`CsrMatrix`] to a [`CscMatrix`].
pub fn convert_csr_csc<T>(csr: &CsrMatrix<T>) -> CscMatrix<T>
where
    T: Scalar,
{
    let (offsets, indices, values) = cs::transpose_cs(
        csr.nrows(),
        csr.ncols(),
        csr.row_offsets(),
        csr.col_indices(),
        csr.values(),
    );

    // TODO: Avoid data validity check?
    CscMatrix::try_from_csc_data(csr.nrows(), csr.ncols(), offsets, indices, values)
        .expect("Internal error: Invalid CSC data during CSR->CSC conversion")
}

/// Converts a [`CscMatrix`] to a [`CsrMatrix`].
pub fn convert_csc_csr<T>(csc: &CscMatrix<T>) -> CsrMatrix<T>
where
    T: Scalar,
{
    let (offsets, indices, values) = cs::transpose_cs(
        csc.ncols(),
        csc.nrows(),
        csc.col_offsets(),
        csc.row_indices(),
        csc.values(),
    );

    // TODO: Avoid data validity check?
    CsrMatrix::try_from_csr_data(csc.nrows(), csc.ncols(), offsets, indices, values)
        .expect("Internal error: Invalid CSR data during CSC->CSR conversion")
}

fn convert_coo_cs<T>(
    major_dim: usize,
    major_indices: &[usize],
    minor_indices: &[usize],
    values: &[T],
) -> (Vec<usize>, Vec<usize>, Vec<T>)
where
    T: Scalar + Zero,
{
    assert_eq!(major_indices.len(), minor_indices.len());
    assert_eq!(minor_indices.len(), values.len());
    let nnz = major_indices.len();

    let (unsorted_major_offsets, unsorted_minor_idx, unsorted_vals) = {
        let mut offsets = vec![0usize; major_dim + 1];
        let mut minor_idx = vec![0usize; nnz];
        let mut vals = vec![T::zero(); nnz];
        coo_to_unsorted_cs(
            &mut offsets,
            &mut minor_idx,
            &mut vals,
            major_dim,
            major_indices,
            minor_indices,
            values,
        );
        (offsets, minor_idx, vals)
    };

    // TODO: If input is sorted and/or without duplicates, we can avoid additional allocations
    // and work. Might want to take advantage of this.

    // At this point, assembly is essentially complete. However, we must ensure
    // that minor indices are sorted within each lane and without duplicates.
    let mut sorted_major_offsets = Vec::new();
    let mut sorted_minor_idx = Vec::new();
    let mut sorted_vals = Vec::new();

    sorted_major_offsets.push(0);

    // We need some temporary storage when working with each lane. Since lanes often have a
    // very small number of non-zero entries, we try to amortize allocations across
    // lanes by reusing workspace vectors
    let mut idx_workspace = Vec::new();
    let mut perm_workspace = Vec::new();
    let mut values_workspace = Vec::new();

    for lane in 0..major_dim {
        let begin = unsorted_major_offsets[lane];
        let end = unsorted_major_offsets[lane + 1];
        let count = end - begin;
        let range = begin..end;

        // Ensure that workspaces can hold enough data
        perm_workspace.resize(count, 0);
        idx_workspace.resize(count, 0);
        values_workspace.resize(count, T::zero());
        sort_lane(
            &mut idx_workspace[..count],
            &mut values_workspace[..count],
            &unsorted_minor_idx[range.clone()],
            &unsorted_vals[range.clone()],
            &mut perm_workspace[..count],
        );

        let sorted_ja_current_len = sorted_minor_idx.len();

        combine_duplicates(
            |idx| sorted_minor_idx.push(idx),
            |val| sorted_vals.push(val),
            &idx_workspace[..count],
            &values_workspace[..count],
            &Add::add,
        );

        let new_col_count = sorted_minor_idx.len() - sorted_ja_current_len;
        sorted_major_offsets.push(sorted_major_offsets.last().unwrap() + new_col_count);
    }

    (sorted_major_offsets, sorted_minor_idx, sorted_vals)
}

/// Converts matrix data given in triplet format to unsorted CSR/CSC, retaining any duplicated
/// indices.
///
/// Here `major/minor` is `row/col` for CSR and `col/row` for CSC.
fn coo_to_unsorted_cs<T: Clone>(
    major_offsets: &mut [usize],
    cs_minor_idx: &mut [usize],
    cs_values: &mut [T],
    major_dim: usize,
    major_indices: &[usize],
    minor_indices: &[usize],
    coo_values: &[T],
) {
    assert_eq!(major_offsets.len(), major_dim + 1);
    assert_eq!(cs_minor_idx.len(), cs_values.len());
    assert_eq!(cs_values.len(), major_indices.len());
    assert_eq!(major_indices.len(), minor_indices.len());
    assert_eq!(minor_indices.len(), coo_values.len());

    // Count the number of occurrences of each row
    for major_idx in major_indices {
        major_offsets[*major_idx] += 1;
    }

    cs::convert_counts_to_offsets(major_offsets);

    {
        // TODO: Instead of allocating a whole new vector storing the current counts,
        // I think it's possible to be a bit more clever by storing each count
        // in the last of the column indices for each row
        let mut current_counts = vec![0usize; major_dim + 1];
        let triplet_iter = major_indices.iter().zip(minor_indices).zip(coo_values);
        for ((i, j), value) in triplet_iter {
            let current_offset = major_offsets[*i] + current_counts[*i];
            cs_minor_idx[current_offset] = *j;
            cs_values[current_offset] = value.clone();
            current_counts[*i] += 1;
        }
    }
}

/// Sort the indices of the given lane.
///
/// The indices and values in `minor_idx` and `values` are sorted according to the
/// minor indices and stored in `minor_idx_result` and `values_result` respectively.
///
/// All input slices are expected to be of the same length. The contents of mutable slices
/// can be arbitrary, as they are anyway overwritten.
fn sort_lane<T: Clone>(
    minor_idx_result: &mut [usize],
    values_result: &mut [T],
    minor_idx: &[usize],
    values: &[T],
    workspace: &mut [usize],
) {
    assert_eq!(minor_idx_result.len(), values_result.len());
    assert_eq!(values_result.len(), minor_idx.len());
    assert_eq!(minor_idx.len(), values.len());
    assert_eq!(values.len(), workspace.len());

    let permutation = workspace;
    // Set permutation to identity
    for (i, p) in permutation.iter_mut().enumerate() {
        *p = i;
    }

    // Compute permutation needed to bring minor indices into sorted order
    // Note: Using sort_unstable here avoids internal allocations, which is crucial since
    // each lane might have a small number of elements
    permutation.sort_unstable_by_key(|idx| minor_idx[*idx]);

    apply_permutation(minor_idx_result, minor_idx, permutation);
    apply_permutation(values_result, values, permutation);
}

// TODO: Move this into `utils` or something?
fn apply_permutation<T: Clone>(out_slice: &mut [T], in_slice: &[T], permutation: &[usize]) {
    assert_eq!(out_slice.len(), in_slice.len());
    assert_eq!(out_slice.len(), permutation.len());
    for (out_element, old_pos) in out_slice.iter_mut().zip(permutation) {
        *out_element = in_slice[*old_pos].clone();
    }
}

/// Given *sorted* indices and corresponding scalar values, combines duplicates with the given
/// associative combiner and calls the provided produce methods with combined indices and values.
fn combine_duplicates<T: Clone>(
    mut produce_idx: impl FnMut(usize),
    mut produce_value: impl FnMut(T),
    idx_array: &[usize],
    values: &[T],
    combiner: impl Fn(T, T) -> T,
) {
    assert_eq!(idx_array.len(), values.len());

    let mut i = 0;
    while i < idx_array.len() {
        let idx = idx_array[i];
        let mut combined_value = values[i].clone();
        let mut j = i + 1;
        while j < idx_array.len() && idx_array[j] == idx {
            let j_val = values[j].clone();
            combined_value = combiner(combined_value, j_val);
            j += 1;
        }
        produce_idx(idx);
        produce_value(combined_value);
        i = j;
    }
}
