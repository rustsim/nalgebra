//! Sparsity patterns for CSR and CSC matrices.
use crate::cs::transpose_cs;
use crate::SparseFormatError;
use std::error::Error;
use std::fmt;

/// A representation of the sparsity pattern of a CSR or CSC matrix.
///
/// CSR and CSC matrices store matrices in a very similar fashion. In fact, in a certain sense,
/// they are transposed. More precisely, when reinterpreting the three data arrays of a CSR
/// matrix as a CSC matrix, we obtain the CSC representation of its transpose.
///
/// [`SparsityPattern`] is an abstraction built on this observation. Whereas CSR matrices
/// store a matrix row-by-row, and a CSC matrix stores a matrix column-by-column, a
/// `SparsityPattern` represents only the index data structure of a matrix *lane-by-lane*.
/// Here, a *lane* is a generalization of rows and columns. We further define *major lanes*
/// and *minor lanes*. The sparsity pattern of a CSR matrix is then obtained by interpreting
/// major/minor as row/column. Conversely, we obtain the sparsity pattern of a CSC matrix by
/// interpreting major/minor as column/row.
///
/// This allows us to use a common abstraction to talk about sparsity patterns of CSR and CSC
/// matrices. This is convenient, because at the abstract level, the invariants of the formats
/// are the same. Hence we may encode the invariants of the index data structure separately from
/// the scalar values of the matrix. This is especially useful in applications where the
/// sparsity pattern is built ahead of the matrix values, or the same sparsity pattern is re-used
/// between different matrices. Finally, we can use `SparsityPattern` to encode adjacency
/// information in graphs.
///
/// # Format
///
/// The format is exactly the same as for the index data structures of CSR and CSC matrices.
/// This means that the sparsity pattern of an `m x n` sparse matrix with `nnz` non-zeros,
/// where in this case `m x n` does *not* mean `rows x columns`, but rather `majors x minors`,
/// is represented by the following two arrays:
///
/// - `major_offsets`, an array of integers with length `m + 1`.
/// - `minor_indices`, an array of integers with length `nnz`.
///
/// The invariants and relationship between `major_offsets` and `minor_indices` remain the same
/// as for `row_offsets` and `col_indices` in the [CSR](`crate::csr::CsrMatrix`) format
/// specification.
#[derive(Debug, Clone, PartialEq, Eq)]
// TODO: Make SparsityPattern parametrized by index type
// (need a solid abstraction for index types though)
pub struct SparsityPattern {
    major_offsets: Vec<usize>,
    minor_indices: Vec<usize>,
    minor_dim: usize,
}

impl SparsityPattern {
    /// Create a sparsity pattern of the given dimensions without explicitly stored entries.
    pub fn zeros(major_dim: usize, minor_dim: usize) -> Self {
        Self {
            major_offsets: vec![0; major_dim + 1],
            minor_indices: vec![],
            minor_dim,
        }
    }

    /// The offsets for the major dimension.
    #[inline]
    #[must_use]
    pub fn major_offsets(&self) -> &[usize] {
        &self.major_offsets
    }

    /// The indices for the minor dimension.
    #[inline]
    #[must_use]
    pub fn minor_indices(&self) -> &[usize] {
        &self.minor_indices
    }

    /// The number of major lanes in the pattern.
    #[inline]
    #[must_use]
    pub fn major_dim(&self) -> usize {
        assert!(self.major_offsets.len() > 0);
        self.major_offsets.len() - 1
    }

    /// The number of minor lanes in the pattern.
    #[inline]
    #[must_use]
    pub fn minor_dim(&self) -> usize {
        self.minor_dim
    }

    /// The number of "non-zeros", i.e. explicitly stored entries in the pattern.
    #[inline]
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.minor_indices.len()
    }

    /// Get the lane at the given index.
    ///
    /// Panics
    /// ------
    ///
    /// Panics if `major_index` is out of bounds.
    #[inline]
    #[must_use]
    pub fn lane(&self, major_index: usize) -> &[usize] {
        self.get_lane(major_index).unwrap()
    }

    /// Get the lane at the given index, or `None` if out of bounds.
    #[inline]
    #[must_use]
    pub fn get_lane(&self, major_index: usize) -> Option<&[usize]> {
        let offset_begin = *self.major_offsets().get(major_index)?;
        let offset_end = *self.major_offsets().get(major_index + 1)?;
        Some(&self.minor_indices()[offset_begin..offset_end])
    }

    /// Try to construct a sparsity pattern from the given dimensions, major offsets
    /// and minor indices.
    ///
    /// Returns an error if the data does not conform to the requirements.
    pub fn try_from_offsets_and_indices(
        major_dim: usize,
        minor_dim: usize,
        major_offsets: Vec<usize>,
        minor_indices: Vec<usize>,
    ) -> Result<Self, SparsityPatternFormatError> {
        use SparsityPatternFormatError::*;

        if major_offsets.len() != major_dim + 1 {
            return Err(InvalidOffsetArrayLength);
        }

        // Check that the first and last offsets conform to the specification
        {
            let first_offset_ok = *major_offsets.first().unwrap() == 0;
            let last_offset_ok = *major_offsets.last().unwrap() == minor_indices.len();
            if !first_offset_ok || !last_offset_ok {
                return Err(InvalidOffsetFirstLast);
            }
        }

        // Test that each lane has strictly monotonically increasing minor indices, i.e.
        // minor indices within a lane are sorted, unique. In addition, each minor index
        // must be in bounds with respect to the minor dimension.
        {
            for lane_idx in 0..major_dim {
                let range_start = major_offsets[lane_idx];
                let range_end = major_offsets[lane_idx + 1];

                // Test that major offsets are monotonically increasing
                if range_start > range_end {
                    return Err(NonmonotonicOffsets);
                }

                let minor_indices = &minor_indices[range_start..range_end];

                // We test for in-bounds, uniqueness and monotonicity at the same time
                // to ensure that we only visit each minor index once
                let mut iter = minor_indices.iter();
                let mut prev = None;

                while let Some(next) = iter.next().copied() {
                    if next >= minor_dim {
                        return Err(MinorIndexOutOfBounds);
                    }

                    if let Some(prev) = prev {
                        if prev > next {
                            return Err(NonmonotonicMinorIndices);
                        } else if prev == next {
                            return Err(DuplicateEntry);
                        }
                    }
                    prev = Some(next);
                }
            }
        }

        Ok(Self {
            major_offsets,
            minor_indices,
            minor_dim,
        })
    }

    /// An iterator over the explicitly stored "non-zero" entries (i, j).
    ///
    /// The iteration happens in a lane-major fashion, meaning that the lane index i
    /// increases monotonically, and the minor index j increases monotonically within each
    /// lane i.
    ///
    /// Examples
    /// --------
    ///
    /// ```
    /// # use nalgebra_sparse::pattern::SparsityPattern;
    /// let offsets = vec![0, 2, 3, 4];
    /// let minor_indices = vec![0, 2, 1, 0];
    /// let pattern = SparsityPattern::try_from_offsets_and_indices(3, 4, offsets, minor_indices)
    ///     .unwrap();
    ///
    /// let entries: Vec<_> = pattern.entries().collect();
    /// assert_eq!(entries, vec![(0, 0), (0, 2), (1, 1), (2, 0)]);
    /// ```
    ///
    #[must_use]
    pub fn entries(&self) -> SparsityPatternIter {
        SparsityPatternIter::from_pattern(self)
    }

    /// Returns the raw offset and index data for the sparsity pattern.
    ///
    /// Examples
    /// --------
    ///
    /// ```
    /// # use nalgebra_sparse::pattern::SparsityPattern;
    /// let offsets = vec![0, 2, 3, 4];
    /// let minor_indices = vec![0, 2, 1, 0];
    /// let pattern = SparsityPattern::try_from_offsets_and_indices(
    ///         3,
    ///         4,
    ///         offsets.clone(),
    ///         minor_indices.clone())
    ///     .unwrap();
    /// let (offsets2, minor_indices2) = pattern.disassemble();
    /// assert_eq!(offsets2, offsets);
    /// assert_eq!(minor_indices2, minor_indices);
    /// ```
    pub fn disassemble(self) -> (Vec<usize>, Vec<usize>) {
        (self.major_offsets, self.minor_indices)
    }

    /// Computes the transpose of the sparsity pattern.
    ///
    /// This is analogous to matrix transposition, i.e. an entry `(i, j)` becomes `(j, i)` in the
    /// new pattern.
    #[must_use]
    pub fn transpose(&self) -> Self {
        // By using unit () values, we can use the same routines as for CSR/CSC matrices
        let values = vec![(); self.nnz()];
        let (new_offsets, new_indices, _) = transpose_cs(
            self.major_dim(),
            self.minor_dim(),
            self.major_offsets(),
            self.minor_indices(),
            &values,
        );
        // TODO: Skip checks
        Self::try_from_offsets_and_indices(
            self.minor_dim(),
            self.major_dim(),
            new_offsets,
            new_indices,
        )
        .expect("Internal error: Transpose should never fail.")
    }
}

/// Error type for `SparsityPattern` format errors.
#[non_exhaustive]
#[derive(Debug, PartialEq, Eq)]
pub enum SparsityPatternFormatError {
    /// Indicates an invalid number of offsets.
    ///
    /// The number of offsets must be equal to (major_dim + 1).
    InvalidOffsetArrayLength,
    /// Indicates that the first or last entry in the offset array did not conform to
    /// specifications.
    ///
    /// The first entry must be 0, and the last entry must be exactly one greater than the
    /// major dimension.
    InvalidOffsetFirstLast,
    /// Indicates that the major offsets are not monotonically increasing.
    NonmonotonicOffsets,
    /// One or more minor indices are out of bounds.
    MinorIndexOutOfBounds,
    /// One or more duplicate entries were detected.
    ///
    /// Two entries are considered duplicates if they are part of the same major lane and have
    /// the same minor index.
    DuplicateEntry,
    /// Indicates that minor indices are not monotonically increasing within each lane.
    NonmonotonicMinorIndices,
}

impl From<SparsityPatternFormatError> for SparseFormatError {
    fn from(err: SparsityPatternFormatError) -> Self {
        use crate::SparseFormatErrorKind;
        use crate::SparseFormatErrorKind::*;
        use SparsityPatternFormatError::DuplicateEntry as PatternDuplicateEntry;
        use SparsityPatternFormatError::*;
        match err {
            InvalidOffsetArrayLength
            | InvalidOffsetFirstLast
            | NonmonotonicOffsets
            | NonmonotonicMinorIndices => {
                SparseFormatError::from_kind_and_error(InvalidStructure, Box::from(err))
            }
            MinorIndexOutOfBounds => {
                SparseFormatError::from_kind_and_error(IndexOutOfBounds, Box::from(err))
            }
            PatternDuplicateEntry => SparseFormatError::from_kind_and_error(
                #[allow(unused_qualifications)]
                SparseFormatErrorKind::DuplicateEntry,
                Box::from(err),
            ),
        }
    }
}

impl fmt::Display for SparsityPatternFormatError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SparsityPatternFormatError::InvalidOffsetArrayLength => {
                write!(f, "Length of offset array is not equal to (major_dim + 1).")
            }
            SparsityPatternFormatError::InvalidOffsetFirstLast => {
                write!(f, "First or last offset is incompatible with format.")
            }
            SparsityPatternFormatError::NonmonotonicOffsets => {
                write!(f, "Offsets are not monotonically increasing.")
            }
            SparsityPatternFormatError::MinorIndexOutOfBounds => {
                write!(f, "A minor index is out of bounds.")
            }
            SparsityPatternFormatError::DuplicateEntry => {
                write!(f, "Input data contains duplicate entries.")
            }
            SparsityPatternFormatError::NonmonotonicMinorIndices => {
                write!(
                    f,
                    "Minor indices are not monotonically increasing within each lane."
                )
            }
        }
    }
}

impl Error for SparsityPatternFormatError {}

/// Iterator type for iterating over entries in a sparsity pattern.
#[derive(Debug, Clone)]
pub struct SparsityPatternIter<'a> {
    // See implementation of Iterator::next for an explanation of how these members are used
    major_offsets: &'a [usize],
    minor_indices: &'a [usize],
    current_lane_idx: usize,
    remaining_minors_in_lane: &'a [usize],
}

impl<'a> SparsityPatternIter<'a> {
    fn from_pattern(pattern: &'a SparsityPattern) -> Self {
        let first_lane_end = pattern.major_offsets().get(1).unwrap_or(&0);
        let minors_in_first_lane = &pattern.minor_indices()[0..*first_lane_end];
        Self {
            major_offsets: pattern.major_offsets(),
            minor_indices: pattern.minor_indices(),
            current_lane_idx: 0,
            remaining_minors_in_lane: minors_in_first_lane,
        }
    }
}

impl<'a> Iterator for SparsityPatternIter<'a> {
    type Item = (usize, usize);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // We ensure fast iteration across each lane by iteratively "draining" a slice
        // corresponding to the remaining column indices in the particular lane.
        // When we reach the end of this slice, we are at the end of a lane,
        // and we must do some bookkeeping for preparing the iteration of the next lane
        // (or stop iteration if we're through all lanes).
        // This way we can avoid doing unnecessary bookkeeping on every iteration,
        // instead paying a small price whenever we jump to a new lane.
        if let Some(minor_idx) = self.remaining_minors_in_lane.first() {
            let item = Some((self.current_lane_idx, *minor_idx));
            self.remaining_minors_in_lane = &self.remaining_minors_in_lane[1..];
            item
        } else {
            loop {
                // Keep skipping lanes until we found a non-empty lane or there are no more lanes
                if self.current_lane_idx + 2 >= self.major_offsets.len() {
                    // We've processed all lanes, so we're at the end of the iterator
                    // (note: keep in mind that offsets.len() == major_dim() + 1, hence we need +2)
                    return None;
                } else {
                    // Bump lane index and check if the lane is non-empty
                    self.current_lane_idx += 1;
                    let lower = self.major_offsets[self.current_lane_idx];
                    let upper = self.major_offsets[self.current_lane_idx + 1];
                    if upper > lower {
                        self.remaining_minors_in_lane = &self.minor_indices[(lower + 1)..upper];
                        return Some((self.current_lane_idx, self.minor_indices[lower]));
                    }
                }
            }
        }
    }
}
