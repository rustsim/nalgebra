use crate::SparseFormatError;

/// A representation of the sparsity pattern of a CSR or CSC matrix.
///
/// ## Format specification
///
/// TODO: Write this out properly
///
/// - offsets[0] == 0
/// - Major offsets must be monotonically increasing
/// - major_offsets.len() == major_dim + 1
/// - Column indices within each lane must be sorted
/// - Column indices must be in-bounds
/// - The last entry in major offsets must correspond to the number of minor indices
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
    pub fn new(major_dim: usize, minor_dim: usize) -> Self {
        Self {
            major_offsets: vec![0; major_dim + 1],
            minor_indices: vec![],
            minor_dim,
        }
    }

    /// The offsets for the major dimension.
    #[inline]
    pub fn major_offsets(&self) -> &[usize] {
        &self.major_offsets
    }

    /// The indices for the minor dimension.
    #[inline]
    pub fn minor_indices(&self) -> &[usize] {
        &self.minor_indices
    }

    /// The major dimension.
    #[inline]
    pub fn major_dim(&self) -> usize {
        assert!(self.major_offsets.len() > 0);
        self.major_offsets.len() - 1
    }

    /// The minor dimension.
    #[inline]
    pub fn minor_dim(&self) -> usize {
        self.minor_dim
    }

    /// The number of "non-zeros", i.e. explicitly stored entries in the pattern.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.minor_indices.len()
    }

    /// Get the lane at the given index.
    #[inline]
    pub fn lane(&self, major_index: usize) -> Option<&[usize]> {
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
    ) -> Result<Self, SparseFormatError> {
        // TODO: If these errors are *directly* propagated to errors from e.g.
        // CSR construction, the error messages will be confusing to users,
        // as the error messages refer to "major" and "minor" lanes, as opposed to
        // rows and columns

        if major_offsets.len() != major_dim + 1 {
            return Err(SparseFormatError::InvalidStructure(
                Box::from("Size of major_offsets must be equal to (major_dim + 1)")));
        }

        // Check that the first and last offsets conform to the specification
        {
            if *major_offsets.first().unwrap() != 0 {
                return Err(SparseFormatError::InvalidStructure(
                    Box::from("First entry in major_offsets must always be 0.")
                ));
            } else if *major_offsets.last().unwrap() != minor_indices.len() {
                return Err(SparseFormatError::InvalidStructure(
                    Box::from("Last entry in major_offsets must always be equal to minor_indices.len()")
                ));
            }
        }

        // Test that each lane has strictly monotonically increasing minor indices, i.e.
        // minor indices within a lane are sorted, unique. In addition, each minor index
        // must be in bounds with respect to the minor dimension.
        {
            for lane_idx in 0 .. major_dim {
                let range_start = major_offsets[lane_idx];
                let range_end = major_offsets[lane_idx + 1];

                // Test that major offsets are monotonically increasing
                if range_start > range_end {
                    return Err(SparseFormatError::InvalidStructure(
                        Box::from("Major offsets are not monotonically increasing.")
                    ));
                }

                let minor_indices = &minor_indices[range_start .. range_end];

                // We test for in-bounds, uniqueness and monotonicity at the same time
                // to ensure that we only visit each minor index once
                let mut iter = minor_indices.iter();
                let mut prev = None;

                while let Some(next) = iter.next().copied() {
                    if next > minor_dim {
                        return Err(SparseFormatError::IndexOutOfBounds(
                            Box::from("Minor index out of bounds.")
                        ));
                    }

                    if let Some(prev) = prev {
                        if prev > next {
                            return Err(SparseFormatError::InvalidStructure(
                                Box::from("Minor indices within a lane must be monotonically increasing (sorted).")
                            ));
                        } else if prev == next {
                            return Err(SparseFormatError::DuplicateEntry(
                                Box::from("Duplicate minor entries detected.")
                            ));
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
    /// # use nalgebra_sparse::{SparsityPattern};
    /// let offsets = vec![0, 2, 3, 4];
    /// let minor_indices = vec![0, 2, 1, 0];
    /// let pattern = SparsityPattern::try_from_offsets_and_indices(3, 4, offsets, minor_indices)
    ///     .unwrap();
    ///
    /// let entries: Vec<_> = pattern.entries().collect();
    /// assert_eq!(entries, vec![(0, 0), (0, 2), (1, 1), (2, 0)]);
    /// ```
    ///
    pub fn entries(&self) -> SparsityPatternIter {
        SparsityPatternIter::from_pattern(self)
    }
}

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
        let minors_in_first_lane = &pattern.minor_indices()[0 .. *first_lane_end];
        Self {
            major_offsets: pattern.major_offsets(),
            minor_indices: pattern.minor_indices(),
            current_lane_idx: 0,
            remaining_minors_in_lane: minors_in_first_lane
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
                        self.remaining_minors_in_lane = &self.minor_indices[(lower + 1) .. upper];
                        return Some((self.current_lane_idx, self.minor_indices[lower]))
                    }
                }
            }
        }
    }
}
