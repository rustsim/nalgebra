#[cfg(not(all(
    feature = "debug",
    feature = "compare",
    feature = "rand",
    feature = "macros"
)))]
compile_error!(
    "Please enable the `debug`, `compare`, `rand` and `macros` features in order to compile and run the tests.
     Example: `cargo test --features debug,compare,rand,macros`"
);

#[cfg(feature = "abomonation-serialize")]
extern crate abomonation;
#[cfg(all(feature = "debug", feature = "compare", feature = "rand"))]
#[macro_use]
extern crate approx;
extern crate nalgebra as na;
extern crate num_traits as num;
#[cfg(feature = "rand")]
extern crate rand_package as rand;

#[cfg(all(feature = "debug", feature = "compare", feature = "rand"))]
mod core;
#[cfg(all(feature = "debug", feature = "compare", feature = "rand"))]
mod geometry;
#[cfg(all(feature = "debug", feature = "compare", feature = "rand"))]
mod linalg;

#[cfg(all(feature = "debug", feature = "compare", feature = "rand"))]
#[cfg(feature = "proptest-support")]
mod proptest;
mod third_party;

//#[cfg(all(feature = "debug", feature = "compare", feature = "rand"))]
//#[cfg(feature = "sparse")]
//mod sparse;
