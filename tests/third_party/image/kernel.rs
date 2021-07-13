use na::{Matrix3, MatrixMN, U10, U8};
use std::panic;

#[test]
fn image_convolve_check() {
    // Static Tests
    let expect = MatrixMN::<usize, U8, U8>::from_element(18);
    let src = MatrixMN::<usize, U10, U10>::from_element(2);
    let kernel = Matrix3::<usize>::from_element(1);
    let result = src.filter_2d(kernel);
    println!("src: {}", src);
    println!("ker: {}", kernel);
    println!("res: {}", result);

    assert_eq!(result, expect);
}
