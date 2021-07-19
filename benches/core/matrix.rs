use na::{
    Const, DMatrix, DVector, Dynamic, Matrix2, Matrix3, Matrix4, OMatrix, Vector2, Vector3,
    Vector4, U10,
};
use rand::Rng;
use rand_isaac::IsaacRng;
use std::ops::{Add, Div, Mul, Sub};

#[path = "../common/macros.rs"]
mod macros;

bench_binop!(mat2_mul_m, Matrix2<f32>, Matrix2<f32>, mul);
bench_binop!(mat3_mul_m, Matrix3<f32>, Matrix3<f32>, mul);
bench_binop!(mat4_mul_m, Matrix4<f32>, Matrix4<f32>, mul);

bench_binop_ref!(mat2_tr_mul_m, Matrix2<f32>, Matrix2<f32>, tr_mul);
bench_binop_ref!(mat3_tr_mul_m, Matrix3<f32>, Matrix3<f32>, tr_mul);
bench_binop_ref!(mat4_tr_mul_m, Matrix4<f32>, Matrix4<f32>, tr_mul);

bench_binop!(mat2_add_m, Matrix2<f32>, Matrix2<f32>, add);
bench_binop!(mat3_add_m, Matrix3<f32>, Matrix3<f32>, add);
bench_binop!(mat4_add_m, Matrix4<f32>, Matrix4<f32>, add);

bench_binop!(mat2_sub_m, Matrix2<f32>, Matrix2<f32>, sub);
bench_binop!(mat3_sub_m, Matrix3<f32>, Matrix3<f32>, sub);
bench_binop!(mat4_sub_m, Matrix4<f32>, Matrix4<f32>, sub);

bench_binop!(mat2_mul_v, Matrix2<f32>, Vector2<f32>, mul);
bench_binop!(mat3_mul_v, Matrix3<f32>, Vector3<f32>, mul);
bench_binop!(mat4_mul_v, Matrix4<f32>, Vector4<f32>, mul);

bench_binop_ref!(mat2_tr_mul_v, Matrix2<f32>, Vector2<f32>, tr_mul);
bench_binop_ref!(mat3_tr_mul_v, Matrix3<f32>, Vector3<f32>, tr_mul);
bench_binop_ref!(mat4_tr_mul_v, Matrix4<f32>, Vector4<f32>, tr_mul);

bench_binop!(mat2_mul_s, Matrix2<f32>, f32, mul);
bench_binop!(mat3_mul_s, Matrix3<f32>, f32, mul);
bench_binop!(mat4_mul_s, Matrix4<f32>, f32, mul);

bench_binop!(mat2_div_s, Matrix2<f32>, f32, div);
bench_binop!(mat3_div_s, Matrix3<f32>, f32, div);
bench_binop!(mat4_div_s, Matrix4<f32>, f32, div);

bench_unop!(mat2_inv, Matrix2<f32>, try_inverse);
bench_unop!(mat3_inv, Matrix3<f32>, try_inverse);
bench_unop!(mat4_inv, Matrix4<f32>, try_inverse);

bench_unop!(mat2_transpose, Matrix2<f32>, transpose);
bench_unop!(mat3_transpose, Matrix3<f32>, transpose);
bench_unop!(mat4_transpose, Matrix4<f32>, transpose);

fn mat_div_scalar(b: &mut criterion::Criterion) {
    let a = DMatrix::from_row_slice(1000, 1000, &vec![2.0; 1000000]);
    let n = 42.0;

    b.bench_function("mat_div_scalar", move |bh| {
        bh.iter(|| {
            let mut aa = a.clone();
            let mut b = aa.slice_mut((0, 0), (1000, 1000));
            b /= n
        })
    });
}

fn mat100_add_mat100(bench: &mut criterion::Criterion) {
    let a = DMatrix::<f64>::new_random(100, 100);
    let b = DMatrix::<f64>::new_random(100, 100);

    bench.bench_function("mat100_add_mat100", move |bh| bh.iter(|| &a + &b));
}

fn mat4_mul_mat4(bench: &mut criterion::Criterion) {
    let a = DMatrix::<f64>::new_random(4, 4);
    let b = DMatrix::<f64>::new_random(4, 4);

    bench.bench_function("mat4_mul_mat4", move |bh| bh.iter(|| &a * &b));
}

fn mat5_mul_mat5(bench: &mut criterion::Criterion) {
    let a = DMatrix::<f64>::new_random(5, 5);
    let b = DMatrix::<f64>::new_random(5, 5);

    bench.bench_function("mat5_mul_mat5", move |bh| bh.iter(|| &a * &b));
}

fn mat6_mul_mat6(bench: &mut criterion::Criterion) {
    let a = DMatrix::<f64>::new_random(6, 6);
    let b = DMatrix::<f64>::new_random(6, 6);

    bench.bench_function("mat6_mul_mat6", move |bh| bh.iter(|| &a * &b));
}

fn mat7_mul_mat7(bench: &mut criterion::Criterion) {
    let a = DMatrix::<f64>::new_random(7, 7);
    let b = DMatrix::<f64>::new_random(7, 7);

    bench.bench_function("mat7_mul_mat7", move |bh| bh.iter(|| &a * &b));
}

fn mat8_mul_mat8(bench: &mut criterion::Criterion) {
    let a = DMatrix::<f64>::new_random(8, 8);
    let b = DMatrix::<f64>::new_random(8, 8);

    bench.bench_function("mat8_mul_mat8", move |bh| bh.iter(|| &a * &b));
}

fn mat9_mul_mat9(bench: &mut criterion::Criterion) {
    let a = DMatrix::<f64>::new_random(9, 9);
    let b = DMatrix::<f64>::new_random(9, 9);

    bench.bench_function("mat9_mul_mat9", move |bh| bh.iter(|| &a * &b));
}

fn mat10_mul_mat10(bench: &mut criterion::Criterion) {
    let a = DMatrix::<f64>::new_random(10, 10);
    let b = DMatrix::<f64>::new_random(10, 10);

    bench.bench_function("mat10_mul_mat10", move |bh| bh.iter(|| &a * &b));
}

fn mat10_mul_mat10_static(bench: &mut criterion::Criterion) {
    let a = OMatrix::<f64, U10, U10>::new_random();
    let b = OMatrix::<f64, U10, U10>::new_random();

    bench.bench_function("mat10_mul_mat10_static", move |bh| bh.iter(|| &a * &b));
}

fn mat100_mul_mat100(bench: &mut criterion::Criterion) {
    let a = DMatrix::<f64>::new_random(100, 100);
    let b = DMatrix::<f64>::new_random(100, 100);

    bench.bench_function("mat100_mul_mat100", move |bh| bh.iter(|| &a * &b));
}

fn mat500_mul_mat500(bench: &mut criterion::Criterion) {
    let a = DMatrix::<f64>::from_element(500, 500, 5f64);
    let b = DMatrix::<f64>::from_element(500, 500, 6f64);

    bench.bench_function("mat500_mul_mat500", move |bh| bh.iter(|| &a * &b));
}

fn iter(bench: &mut criterion::Criterion) {
    let a = DMatrix::<f64>::new_random(1000, 1000);

    bench.bench_function("iter", move |bh| {
        bh.iter(|| {
            for value in a.iter() {
                criterion::black_box(value);
            }
        })
    });
}

fn iter_rev(bench: &mut criterion::Criterion) {
    let a = DMatrix::<f64>::new_random(1000, 1000);

    bench.bench_function("iter_rev", move |bh| {
        bh.iter(|| {
            for value in a.iter().rev() {
                criterion::black_box(value);
            }
        })
    });
}

fn copy_from(bench: &mut criterion::Criterion) {
    let a = DMatrix::<f64>::new_random(1000, 1000);
    let mut b = DMatrix::<f64>::new_random(1000, 1000);

    bench.bench_function("copy_from", move |bh| {
        bh.iter(|| {
            b.copy_from(&a);
        })
    });
}

fn axpy(bench: &mut criterion::Criterion) {
    let x = DVector::<f64>::from_element(100000, 2.0);
    let mut y = DVector::<f64>::from_element(100000, 3.0);
    let a = 42.0;

    bench.bench_function("axpy", move |bh| {
        bh.iter(|| {
            y.axpy(a, &x, 1.0);
        })
    });
}

fn tr_mul_to(bench: &mut criterion::Criterion) {
    let a = DMatrix::<f64>::new_random(1000, 1000);
    let b = DVector::<f64>::new_random(1000);
    let mut c = DVector::new_uninitialized_generic(Dynamic::new(1000), Const::<1>);

    bench.bench_function("tr_mul_to", move |bh| bh.iter(|| a.tr_mul_to(&b, &mut c)));
}

fn mat_mul_mat(bench: &mut criterion::Criterion) {
    let a = DMatrix::<f64>::new_random(100, 100);
    let b = DMatrix::<f64>::new_random(100, 100);
    let mut ab = DMatrix::new_uninitialized_generic(Dynamic::new(100), Dynamic::new(100));

    bench.bench_function("mat_mul_mat", move |bh| {
        bh.iter(|| {
            std::hint::black_box(a.mul_to(&b, &mut ab));
        })
    });
}

fn mat100_from_fn(bench: &mut criterion::Criterion) {
    bench.bench_function("mat100_from_fn", move |bh| {
        bh.iter(|| DMatrix::from_fn(100, 100, |a, b| a + b))
    });
}

fn mat500_from_fn(bench: &mut criterion::Criterion) {
    bench.bench_function("mat500_from_fn", move |bh| {
        bh.iter(|| DMatrix::from_fn(500, 500, |a, b| a + b))
    });
}

criterion_group!(
    matrix,
    mat2_mul_m,
    mat3_mul_m,
    mat4_mul_m,
    mat2_tr_mul_m,
    mat3_tr_mul_m,
    mat4_tr_mul_m,
    mat2_add_m,
    mat3_add_m,
    mat4_add_m,
    mat2_sub_m,
    mat3_sub_m,
    mat4_sub_m,
    mat2_mul_v,
    mat3_mul_v,
    mat4_mul_v,
    mat2_tr_mul_v,
    mat3_tr_mul_v,
    mat4_tr_mul_v,
    mat2_mul_s,
    mat3_mul_s,
    mat4_mul_s,
    mat2_div_s,
    mat3_div_s,
    mat4_div_s,
    mat2_inv,
    mat3_inv,
    mat4_inv,
    mat2_transpose,
    mat3_transpose,
    mat4_transpose,
    mat_div_scalar,
    mat100_add_mat100,
    mat4_mul_mat4,
    mat5_mul_mat5,
    mat6_mul_mat6,
    mat7_mul_mat7,
    mat8_mul_mat8,
    mat9_mul_mat9,
    mat10_mul_mat10,
    mat10_mul_mat10_static,
    mat100_mul_mat100,
    mat500_mul_mat500,
    iter,
    iter_rev,
    copy_from,
    axpy,
    tr_mul_to,
    mat_mul_mat,
    mat100_from_fn,
    mat500_from_fn,
);
