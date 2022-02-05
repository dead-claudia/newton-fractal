use num_complex::Complex32;
use wasm_bindgen::prelude::*;

use std::arch::wasm32::*;
use std::mem::transmute;
use std::ptr::addr_of;

use crate::simd_constants::SimdHelper;

use super::logger::*;

#[wasm_bindgen]
#[derive(Clone)]
pub struct Polynomial {
    roots: Vec<Complex32>,
}

#[wasm_bindgen]
impl Polynomial {
    #[wasm_bindgen(constructor)]
    pub fn new(roots: &JsValue) -> Option<Polynomial> {
        let roots: Vec<(f32, f32)> = match roots.into_serde() {
            Ok(v) => v,
            Err(_) => {
                log!("Error getting roots from: {:?}", roots);
                return None;
            }
        };

        Some(Polynomial {
            roots: roots
                .iter()
                .map(|(re, im)| Complex32 { re: *re, im: *im })
                .collect(),
        })
    }

    #[wasm_bindgen]
    pub fn add_root(&mut self, re: f32, im: f32) {
        self.roots.push(Complex32 { re, im });
    }

    #[wasm_bindgen]
    pub fn get_closest_root_id(&self, re: f32, im: f32) -> JsValue {
        let mut min_d = f32::MAX;
        let mut idx = u32::MAX;

        let p = Complex32::new(re, im);
        for (i, root) in self.roots.iter().enumerate() {
            let d = p - root;
            let d = (d.re * d.re + d.im * d.im).sqrt();
            if d < min_d {
                min_d = d;
                idx = i as u32;
            }
        }

        if idx as usize > self.roots.len() {
            return JsValue::from_serde(&(-1.0, u32::MAX)).unwrap();
        }
        JsValue::from_serde(&(idx, min_d)).unwrap()
    }

    #[wasm_bindgen]
    pub fn remove_root_by_id(&mut self, id: usize) {
        if id > self.roots.len() {
            return;
        }
        self.roots.remove(id);
    }

    #[wasm_bindgen]
    pub fn set_root_by_id(&mut self, id: usize, re: f32, im: f32) {
        if id > self.roots.len() {
            return;
        }
        self.roots[id] = Complex32::new(re, im);
    }
}

impl Polynomial {
    pub fn get_roots(&self) -> &[Complex32] {
        &self.roots
    }

    pub fn calculate(&self, z: Complex32) -> Option<Complex32> {
        let mut prod = match self.roots.get(0) {
            Some(v) => z - v,
            None => return None,
        };
        for root in self.roots.iter().skip(1) {
            prod *= z - root;
        }
        Some(prod)
    }

    pub fn derivative(&self, z: Complex32) -> Option<Complex32> {
        let (mut prod, mut sum) = match self.roots.get(0) {
            Some(v) => (z - v, 1.0 / (z - v)),
            None => return None,
        };
        for root in self.roots.iter().skip(1) {
            prod *= z - root;
            sum += 1.0 / (z - root);
        }
        Some(prod * sum)
    }

    pub fn newton_method_approx(&self, z: Complex32) -> Complex32 {
        let mut sum = Complex32::new(0.0, 0.0);
        for root in self.roots.iter() {
            sum += 1.0 / (z - root);
            if sum.is_nan() {
                return root.clone();
            }
        }
        z - 1.0 / sum
    }

    // #[inline]
    #[target_feature(enable = "simd128")]
    pub fn simd_newton_method_approx_for_two_numbers(&self, _two_z: v128) -> v128 {
        // In scalar implementation we approximate only one number at a timer.
        // When using SIMDs, we approximate two numbers at the same time.
        // In following comments "z" means either "z1" or "z2".

        let mut _sum1 = f32x4_splat(0.0);
        let mut _sum2 = f32x4_splat(0.0);
        let (z1, z2): (u64, u64) = unsafe { transmute(_two_z) };
        let _z1 = u64x2_splat(z1);
        let _z2 = u64x2_splat(z2);
        let mut _ans = SimdHelper::F64_NANS;
        let _ans_addr = addr_of!(_ans) as *mut u64;

        // In scalar implementation we process only one root at a time.
        // When using SIMDs, we process two roots at the same time.
        // We have f32x4 [A, B, C, D], in which (A, B): re and im parts
        // of first complex root and (C, D): re and im parts of second
        // complex root.
        // To get single complex value we need to sum (A + C, B + D)

        let roots_chunks_iter = self.roots.chunks_exact(2);
        let rem = roots_chunks_iter.remainder();
        for roots_chunk in roots_chunks_iter {
            unsafe {
                // General formula: sum += 1.0 / (z - root)
                // 1. Subtraction (z - root)

                let _two_roots = *(roots_chunk.as_ptr() as *const v128);
                let _diff1 = f32x4_sub(_z1, _two_roots);
                let _diff2 = f32x4_sub(_z2, _two_roots);

                // 1*. Check if difference == 0 <=> z1 or z2 == one of roots

                let _diff_eq1 = f64x2_eq(_diff1, SimdHelper::F64_ZEROES);
                let _diff_eq2 = f64x2_eq(_diff2, SimdHelper::F64_ZEROES);

                if v128_any_true(_diff_eq1) {
                    *_ans_addr = z1;
                }
                if v128_any_true(_diff_eq2) {
                    *(_ans_addr.offset(1)) = z2;
                }

                // 2. Inversion (1.0 / _diff <=> 1.0 / (z - root))
                let _inversion1 = SimdHelper::complex_number_inversion(_diff1);
                let _inversion2 = SimdHelper::complex_number_inversion(_diff2);

                // 3. Addition (sum += 1.0 / (z - root))
                _sum1 = f32x4_add(_sum1, _inversion1);
                _sum2 = f32x4_add(_sum2, _inversion2);
            }
        }

        // Move second complex values to two first lanes
        let _sum_shift1 = i64x2_shuffle::<1, 0>(_sum1, _sum1);
        let _sum_shift2 = i64x2_shuffle::<1, 0>(_sum2, _sum2);

        // Process odd root
        if let Some(rem) = rem.get(0) {
            unsafe {
                // This process is same as the processing of two roots, except second
                // complex value in vector is equal to 0 <=> vector: [A, B, 0, 0];

                let _subtrahend = v128_load64_splat(addr_of!(*rem) as *const u64);
                let _diff1 = f32x4_sub(_z1, _subtrahend);
                let _diff2 = f32x4_sub(_z2, _subtrahend);

                let _diff_eq1 = f64x2_eq(_diff1, SimdHelper::F64_ZEROES);
                let _diff_eq2 = f64x2_eq(_diff2, SimdHelper::F64_ZEROES);

                if v128_any_true(_diff_eq1) {
                    *_ans_addr = z1;
                }
                if v128_any_true(_diff_eq2) {
                    *(_ans_addr.offset(1)) = z2;
                }

                let _inversion1 = SimdHelper::complex_number_inversion(_diff1);
                let _inversion2 = SimdHelper::complex_number_inversion(_diff2);

                _sum1 = f32x4_add(_sum1, _inversion1);
                _sum2 = f32x4_add(_sum2, _inversion2);
            }
        }

        // Sum first and second complex values
        _sum1 = f32x4_add(_sum1, _sum_shift1);
        _sum2 = f32x4_add(_sum2, _sum_shift2);

        // Return value: z - 1.0 / sum
        let _inversion1 = SimdHelper::complex_number_inversion(_sum1);
        let _inversion2 = SimdHelper::complex_number_inversion(_sum2);
        let _sub1 = f32x4_sub(_z1, _inversion1);
        let _sub2 = f32x4_sub(_z2, _inversion2);
        let _sub = u32x4_shuffle::<0, 1, 4, 5>(_sub1, _sub2);

        let _change_mask = u64x2_eq(_ans, SimdHelper::F64_NANS);

        v128_bitselect(_sub, _ans, _change_mask)
    }
}
