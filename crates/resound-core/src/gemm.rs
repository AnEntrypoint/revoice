use std::sync::OnceLock;

use candle_core::DType;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GemmPrecision {
    F32,
    Tf32,
    F16,
}

static PRECISION: OnceLock<GemmPrecision> = OnceLock::new();

pub fn gemm_precision() -> GemmPrecision {
    *PRECISION.get_or_init(|| match std::env::var("RESOUND_GEMM").as_deref() {
        Ok("tf32") => {
            candle_core::cuda::set_gemm_reduced_precision_f32(true);
            GemmPrecision::Tf32
        }
        Ok("f16") => GemmPrecision::F16,
        _ => GemmPrecision::F32,
    })
}

pub fn gemm_dtype() -> DType {
    match gemm_precision() {
        GemmPrecision::F16 => DType::F16,
        _ => DType::F32,
    }
}
