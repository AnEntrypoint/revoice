use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

pub struct Normalizer {
    running_mean: Tensor,
    running_var: Tensor,
}

impl Normalizer {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let running_mean = vb.get_with_hints((), "running_mean_unsafe", candle_nn::init::ZERO)?;
        let running_var = vb.get_with_hints((), "running_var_unsafe", candle_nn::init::ONE)?;
        Ok(Self {
            running_mean,
            running_var,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = self.running_mean.to_scalar::<f32>()?;
        let var = self.running_var.to_scalar::<f32>()?;
        let std = var.sqrt();
        (x - mean as f64)? / std as f64
    }

    pub fn inverse(&self, x: &Tensor) -> Result<Tensor> {
        let mean = self.running_mean.to_scalar::<f32>()?;
        let var = self.running_var.to_scalar::<f32>()?;
        let std = var.sqrt();
        (x * std as f64)? + mean as f64
    }
}
