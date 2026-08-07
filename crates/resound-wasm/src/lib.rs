use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use wasm_bindgen::prelude::*;
use resound_core::denoiser::Denoiser;
use resound_core::enhancer::Enhancer;
use resound_core::melspec::MelSpectrogram;
use resound_core::resample::Resampler;

#[wasm_bindgen]
pub fn resound_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[wasm_bindgen]
pub fn resample_to_44100(input: &[f32], input_rate: u32) -> Vec<f32> {
    let resampler = Resampler::new(input_rate as usize, 44100);
    resampler.process(input)
}

#[wasm_bindgen]
pub fn mel_spectrogram_frames(wav: &[f32]) -> Result<usize, JsError> {
    let mel = MelSpectrogram::new(44100.0, 2048, 420, 2048, 128, 0.0, 22050.0, 0.97);
    let out = mel.forward(wav).map_err(|e| JsError::new(&e.to_string()))?;
    Ok(out[0].len())
}

#[wasm_bindgen]
pub fn denoise(wav: &[f32], weight_bytes: &[u8]) -> Result<Vec<f32>, JsError> {
    let device = Device::Cpu;
    let vb = VarBuilder::from_buffered_safetensors(weight_bytes.to_vec(), DType::F32, &device)
        .map_err(|e| JsError::new(&e.to_string()))?;
    let denoiser = Denoiser::new(vb.pp("denoiser")).map_err(|e| JsError::new(&e.to_string()))?;
    denoiser
        .forward(wav, &device)
        .map_err(|e| JsError::new(&e.to_string()))
}

#[wasm_bindgen]
pub fn enhance(wav: &[f32], weight_bytes: &[u8], nfe: usize, tau: f64) -> Result<Vec<f32>, JsError> {
    let device = Device::Cpu;
    let vb = VarBuilder::from_buffered_safetensors(weight_bytes.to_vec(), DType::F32, &device)
        .map_err(|e| JsError::new(&e.to_string()))?;
    let enhancer = Enhancer::new(vb, &device).map_err(|e| JsError::new(&e.to_string()))?;
    enhancer
        .forward(wav, nfe, tau, &device)
        .map_err(|e| JsError::new(&e.to_string()))
}
