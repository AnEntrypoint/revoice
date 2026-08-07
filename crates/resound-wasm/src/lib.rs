use wasm_bindgen::prelude::*;
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
pub fn mel_spectrogram_frames(wav: &[f32]) -> usize {
    let mel = MelSpectrogram::new(44100.0, 2048, 420, 2048, 128, 0.0, 22050.0, 0.97);
    let out = mel.forward(wav);
    out[0].len()
}
