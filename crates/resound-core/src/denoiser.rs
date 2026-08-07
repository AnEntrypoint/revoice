use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use realfft::num_complex::Complex32;

use crate::stft::Stft;
use crate::unet::UNet;

pub struct Denoiser {
    net: UNet,
    stft: Stft,
}

impl Denoiser {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let net = UNet::new(3, 3, 16, 4, 2, vb.pp("net"))?;
        let hop_size = 420;
        let stft = Stft::new(hop_size * 4, hop_size, hop_size * 4);
        Ok(Self { net, stft })
    }

    pub fn forward(&self, wav: &[f32], device: &Device) -> Result<Vec<f32>> {
        let abs_max = wav.iter().fold(1e-8f32, |a, &b| a.max(b.abs()));
        let normalized: Vec<f32> = wav.iter().map(|&s| s / abs_max).collect();

        let frames = self
            .stft
            .forward(&normalized)
            .map_err(candle_core::Error::wrap)?;
        let n_frames = frames.len();
        let n_freqs = frames[0].len();

        let mut mag = vec![0.0f32; n_freqs * n_frames];
        let mut cos = vec![0.0f32; n_freqs * n_frames];
        let mut sin = vec![0.0f32; n_freqs * n_frames];
        for (t, frame) in frames.iter().enumerate() {
            for (f, c) in frame.iter().enumerate() {
                let m = c.norm();
                let idx = f * n_frames + t;
                mag[idx] = m;
                if m > 1e-8 {
                    cos[idx] = c.re / m;
                    sin[idx] = c.im / m;
                } else {
                    cos[idx] = 1.0;
                    sin[idx] = 0.0;
                }
            }
        }

        let mag_t = Tensor::from_vec(mag, (1, 1, n_freqs, n_frames), device)?;
        let cos_t = Tensor::from_vec(cos, (1, 1, n_freqs, n_frames), device)?;
        let sin_t = Tensor::from_vec(sin, (1, 1, n_freqs, n_frames), device)?;
        let input = Tensor::cat(&[mag_t.clone(), cos_t, sin_t], 1)?.to_dtype(DType::F32)?;

        let output = self.net.forward(&input)?;
        let mag_mask = output.narrow(1, 0, 1)?.relu()?;
        let real_res = output.narrow(1, 1, 1)?;
        let imag_res = output.narrow(1, 2, 1)?;

        let out_mag = (mag_t.clone() * mag_mask)?;
        let denom = ((real_res.sqr()? + imag_res.sqr()?)? + 1e-8)?.sqrt()?;
        let out_cos = (real_res / &denom)?;
        let out_sin = (imag_res / &denom)?;

        let out_re = (out_mag.clone() * out_cos)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let out_im = (out_mag * out_sin)?.flatten_all()?.to_vec1::<f32>()?;

        let mut out_frames = Vec::with_capacity(n_frames);
        for t in 0..n_frames {
            let mut frame = Vec::with_capacity(n_freqs);
            for f in 0..n_freqs {
                let idx = f * n_frames + t;
                let is_edge_bin = f == 0 || f == n_freqs - 1;
                let im = if is_edge_bin { 0.0 } else { out_im[idx] };
                frame.push(Complex32::new(out_re[idx], im));
            }
            out_frames.push(frame);
        }

        let restored = self
            .stft
            .inverse(&out_frames, normalized.len())
            .map_err(candle_core::Error::wrap)?;
        Ok(restored.iter().map(|&s| s * abs_max).collect())
    }
}
