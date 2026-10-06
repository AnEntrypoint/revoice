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

        let t_stft = std::time::Instant::now();
        let frames = self
            .stft
            .forward(&normalized)
            .map_err(candle_core::Error::wrap)?;
        if std::env::var("RESOUND_PROFILE").is_ok() {
            eprintln!("stft_forward_ms={}", t_stft.elapsed().as_millis());
        }
        let n_frames = frames.len();
        let n_freqs = frames[0].len();

        let mut mag = vec![0.0f32; n_freqs * n_frames];
        let mut cos = vec![0.0f32; n_freqs * n_frames];
        let mut sin = vec![0.0f32; n_freqs * n_frames];
        for (t, frame) in frames.iter().enumerate() {
            let base = t * n_freqs;
            for (f, c) in frame.iter().enumerate() {
                let m = c.norm();
                let idx = base + f;
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

        let mag_t = Tensor::from_vec(mag, (1, 1, n_frames, n_freqs), device)?;
        let cos_t = Tensor::from_vec(cos, (1, 1, n_frames, n_freqs), device)?;
        let sin_t = Tensor::from_vec(sin, (1, 1, n_frames, n_freqs), device)?;
        let input = Tensor::cat(&[mag_t, cos_t, sin_t], 1)?
            .transpose(2, 3)?
            .contiguous()?
            .to_dtype(DType::F32)?;
        let mag_t = input.narrow(1, 0, 1)?.contiguous()?;
        let cos_t = input.narrow(1, 1, 1)?.contiguous()?;
        let sin_t = input.narrow(1, 2, 1)?.contiguous()?;

        let t_unet = std::time::Instant::now();
        let output = self.net.forward(&input)?;
        if std::env::var("RESOUND_PROFILE").is_ok() {
            eprintln!("unet_forward_ms={}", t_unet.elapsed().as_millis());
        }
        let mag_mask = candle_nn::ops::sigmoid(&output.narrow(1, 0, 1)?)?;
        let real = output.narrow(1, 1, 1)?.tanh()?;
        let imag = output.narrow(1, 2, 1)?.tanh()?;
        let res_mag = ((real.sqr()? + imag.sqr()?)? + 1e-7)?.sqrt()?;
        let cos_res = (&real / &res_mag)?;
        let sin_res = (&imag / &res_mag)?;

        let sep_mag = (mag_t.clone() * mag_mask)?.relu()?;
        let sep_cos = ((cos_t.clone() * &cos_res)? - (sin_t.clone() * &sin_res)?)?;
        let sep_sin = ((sin_t * &cos_res)? + (cos_t * &sin_res)?)?;

        let to_frames = |t: Tensor| -> Result<Vec<f32>> {
            t.transpose(2, 3)?.contiguous()?.flatten_all()?.to_vec1::<f32>()
        };
        let out_re = to_frames((sep_mag.clone() * sep_cos)?)?;
        let out_im = to_frames((sep_mag * sep_sin)?)?;

        let mut out_frames = Vec::with_capacity(n_frames);
        for t in 0..n_frames {
            let base = t * n_freqs;
            let mut frame = Vec::with_capacity(n_freqs);
            for f in 0..n_freqs {
                let idx = base + f;
                let is_edge_bin = f == 0 || f == n_freqs - 1;
                let im = if is_edge_bin { 0.0 } else { out_im[idx] };
                frame.push(Complex32::new(out_re[idx], im));
            }
            out_frames.push(frame);
        }

        let t_istft = std::time::Instant::now();
        let restored = self
            .stft
            .inverse(&out_frames, normalized.len())
            .map_err(candle_core::Error::wrap)?;
        if std::env::var("RESOUND_PROFILE").is_ok() {
            eprintln!("stft_inverse_ms={}", t_istft.elapsed().as_millis());
        }
        Ok(restored.iter().map(|&s| s * abs_max).collect())
    }
}
