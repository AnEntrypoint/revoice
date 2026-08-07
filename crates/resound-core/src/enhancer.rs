use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::cfm::{CfmSolver, WaveNet};
use crate::common::Normalizer;
use crate::irmae::{IrmaeDecoder, IrmaeEncoder};
use crate::melspec::MelSpectrogram;
use crate::univnet::UnivNet;

pub struct Enhancer {
    mel_fn: MelSpectrogram,
    normalizer: Normalizer,
    ae_encoder: IrmaeEncoder,
    ae_decoder: IrmaeDecoder,
    cfm_net: WaveNet,
    vocoder: UnivNet,
    n_mels: usize,
    z_scale: f64,
}

impl Enhancer {
    pub fn new(vb: VarBuilder, device: &Device) -> Result<Self> {
        let n_mels = 128;
        let hidden_dim = 1024;
        let latent_dim = 64;
        let mel_fn = MelSpectrogram::new(44100.0, 2048, 420, 2048, n_mels, 0.0, 22050.0, 0.97);
        let normalizer = Normalizer::new(vb.pp("normalizer"))?;

        let ae_vb = vb.pp("lcfm.ae");
        let ae_encoder = IrmaeEncoder::new(n_mels, hidden_dim, latent_dim, 4, 4, ae_vb.pp("encoder"))?;
        let ae_decoder = IrmaeDecoder::new(latent_dim, hidden_dim, n_mels + 32, 4, ae_vb.pp("decoder"))?;

        let cfm_net = WaveNet::new(latent_dim, latent_dim, n_mels, 128, 30, 5, 512, vb.pp("lcfm.cfm.net"), device)?;

        let vocoder = UnivNet::new(
            n_mels + 32,
            128,
            96,
            &[7, 5, 4, 3],
            &[1, 3, 9, 27],
            3,
            vb.pp("vocoder"),
            device,
        )?;

        Ok(Self {
            mel_fn,
            normalizer,
            ae_encoder,
            ae_decoder,
            cfm_net,
            vocoder,
            n_mels,
            z_scale: 5.0,
        })
    }

    pub fn forward(&self, wav: &[f32], nfe: usize, tau: f64, device: &Device) -> Result<Vec<f32>> {
        let abs_max = wav.iter().fold(1e-8f32, |a, &b| a.max(b.abs()));
        let normalized: Vec<f32> = wav.iter().map(|&s| s / abs_max).collect();

        let mel = self
            .mel_fn
            .forward(&normalized)
            .map_err(candle_core::Error::wrap)?;
        let n_frames = mel[0].len();
        let mut mel_flat = vec![0.0f32; self.n_mels * n_frames];
        for (m, row) in mel.iter().enumerate() {
            for (t, &v) in row.iter().enumerate() {
                mel_flat[m * n_frames + t] = crate::melspec::normalize_db(
                    crate::melspec::amp_to_db(v, 1e-4),
                    -80.0,
                );
            }
        }
        let mel_t = Tensor::from_vec(mel_flat, (1, self.n_mels, n_frames), device)?;
        let mel_t = self.normalizer.forward(&mel_t)?;

        let latent = self.ae_encoder.forward(&mel_t)?;
        let scaled_latent = (latent * self.z_scale)?;

        let noise = Tensor::randn(0.0f32, 1.0f32, scaled_latent.dims(), device)?;
        let psi0 = ((noise * tau)? + (scaled_latent * (1.0 - tau))?)?;

        let solver = CfmSolver::new(nfe);
        let z = solver.sample(&self.cfm_net, &mel_t, &psi0, device)?;
        let z = (z / self.z_scale)?;

        let decoded = self.ae_decoder.forward(&z)?;

        let vocoder_noise = Tensor::randn(0.0f32, 1.0f32, (1, 128, n_frames), device)?;
        let out_wav = self.vocoder.forward(&vocoder_noise, &decoded)?;
        let out_wav = out_wav.flatten_all()?.to_vec1::<f32>()?;

        Ok(out_wav.iter().map(|&s| s * abs_max).collect())
    }
}
