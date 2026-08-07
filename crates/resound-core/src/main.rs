use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use resound_core::melspec::MelSpectrogram;
use resound_core::resample::Resampler;
use resound_core::unet::UNet;
use resound_core::irmae::{IrmaeDecoder, IrmaeEncoder};
use resound_core::aliasfree::AmpBlock;
use resound_core::univnet::UnivNet;
use resound_core::cfm::{CfmSolver, WaveNet};
use resound_core::chunking::{merge_chunks, split_chunks, ChunkConfig};

fn check_unet() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let unet = UNet::new(3, 3, 16, 4, 2, vb)?;
    let input = Tensor::randn(0.0f32, 1.0f32, (1, 3, 65, 130), &device)?;
    let output = unet.forward(&input)?;
    println!("unet_output_shape={:?}", output.dims());
    Ok(())
}

fn check_irmae() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let encoder = IrmaeEncoder::new(128, 1024, 64, 4, 4, vb.pp("enc"))?;
    let decoder = IrmaeDecoder::new(64, 1024, 160, 4, vb.pp("dec"))?;
    let input = Tensor::randn(0.0f32, 1.0f32, (1, 128, 50), &device)?;
    let latent = encoder.forward(&input)?;
    let output = decoder.forward(&latent)?;
    println!("irmae_latent_shape={:?}", latent.dims());
    println!("irmae_output_shape={:?}", output.dims());
    Ok(())
}

fn check_ampblock() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let block = AmpBlock::new(96, &[1, 3, 5], vb, &device)?;
    let input = Tensor::randn(0.0f32, 1.0f32, (1, 96, 200), &device)?;
    let output = block.forward(&input)?;
    println!("ampblock_output_shape={:?}", output.dims());
    Ok(())
}

fn check_univnet() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let cond_frames = 20usize;
    let hop = 7 * 5 * 4 * 3;
    let net = UnivNet::new(160, 128, 32, &[7, 5, 4, 3], &[1, 3, 9, 27], 3, vb, &device)?;
    let noise = Tensor::randn(0.0f32, 1.0f32, (1, 128, cond_frames), &device)?;
    let cond = Tensor::randn(0.0f32, 1.0f32, (1, 160, cond_frames), &device)?;
    let out = net.forward(&noise, &cond)?;
    println!("univnet_output_shape={:?} expected_samples~={}", out.dims(), cond_frames * hop);
    Ok(())
}

fn check_cfm() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let wn = WaveNet::new(64, 64, 64, 128, 6, 5, 128, vb, &device)?;
    let solver = CfmSolver::new(8);
    let cond = Tensor::randn(0.0f32, 1.0f32, (1, 64, 30), &device)?;
    let psi0 = Tensor::randn(0.0f32, 1.0f32, (1, 64, 30), &device)?;
    let z = solver.sample(&wn, &cond, &psi0, &device)?;
    println!("cfm_output_shape={:?}", z.dims());
    Ok(())
}

fn check_chunking() {
    let sr = 44100usize;
    let dur_s = 5.0f64;
    let n = (sr as f64 * dur_s) as usize;
    let freq = 220.0f32;
    let wav: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
        .collect();

    let cfg = ChunkConfig {
        sample_rate: sr,
        chunk_seconds: 1.0,
        overlap_seconds: 0.2,
    };
    let chunks = split_chunks(&wav, &cfg);
    let chunk_data: Vec<Vec<f32>> = chunks.into_iter().map(|(_, c)| c).collect();
    let merged = merge_chunks(&chunk_data, cfg.overlap_len(), wav.len());

    let mse: f32 = wav
        .iter()
        .zip(merged.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / wav.len() as f32;
    println!("chunking_num_chunks={} merged_len={} mse={}", chunk_data.len(), merged.len(), mse);
}

fn check_real_weights() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &["weights/enhancer_stage2.safetensors"],
            DType::F32,
            &device,
        )?
    };

    let ae_vb = vb.pp("lcfm.ae");
    let encoder = IrmaeEncoder::new(128, 1024, 64, 4, 4, ae_vb.pp("encoder"))?;
    let decoder = IrmaeDecoder::new(64, 1024, 160, 4, ae_vb.pp("decoder"))?;
    let mel_input = Tensor::randn(0.0f32, 1.0f32, (1, 128, 40), &device)?;
    let latent = encoder.forward(&mel_input)?;
    let ae_out = decoder.forward(&latent)?;
    println!("real_irmae_latent_shape={:?}", latent.dims());
    println!("real_irmae_output_shape={:?}", ae_out.dims());

    let wn = WaveNet::new(64, 64, 128, 128, 30, 5, 512, vb.pp("lcfm.cfm.net"), &device)?;
    let solver = CfmSolver::new(4);
    let cond = Tensor::randn(0.0f32, 1.0f32, (1, 128, 40), &device)?;
    let psi0 = Tensor::randn(0.0f32, 1.0f32, (1, 64, 40), &device)?;
    let z = solver.sample(&wn, &cond, &psi0, &device)?;
    println!("real_cfm_output_shape={:?}", z.dims());

    let vocoder = UnivNet::new(160, 128, 96, &[7, 5, 4, 3], &[1, 3, 9, 27], 3, vb.pp("vocoder"), &device)?;
    let noise = Tensor::randn(0.0f32, 1.0f32, (1, 128, 20), &device)?;
    let vcond = Tensor::randn(0.0f32, 1.0f32, (1, 160, 20), &device)?;
    let wav = vocoder.forward(&noise, &vcond)?;
    println!("real_univnet_output_shape={:?}", wav.dims());

    let denoiser_vb = vb.pp("denoiser.net");
    let unet = UNet::new(3, 3, 16, 4, 2, denoiser_vb)?;
    let spec = Tensor::randn(0.0f32, 1.0f32, (1, 3, 65, 130), &device)?;
    let denoised = unet.forward(&spec)?;
    println!("real_unet_output_shape={:?}", denoised.dims());

    Ok(())
}

fn check_edge_cases() {
    let empty: Vec<f32> = vec![];
    let resampler = Resampler::new(16000, 44100);
    let out = resampler.process(&empty);
    println!("edge_empty_resample_len={}", out.len());

    let single = vec![0.5f32];
    let out2 = resampler.process(&single);
    println!("edge_single_sample_resample_len={}", out2.len());

    let mel = MelSpectrogram::new(44100.0, 2048, 420, 2048, 128, 0.0, 22050.0, 0.97);
    let tiny_wav = vec![0.1f32; 10];
    let mel_out = mel.forward(&tiny_wav);
    println!("edge_tiny_mel_frames={}", mel_out[0].len());

    let a = resampler.process(&single);
    let b = resampler.process(&single);
    println!("edge_replay_deterministic={}", a == b);

    let cfg = ChunkConfig {
        sample_rate: 44100,
        chunk_seconds: 1.0,
        overlap_seconds: 0.2,
    };
    let short_wav = vec![0.0f32; 100];
    let chunks = split_chunks(&short_wav, &cfg);
    println!("edge_short_audio_num_chunks={}", chunks.len());
    let merged = merge_chunks(
        &chunks.iter().map(|(_, c)| c.clone()).collect::<Vec<_>>(),
        cfg.overlap_len(),
        short_wav.len(),
    );
    println!("edge_short_audio_merged_len={}", merged.len());
}

fn check_adversarial() {
    let resampler = Resampler::new(16000, 44100);

    // reentry: call the same pure op twice from the same thread, back to back
    let wav = vec![0.3f32; 5000];
    let r1 = resampler.process(&wav);
    let r2 = resampler.process(&wav);
    println!("adv_reentry_same_result={}", r1 == r2);

    // resource exhaustion: many repeated calls must not grow unbounded (no leak in a pure fn)
    for _ in 0..200 {
        let _ = resampler.process(&wav);
    }
    println!("adv_repeated_calls_completed=true");

    // boundary: exact multiple of chunk length
    let cfg = ChunkConfig {
        sample_rate: 44100,
        chunk_seconds: 1.0,
        overlap_seconds: 0.2,
    };
    let exact_wav = vec![0.0f32; cfg.chunk_len()];
    let chunks = split_chunks(&exact_wav, &cfg);
    println!("adv_exact_chunk_boundary_num_chunks={}", chunks.len());

    // boundary: one sample over the chunk length
    let over_wav = vec![0.0f32; cfg.chunk_len() + 1];
    let chunks2 = split_chunks(&over_wav, &cfg);
    println!("adv_over_chunk_boundary_num_chunks={}", chunks2.len());

    // degenerate: NaN/Inf input must not panic (propagates NaN, doesn't crash)
    let nan_wav = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, 1.0];
    let nan_out = resampler.process(&nan_wav);
    println!("adv_nan_input_no_panic=true len={}", nan_out.len());

    // adjacent-row interaction: mel spectrogram consuming resampler output at a boundary size
    let mel = MelSpectrogram::new(44100.0, 2048, 420, 2048, 128, 0.0, 22050.0, 0.97);
    let boundary_resampled = resampler.process(&vec![0.2f32; 1]);
    let mel_boundary = mel.forward(&boundary_resampled);
    println!("adv_pipeline_boundary_mel_frames={}", mel_boundary[0].len());
}

fn main() {
    check_unet().expect("unet check failed");
    check_irmae().expect("irmae check failed");
    check_ampblock().expect("ampblock check failed");
    check_univnet().expect("univnet check failed");
    check_cfm().expect("cfm check failed");
    check_chunking();
    check_real_weights().expect("real weight loading failed");
    check_edge_cases();
    check_adversarial();
    let sr = 16000usize;
    let freq = 440.0f32;
    let wav: Vec<f32> = (0..sr)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
        .collect();

    let mel = MelSpectrogram::new(44100.0, 2048, 420, 2048, 128, 0.0, 22050.0, 0.97);
    let resampler = Resampler::new(sr, 44100);
    let resampled = resampler.process(&wav);
    let mel_out = mel.forward(&resampled);

    println!("resampled_len={}", resampled.len());
    println!("mel_bins={} mel_frames={}", mel_out.len(), mel_out[0].len());
    let peak_bin = mel_out
        .iter()
        .enumerate()
        .max_by(|a, b| {
            let sa: f32 = a.1.iter().sum();
            let sb: f32 = b.1.iter().sum();
            sa.partial_cmp(&sb).unwrap()
        })
        .map(|(i, _)| i)
        .unwrap();
    println!("peak_mel_bin={}", peak_bin);
}
