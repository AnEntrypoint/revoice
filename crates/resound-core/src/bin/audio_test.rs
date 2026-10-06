use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use resound_core::denoiser::Denoiser;
use resound_core::enhancer::Enhancer;

fn rms(samples: &[f32]) -> f32 {
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len().max(1) as f32).sqrt()
}

fn main() -> candle_core::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let wav_path = &args[1];

    let mut reader = hound::WavReader::open(wav_path).expect("failed to open wav");
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.unwrap_or(0) as f32 / 32768.0)
            .collect(),
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
    };

    println!("input_sample_rate={}", spec.sample_rate);
    println!("input_num_samples={}", samples.len());
    println!("input_rms={:.6}", rms(&samples));

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("device={:?}", device);
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &["weights/enhancer_stage2.safetensors"],
            DType::F32,
            &device,
        )?
    };

    let resampled = if spec.sample_rate != 44100 {
        let resampler = resound_core::resample::Resampler::new(spec.sample_rate as usize, 44100);
        resampler.process(&samples)
    } else {
        samples
    };
    println!("resampled_num_samples={}", resampled.len());
    println!("resampled_rms={:.6}", rms(&resampled));

    let denoiser = Denoiser::new(vb.pp("denoiser"))?;
    let t0 = std::time::Instant::now();
    let denoised = denoiser.forward(&resampled, &device).map_err(candle_core::Error::wrap)?;
    println!("denoise_elapsed_ms={}", t0.elapsed().as_millis());
    println!("denoised_num_samples={}", denoised.len());
    println!("denoised_rms={:.6}", rms(&denoised));

    let mut writer = hound::WavWriter::create(
        "testsound/denoised_output.wav",
        hound::WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .expect("failed to create output wav");
    for &s in &denoised {
        let clamped = s.clamp(-1.0, 1.0);
        writer
            .write_sample((clamped * 32767.0) as i16)
            .expect("failed to write sample");
    }
    writer.finalize().expect("failed to finalize wav");
    println!("denoised_written=testsound/denoised_output.wav");

    let enhancer = Enhancer::new(vb, &device)?;
    let t1 = std::time::Instant::now();
    let enhanced = enhancer.forward(&resampled, 64, 0.5, &device)?;
    println!("enhance_elapsed_ms={}", t1.elapsed().as_millis());
    println!("enhanced_num_samples={}", enhanced.len());
    println!("enhanced_rms={:.6}", rms(&enhanced));

    let mut writer2 = hound::WavWriter::create(
        "testsound/enhanced_output.wav",
        hound::WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .expect("failed to create output wav");
    for &s in &enhanced {
        let clamped = s.clamp(-1.0, 1.0);
        writer2
            .write_sample((clamped * 32767.0) as i16)
            .expect("failed to write sample");
    }
    writer2.finalize().expect("failed to finalize wav");
    println!("enhanced_written=testsound/enhanced_output.wav");

    Ok(())
}
