use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use resound_core::chunking::{merge_chunks, split_chunks, ChunkConfig};
use resound_core::denoiser::Denoiser;
use resound_core::enhancer::Enhancer;
use resound_core::resample::Resampler;

const TARGET_RATE: usize = 44100;

struct Args {
    command: String,
    inputs: Vec<PathBuf>,
    out_dir: Option<PathBuf>,
    weights: PathBuf,
    nfe: usize,
    tau: f64,
    chunk_seconds: f64,
    overlap_seconds: f64,
    device: String,
}

fn usage() -> ! {
    eprintln!(
        "usage: resound <denoise|enhance> [options] <input.wav>...\n\
         \n\
         options:\n\
         \x20 --out-dir <dir>          write outputs here instead of beside each input\n\
         \x20 --weights <path>         checkpoint (default weights/enhancer_stage2.safetensors)\n\
         \x20 --nfe <n>                enhance solver steps, higher is slower/cleaner (default 64)\n\
         \x20 --tau <f>                enhance noise temperature in [0,1] (default 0.5)\n\
         \x20 --chunk-seconds <f>      gpu window per pass (default 5)\n\
         \x20 --overlap-seconds <f>    crossfaded overlap between windows (default 0.5)\n\
         \x20 --device <cpu|cuda:N>    (default cuda:0, falls back to cpu)"
    );
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut it = std::env::args().skip(1);
    let command = match it.next() {
        Some(c) if c == "denoise" || c == "enhance" => c,
        Some(c) if c == "--help" || c == "-h" => usage(),
        _ => usage(),
    };

    let mut inputs = Vec::new();
    let mut out_dir = None;
    let mut weights = PathBuf::from("weights/enhancer_stage2.safetensors");
    let mut nfe = 64usize;
    let mut tau = 0.5f64;
    let mut chunk_seconds = 5.0f64;
    let mut overlap_seconds = 0.5f64;
    let mut device = String::from("cuda:0");

    let rest: Vec<String> = it.collect();
    let mut i = 0usize;
    while i < rest.len() {
        let flag = rest[i].as_str();
        let value = || -> String { rest[i + 1].clone() };
        match flag {
            "--out-dir" => {
                out_dir = Some(PathBuf::from(value()));
                i += 2;
            }
            "--weights" => {
                weights = PathBuf::from(value());
                i += 2;
            }
            "--nfe" => {
                nfe = value().parse().unwrap_or_else(|_| usage());
                i += 2;
            }
            "--tau" => {
                tau = value().parse().unwrap_or_else(|_| usage());
                i += 2;
            }
            "--chunk-seconds" => {
                chunk_seconds = value().parse().unwrap_or_else(|_| usage());
                i += 2;
            }
            "--overlap-seconds" => {
                overlap_seconds = value().parse().unwrap_or_else(|_| usage());
                i += 2;
            }
            "--device" => {
                device = value();
                i += 2;
            }
            other => {
                inputs.push(PathBuf::from(other));
                i += 1;
            }
        }
    }

    if inputs.is_empty() {
        usage();
    }

    Args {
        command,
        inputs,
        out_dir,
        weights,
        nfe,
        tau,
        chunk_seconds,
        overlap_seconds,
        device,
    }
}

fn pick_device(spec: &str) -> Device {
    if let Some(ordinal) = spec.strip_prefix("cuda:") {
        let ordinal = ordinal.parse::<usize>().unwrap_or(0);
        if let Ok(d) = Device::new_cuda(ordinal) {
            return d;
        }
        eprintln!("device=cuda:{ordinal} unavailable, falling back to cpu");
    } else if spec == "cuda" {
        if let Ok(d) = Device::new_cuda(0) {
            return d;
        }
    }
    Device::Cpu
}

fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, u32), String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| format!("{path:?}: {e}"))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let to_mono = |s: &[f32]| -> f32 { s.iter().sum::<f32>() / channels as f32 };

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = 1.0f32 / ((1u64 << (spec.bits_per_sample - 1)) as f32);
            let mut acc = Vec::new();
            let mut frame = Vec::with_capacity(channels);
            for s in reader.samples::<i32>() {
                frame.push(s.map_err(|e| format!("{path:?}: {e}"))? as f32 * scale);
                if frame.len() == channels {
                    acc.push(to_mono(&frame));
                    frame.clear();
                }
            }
            acc
        }
        hound::SampleFormat::Float => {
            let mut acc = Vec::new();
            let mut frame = Vec::with_capacity(channels);
            for s in reader.samples::<f32>() {
                frame.push(s.map_err(|e| format!("{path:?}: {e}"))?);
                if frame.len() == channels {
                    acc.push(to_mono(&frame));
                    frame.clear();
                }
            }
            acc
        }
    };

    Ok((samples, spec.sample_rate))
}

fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<(), String> {
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .map_err(|e| format!("{path:?}: {e}"))?;
    for &s in samples {
        writer
            .write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)
            .map_err(|e| format!("{path:?}: {e}"))?;
    }
    writer.finalize().map_err(|e| format!("{path:?}: {e}"))
}

fn out_path(input: &Path, command: &str, out_dir: Option<&Path>) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    let name = format!("{stem}.{command}.wav");
    match out_dir {
        Some(dir) => dir.join(name),
        None => input.with_file_name(name),
    }
}

fn warm_allocator(device: &Device) {
    let mb: usize = std::env::var("RESOUND_WARMUP_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if mb == 0 {
        return;
    }
    match Tensor::zeros(mb * 1024 * 1024, DType::U8, device) {
        Ok(scratch) => drop(scratch),
        Err(e) => eprintln!("warmup skipped: {e}"),
    }
}

fn main() {
    let args = parse_args();
    let device = pick_device(&args.device);
    println!("device={device:?}");
    warm_allocator(&device);

    let vb = match unsafe {
        VarBuilder::from_mmaped_safetensors(&[args.weights.clone()], DType::F32, &device)
    } {
        Ok(vb) => vb,
        Err(e) => {
            eprintln!("failed to load {:?}: {e}", args.weights);
            std::process::exit(1);
        }
    };

    let denoiser = match Denoiser::new(vb.pp("denoiser")) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("failed to build denoiser: {e}");
            None
        }
    };
    let enhancer = match Enhancer::new(vb, &device) {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("failed to build enhancer: {e}");
            None
        }
    };

    if (args.command == "denoise" && denoiser.is_none())
        || (args.command == "enhance" && enhancer.is_none())
    {
        std::process::exit(1);
    }

    if let Some(dir) = &args.out_dir {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("failed to create out dir {dir:?}: {e}");
            std::process::exit(1);
        }
    }

    let mut total_audio_s = 0.0f64;
    let mut total_wall_s = 0.0f64;
    let mut failures = 0usize;

    for input in &args.inputs {
        let started = std::time::Instant::now();
        match process_file(&args, input, &device, denoiser.as_ref(), enhancer.as_ref()) {
            Ok((audio_s, out)) => {
                let wall = started.elapsed().as_secs_f64();
                total_audio_s += audio_s;
                total_wall_s += wall;
                println!(
                    "file={:?} out={:?} audio_s={audio_s:.2} wall_s={wall:.2} realtime_x={:.2}",
                    input,
                    out,
                    if wall > 0.0 { audio_s / wall } else { 0.0 }
                );
            }
            Err(e) => {
                failures += 1;
                eprintln!("skipped {input:?}: {e}");
            }
        }
    }

    println!(
        "batch_files={} failed={failures} audio_s={total_audio_s:.2} wall_s={total_wall_s:.2} realtime_x={:.2}",
        args.inputs.len(),
        if total_wall_s > 0.0 {
            total_audio_s / total_wall_s
        } else {
            0.0
        }
    );
}

fn process_file(
    args: &Args,
    input: &Path,
    device: &Device,
    denoiser: Option<&Denoiser>,
    enhancer: Option<&Enhancer>,
) -> Result<(f64, PathBuf), String> {
    let (raw, sample_rate) = read_wav_mono(input)?;
    if raw.is_empty() {
        return Err("empty audio".into());
    }

    let wav = if sample_rate as usize == TARGET_RATE {
        raw
    } else {
        Resampler::new(sample_rate as usize, TARGET_RATE).process(&raw)
    };

    let wav_len = wav.len();
    let cfg = ChunkConfig {
        sample_rate: TARGET_RATE,
        chunk_seconds: args.chunk_seconds,
        overlap_seconds: args.overlap_seconds,
    };
    let chunks = split_chunks(&wav, &cfg);
    drop(wav);
    let chunk_count = chunks.len();

    let lengths: Vec<usize> = chunks.iter().map(|(_, c)| c.len()).collect();
    let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(chunk_count);
    if args.command == "denoise" {
        let denoiser = denoiser.ok_or_else(|| "denoiser unavailable".to_string())?;
        for ((_, chunk), len) in chunks.iter().zip(&lengths) {
            let mut out = denoiser.forward(chunk, device).map_err(|e| e.to_string())?;
            out.truncate(*len);
            outputs.push(out);
        }
    } else {
        let enhancer = enhancer.ok_or_else(|| "enhancer unavailable".to_string())?;
        let waves: Vec<Vec<f32>> = chunks.into_iter().map(|(_, c)| c).collect();
        let started = std::time::Instant::now();
        let mut enhanced = enhancer
            .forward_many(&waves, args.nfe, args.tau, device)
            .map_err(|e| e.to_string())?;
        if std::env::var("RESOUND_PROFILE").is_ok() {
            eprintln!(
                "solve_group_chunks={} elapsed_ms={}",
                enhanced.len(),
                started.elapsed().as_millis()
            );
        }
        for (out, len) in enhanced.iter_mut().zip(&lengths) {
            out.truncate(*len);
        }
        outputs = enhanced;
    }

    let merged = merge_chunks(&outputs, cfg.overlap_len(), wav_len);
    let out = out_path(input, &args.command, args.out_dir.as_deref());
    write_wav(&out, &merged, TARGET_RATE as u32)?;

    Ok((wav_len as f64 / TARGET_RATE as f64, out))
}
