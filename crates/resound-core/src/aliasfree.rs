use candle_core::{Device, Result, Tensor};
use candle_nn::{conv1d, Conv1d, Conv1dConfig, Module, VarBuilder};

fn i0(x: f64) -> f64 {
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let half_x = x / 2.0;
    for k in 1..64 {
        term *= (half_x * half_x) / (k as f64 * k as f64);
        sum += term;
        if term < 1e-16 * sum {
            break;
        }
    }
    sum
}

fn kaiser_beta_from_attenuation(a: f64) -> f64 {
    if a > 50.0 {
        0.1102 * (a - 8.7)
    } else if a >= 21.0 {
        0.5842 * (a - 21.0).powf(0.4) + 0.07886 * (a - 21.0)
    } else {
        0.0
    }
}

pub fn kaiser_sinc_filter1d(cutoff: f64, half_width: f64, kernel_size: usize) -> Vec<f32> {
    let delta_f = 4.0 * half_width;
    let half_size = kernel_size as f64 / 2.0;
    let a = 2.285 * (half_size - 1.0) * std::f64::consts::PI * delta_f + 7.95;
    let beta = kaiser_beta_from_attenuation(a);
    let denom = i0(beta);

    let mut filt = vec![0.0f64; kernel_size];
    let mut sum = 0.0f64;
    for i in 0..kernel_size {
        let t = i as f64 - (kernel_size as f64 - 1.0) / 2.0;
        let sinc_val = if t.abs() < 1e-9 {
            2.0 * cutoff
        } else {
            (2.0 * std::f64::consts::PI * cutoff * t).sin() / (std::f64::consts::PI * t)
        };
        let ratio = t / half_size;
        let arg = beta * (1.0 - ratio * ratio).max(0.0).sqrt();
        let window = i0(arg) / denom;
        let v = sinc_val * window;
        filt[i] = v;
        sum += v;
    }
    filt.iter().map(|&v| (v / sum) as f32).collect()
}

fn pad_replicate1d(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    let first = x.narrow(2, 0, 1)?;
    let last = x.narrow(2, x.dim(2)? - 1, 1)?;
    let mut parts = Vec::with_capacity(left + right + 1);
    for _ in 0..left {
        parts.push(first.clone());
    }
    parts.push(x.clone());
    for _ in 0..right {
        parts.push(last.clone());
    }
    Tensor::cat(&parts, 2)
}

pub struct LowPassFilter1d {
    kernel: Tensor,
    stride: usize,
    pad: usize,
    channels: usize,
}

impl LowPassFilter1d {
    pub fn new(ratio: usize, kernel_size: usize, channels: usize, vb: VarBuilder, device: &Device) -> Result<Self> {
        let cutoff = 0.5 / ratio as f64;
        let half_width = 0.6 / ratio as f64;
        let filt = if let Ok(t) = vb.get(kernel_size, "filter") {
            t.flatten_all()?.to_vec1::<f32>()?
        } else {
            kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        };
        let kernel = Tensor::from_vec(filt, (1, 1, kernel_size), device)?
            .broadcast_as((channels, 1, kernel_size))?
            .contiguous()?;
        let pad = kernel_size / 2 - 1;
        Ok(Self {
            kernel,
            stride: ratio,
            pad,
            channels,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = pad_replicate1d(x, self.pad, self.pad)?;
        x.conv1d(&self.kernel, 0, self.stride, 1, self.channels)
    }
}

pub struct UpSample1d {
    kernel: Tensor,
    ratio: usize,
    channels: usize,
    pad: usize,
    pad_left: usize,
    pad_right: usize,
}

impl UpSample1d {
    pub fn new(ratio: usize, kernel_size: usize, channels: usize, vb: VarBuilder, device: &Device) -> Result<Self> {
        let cutoff = 0.5 / ratio as f64;
        let half_width = 0.6 / ratio as f64;
        let filt = if let Ok(t) = vb.get(kernel_size, "filter") {
            t.flatten_all()?.to_vec1::<f32>()?
        } else {
            kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        };
        let filt: Vec<f32> = filt.iter().map(|&v| v * ratio as f32).collect();
        let kernel = Tensor::from_vec(filt, (1, 1, kernel_size), device)?
            .broadcast_as((channels, 1, kernel_size))?
            .contiguous()?;
        let pad = kernel_size / ratio - 1;
        let pad_left = pad * ratio + (kernel_size - ratio) / 2;
        let pad_right = pad * ratio + (kernel_size - ratio + 1) / 2;
        Ok(Self {
            kernel,
            ratio,
            channels,
            pad,
            pad_left,
            pad_right,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = pad_replicate1d(x, self.pad, self.pad)?;
        let out = x.conv_transpose1d(&self.kernel, 0, 0, self.ratio, 1, self.channels)?;
        let out = (out * self.ratio as f64)?;
        let len = out.dim(2)?;
        let new_len = len.saturating_sub(self.pad_left + self.pad_right);
        out.narrow(2, self.pad_left, new_len)
    }
}

pub struct SnakeBeta {
    log_alpha: Tensor,
    log_beta: Tensor,
}

impl SnakeBeta {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let log_alpha = vb.get_with_hints(channels, "log_alpha", candle_nn::init::ZERO)?;
        let log_beta = vb.get_with_hints(channels, "log_beta", candle_nn::init::ZERO)?;
        Ok(Self { log_alpha, log_beta })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let alpha = self.log_alpha.exp()?.clamp(1e-2, 50.0)?.reshape((1, (), 1))?;
        let beta = self.log_beta.exp()?.clamp(1e-2, 50.0)?.reshape((1, (), 1))?;
        let sin_ax = x.broadcast_mul(&alpha)?.sin()?;
        let term = sin_ax.sqr()?.broadcast_div(&beta)?;
        x + term
    }
}

pub struct UpActDown {
    upsample: UpSample1d,
    act: SnakeBeta,
    downsample: LowPassFilter1d,
}

impl UpActDown {
    pub fn new(channels: usize, vb: VarBuilder, device: &Device) -> Result<Self> {
        Ok(Self {
            upsample: UpSample1d::new(2, 12, channels, vb.pp("upsample"), device)?,
            act: SnakeBeta::new(channels, vb.pp("act"))?,
            downsample: LowPassFilter1d::new(2, 12, channels, vb.pp("downsample.lowpass"), device)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.upsample.forward(x)?;
        let x = self.act.forward(&x)?;
        self.downsample.forward(&x)
    }
}

pub struct AmpBlock {
    layers: Vec<(Conv1d, UpActDown, Conv1d)>,
}

impl AmpBlock {
    pub fn new(channels: usize, dilations: &[usize], vb: VarBuilder, device: &Device) -> Result<Self> {
        let mut layers = Vec::with_capacity(dilations.len());
        for (i, &d) in dilations.iter().enumerate() {
            let layer_vb = vb.pp(i);
            let cfg1 = Conv1dConfig {
                padding: d,
                dilation: d,
                ..Default::default()
            };
            let conv1 = conv1d(channels, channels, 3, cfg1, layer_vb.pp("0"))?;
            let act = UpActDown::new(channels, layer_vb.pp("1"), device)?;
            let cfg2 = Conv1dConfig {
                padding: 1,
                ..Default::default()
            };
            let conv2 = conv1d(channels, channels, 3, cfg2, layer_vb.pp("2"))?;
            layers.push((conv1, act, conv2));
        }
        Ok(Self { layers })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for (conv1, act, conv2) in &self.layers {
            h = conv1.forward(&h)?;
            h = act.forward(&h)?;
            h = conv2.forward(&h)?;
        }
        x + h
    }
}
