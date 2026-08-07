fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

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

fn kaiser_beta(lowpass_filter_width: usize) -> f64 {
    let _ = lowpass_filter_width;
    14.769656459379492
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-20 {
        1.0
    } else {
        (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
    }
}

pub struct Resampler {
    orig_freq: usize,
    new_freq: usize,
    kernel: Vec<Vec<f64>>,
    width: usize,
}

impl Resampler {
    pub fn new(orig_freq: usize, new_freq: usize) -> Self {
        let g = gcd(orig_freq, new_freq);
        let orig_freq = orig_freq / g;
        let new_freq = new_freq / g;

        let lowpass_filter_width = 64usize;
        let rolloff = 0.9475937167399596f64;
        let beta = kaiser_beta(lowpass_filter_width);

        let base_freq = (orig_freq.min(new_freq)) as f64 * rolloff;
        let width = (lowpass_filter_width as f64 * orig_freq as f64 / base_freq).ceil() as usize;

        let mut kernel = vec![vec![0.0f64; 2 * width + 1]; new_freq];
        for i in 0..new_freq {
            let t = i as f64 / new_freq as f64 * orig_freq as f64;
            for (j, k) in kernel[i].iter_mut().enumerate() {
                let idx = j as i64 - width as i64;
                let time = idx as f64 - (t - t.round());
                let val = sinc(time * base_freq / orig_freq as f64) * base_freq / orig_freq as f64;
                let window_arg = std::f64::consts::PI * beta
                    * (1.0 - (time / width as f64).powi(2)).max(0.0).sqrt();
                let kaiser = i0(window_arg) / i0(std::f64::consts::PI * beta);
                *k = val * kaiser;
            }
        }

        Resampler {
            orig_freq,
            new_freq,
            kernel,
            width,
        }
    }

    pub fn process(&self, input: &[f32]) -> Vec<f32> {
        let out_len = (input.len() * self.new_freq + self.orig_freq - 1) / self.orig_freq;
        let mut out = vec![0.0f32; out_len];
        for i in 0..out_len {
            let t = i as f64 / self.new_freq as f64 * self.orig_freq as f64;
            let center = t.round() as i64;
            let mut acc = 0.0f64;
            for (j, &k) in self.kernel[i % self.new_freq].iter().enumerate() {
                let src_idx = center + (j as i64 - self.width as i64);
                if src_idx >= 0 && (src_idx as usize) < input.len() {
                    acc += k * input[src_idx as usize] as f64;
                }
            }
            out[i] = acc as f32;
        }
        out
    }
}
