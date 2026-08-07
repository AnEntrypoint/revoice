use realfft::RealFftPlanner;
use realfft::num_complex::Complex32;

pub struct Stft {
    n_fft: usize,
    hop_length: usize,
    window: Vec<f32>,
    planner_fwd: std::sync::Arc<dyn realfft::RealToComplex<f32>>,
    planner_inv: std::sync::Arc<dyn realfft::ComplexToReal<f32>>,
}

fn hann_window(win_length: usize) -> Vec<f32> {
    (0..win_length)
        .map(|i| {
            let x = std::f32::consts::PI * i as f32 / win_length as f32;
            x.sin().powi(2)
        })
        .collect()
}

fn centered_window(n_fft: usize, win_length: usize) -> Vec<f32> {
    let raw = hann_window(win_length);
    if win_length == n_fft {
        return raw;
    }
    let left = (n_fft - win_length) / 2;
    let mut full = vec![0.0f32; n_fft];
    full[left..left + win_length].copy_from_slice(&raw);
    full
}

impl Stft {
    pub fn new(n_fft: usize, hop_length: usize, win_length: usize) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        Stft {
            n_fft,
            hop_length,
            window: centered_window(n_fft, win_length),
            planner_fwd: planner.plan_fft_forward(n_fft),
            planner_inv: planner.plan_fft_inverse(n_fft),
        }
    }

    fn padded(&self, wav: &[f32]) -> Vec<f32> {
        let pad = self.n_fft / 2;
        let mut out = vec![0.0f32; pad + wav.len() + pad];
        out[pad..pad + wav.len()].copy_from_slice(wav);
        out
    }

    pub fn num_frames(&self, wav_len: usize) -> usize {
        let padded_len = wav_len + self.n_fft;
        1 + (padded_len - self.n_fft) / self.hop_length
    }

    pub fn forward(&self, wav: &[f32]) -> Vec<Vec<Complex32>> {
        let padded = self.padded(wav);
        let n_frames = self.num_frames(wav.len());
        let mut frames = Vec::with_capacity(n_frames);
        let mut scratch = self.planner_fwd.make_scratch_vec();
        for i in 0..n_frames {
            let start = i * self.hop_length;
            let mut input = self.planner_fwd.make_input_vec();
            for j in 0..self.n_fft {
                let sample = padded.get(start + j).copied().unwrap_or(0.0);
                input[j] = sample * self.window[j];
            }
            let mut output = self.planner_fwd.make_output_vec();
            self.planner_fwd
                .process_with_scratch(&mut input, &mut output, &mut scratch)
                .unwrap();
            frames.push(output);
        }
        frames
    }

    pub fn inverse(&self, frames: &[Vec<Complex32>], out_len: usize) -> Vec<f32> {
        let pad = self.n_fft / 2;
        let total_len = pad + out_len + pad + self.n_fft;
        let mut out = vec![0.0f32; total_len];
        let mut win_sum = vec![0.0f32; total_len];
        let mut scratch = self.planner_inv.make_scratch_vec();
        for (i, frame) in frames.iter().enumerate() {
            let start = i * self.hop_length;
            let mut input = self.planner_inv.make_input_vec();
            input.copy_from_slice(frame);
            let mut output = self.planner_inv.make_output_vec();
            self.planner_inv
                .process_with_scratch(&mut input, &mut output, &mut scratch)
                .unwrap();
            let norm = 1.0 / self.n_fft as f32;
            for j in 0..self.n_fft {
                out[start + j] += output[j] * norm * self.window[j];
                win_sum[start + j] += self.window[j] * self.window[j];
            }
        }
        for i in 0..total_len {
            if win_sum[i] > 1e-8 {
                out[i] /= win_sum[i];
            }
        }
        out[pad..pad + out_len].to_vec()
    }
}
