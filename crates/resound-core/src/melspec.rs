use crate::stft::Stft;

fn hz_to_mel(hz: f32) -> f32 {
    let f_min = 0.0f32;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0f32;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if hz < min_log_hz {
        (hz - f_min) / f_sp
    } else {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    let f_min = 0.0f32;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0f32;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if mel < min_log_mel {
        f_min + f_sp * mel
    } else {
        min_log_hz * ((mel - min_log_mel) * logstep).exp()
    }
}

fn slaney_mel_filterbank(
    n_fft: usize,
    n_mels: usize,
    sample_rate: f32,
    f_min: f32,
    f_max: f32,
) -> Vec<Vec<f32>> {
    let n_freqs = n_fft / 2 + 1;
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * sample_rate / n_fft as f32)
        .collect();

    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let mel_pts: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();
    let hz_pts: Vec<f32> = mel_pts.iter().map(|&m| mel_to_hz(m)).collect();

    let mut filters = vec![vec![0.0f32; n_freqs]; n_mels];
    for m in 0..n_mels {
        let (left, center, right) = (hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]);
        let enorm = 2.0 / (right - left);
        for (k, &freq) in fft_freqs.iter().enumerate() {
            let up = (freq - left) / (center - left);
            let down = (right - freq) / (right - center);
            let w = up.min(down).max(0.0);
            filters[m][k] = w * enorm;
        }
    }
    filters
}

pub struct MelSpectrogram {
    stft: Stft,
    filterbank: Vec<Vec<f32>>,
    preemphasis: f32,
}

impl MelSpectrogram {
    pub fn new(
        sample_rate: f32,
        n_fft: usize,
        hop_length: usize,
        win_length: usize,
        n_mels: usize,
        f_min: f32,
        f_max: f32,
        preemphasis: f32,
    ) -> Self {
        MelSpectrogram {
            stft: Stft::new(n_fft, hop_length, win_length),
            filterbank: slaney_mel_filterbank(n_fft, n_mels, sample_rate, f_min, f_max),
            preemphasis,
        }
    }

    fn apply_preemphasis(&self, wav: &[f32]) -> Vec<f32> {
        if self.preemphasis == 0.0 {
            return wav.to_vec();
        }
        let mut out = vec![0.0f32; wav.len()];
        out[0] = wav[0];
        for i in 1..wav.len() {
            out[i] = wav[i] - self.preemphasis * wav[i - 1];
        }
        out
    }

    pub fn forward(&self, wav: &[f32]) -> Vec<Vec<f32>> {
        let pre = self.apply_preemphasis(wav);
        let frames = self.stft.forward(&pre);
        let n_frames = frames.len();
        let n_mels = self.filterbank.len();
        let mut mel = vec![vec![0.0f32; n_frames]; n_mels];
        for (t, frame) in frames.iter().enumerate() {
            let mag: Vec<f32> = frame.iter().map(|c| c.norm()).collect();
            for m in 0..n_mels {
                let mut acc = 0.0f32;
                for (k, &w) in self.filterbank[m].iter().enumerate() {
                    if w > 0.0 {
                        acc += w * mag[k];
                    }
                }
                mel[m][t] = acc;
            }
        }
        mel
    }
}

pub fn amp_to_db(x: f32, stft_magnitude_min: f32) -> f32 {
    20.0 * x.max(stft_magnitude_min).log10()
}

pub fn normalize_db(db: f32, min_level_db: f32) -> f32 {
    (db - min_level_db) / (-min_level_db + 15.0)
}
