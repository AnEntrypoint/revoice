use std::collections::VecDeque;

pub struct ChunkConfig {
    pub sample_rate: usize,
    pub chunk_seconds: f64,
    pub overlap_seconds: f64,
}

impl ChunkConfig {
    pub fn chunk_len(&self) -> usize {
        (self.chunk_seconds * self.sample_rate as f64) as usize
    }

    pub fn overlap_len(&self) -> usize {
        (self.overlap_seconds * self.sample_rate as f64) as usize
    }

    pub fn hop_len(&self) -> usize {
        self.chunk_len() - self.overlap_len()
    }
}

pub struct ChunkMerger {
    overlap_len: usize,
    unflushable: usize,
    pending: VecDeque<f32>,
    pending_start: usize,
    total_len: usize,
    first: bool,
}

impl ChunkMerger {
    pub fn new(cfg: &ChunkConfig, total_len: usize) -> Self {
        let overlap_len = cfg.overlap_len();
        Self {
            overlap_len,
            unflushable: overlap_len,
            pending: VecDeque::new(),
            pending_start: 0,
            total_len,
            first: true,
        }
    }

    pub fn placed(&self) -> usize {
        self.pending_start + self.pending.len()
    }

    pub fn push(
        &mut self,
        chunk: &[f32],
        sink: &mut dyn FnMut(f32) -> Result<(), String>,
    ) -> Result<(), String> {
        if self.first {
            self.first = false;
            self.pending.extend(chunk.iter().copied());
            return self.drain(sink);
        }

        let aligned = self.placed().saturating_sub(self.overlap_len);
        let n_overlap = self
            .overlap_len
            .min(chunk.len())
            .min(self.total_len.saturating_sub(aligned));

        for k in 0..n_overlap {
            let fade = k as f32 / n_overlap.max(1) as f32;
            let slot = &mut self.pending[aligned + k - self.pending_start];
            *slot = *slot * (1.0 - fade) + chunk[k] * fade;
        }

        let tail = &chunk[n_overlap..];
        let room = self.total_len.saturating_sub(aligned + n_overlap);
        self.pending.extend(tail.iter().copied().take(room));
        self.drain(sink)
    }

    pub fn finish(
        &mut self,
        sink: &mut dyn FnMut(f32) -> Result<(), String>,
    ) -> Result<(), String> {
        let want = self.total_len.saturating_sub(self.pending_start);
        self.pending.truncate(want);
        while let Some(sample) = self.pending.pop_front() {
            sink(sample)?;
        }
        Ok(())
    }

    fn drain(&mut self, sink: &mut dyn FnMut(f32) -> Result<(), String>) -> Result<(), String> {
        while self.pending.len() > self.unflushable {
            let sample = self.pending.pop_front().unwrap_or(0.0);
            self.pending_start += 1;
            sink(sample)?;
        }
        Ok(())
    }
}

