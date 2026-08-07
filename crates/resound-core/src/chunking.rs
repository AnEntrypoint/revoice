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

pub fn split_chunks(wav: &[f32], cfg: &ChunkConfig) -> Vec<(usize, Vec<f32>)> {
    let chunk_len = cfg.chunk_len();
    let hop_len = cfg.hop_len();
    if wav.len() <= chunk_len {
        return vec![(0, wav.to_vec())];
    }
    let mut chunks = Vec::new();
    let mut start = 0usize;
    loop {
        let end = (start + chunk_len).min(wav.len());
        chunks.push((start, wav[start..end].to_vec()));
        if end == wav.len() {
            break;
        }
        start += hop_len;
    }
    chunks
}

fn cross_correlate_offset(prev_tail: &[f32], cur_head: &[f32], max_shift: usize) -> i64 {
    let mut best_shift = 0i64;
    let mut best_score = f64::MIN;
    let n = prev_tail.len().min(cur_head.len());
    for shift in -(max_shift as i64)..=(max_shift as i64) {
        let mut score = 0.0f64;
        let mut count = 0usize;
        for i in 0..n {
            let j = i as i64 + shift;
            if j >= 0 && (j as usize) < cur_head.len() {
                score += (prev_tail[i] as f64) * (cur_head[j as usize] as f64);
                count += 1;
            }
        }
        if count > 0 {
            let normalized = score / count as f64;
            if normalized > best_score {
                best_score = normalized;
                best_shift = shift;
            }
        }
    }
    best_shift
}

pub fn merge_chunks(chunks: &[Vec<f32>], overlap_len: usize, total_len: usize) -> Vec<f32> {
    if chunks.len() == 1 {
        let mut out = chunks[0].clone();
        out.truncate(total_len);
        return out;
    }
    let mut out = vec![0.0f32; total_len];
    let mut write_pos = 0usize;

    for (i, chunk) in chunks.iter().enumerate() {
        if i == 0 {
            let n = chunk.len().min(total_len);
            out[..n].copy_from_slice(&chunk[..n]);
            write_pos = n;
            continue;
        }

        let search_window = (overlap_len / 4).max(1);
        let prev_tail_start = write_pos.saturating_sub(overlap_len);
        let prev_tail = &out[prev_tail_start..write_pos];
        let cur_head = &chunk[..overlap_len.min(chunk.len())];
        let shift = cross_correlate_offset(prev_tail, cur_head, search_window);

        let aligned_start = (write_pos as i64 - overlap_len as i64 + shift).max(0) as usize;
        let n_overlap = overlap_len.min(chunk.len()).min(total_len - aligned_start);

        for k in 0..n_overlap {
            let fade = k as f32 / n_overlap.max(1) as f32;
            let idx = aligned_start + k;
            if idx < total_len {
                out[idx] = out[idx] * (1.0 - fade) + chunk[k] * fade;
            }
        }

        let tail_start = aligned_start + n_overlap;
        let tail_src_start = n_overlap;
        let remaining = chunk.len().saturating_sub(tail_src_start);
        let copy_len = remaining.min(total_len.saturating_sub(tail_start));
        if copy_len > 0 {
            out[tail_start..tail_start + copy_len]
                .copy_from_slice(&chunk[tail_src_start..tail_src_start + copy_len]);
        }
        write_pos = tail_start + copy_len;
    }
    out
}
