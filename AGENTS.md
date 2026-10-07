# resound

Rust port of the resemble-enhance inference pipeline (candle 0.10.2, CUDA). `resound enhance`
and `resound denoise` over wav files; long files are split into chunks and crossfaded back.

## Run

```
cargo build --release --bin resound
./target/release/resound.exe enhance --out-dir out --chunk-seconds 5 --overlap-seconds 0.5 in.wav...
```

Inputs are resampled to 44.1 kHz mono first. A file that fails (OOM) is skipped and the batch
continues.

## Knobs

| env | meaning |
|---|---|
| `RESOUND_GEMM` | `f32` (default) / `tf32` / `f16` gemm precision |
| `RESOUND_SOLVE_FRAMES` | mel frames per CFM solver call (default 1680) |
| `RESOUND_PROFILE` | `1` stage ms, `2` +gpu tick, `3` +per-layer tick |
| `RESOUND_WARMUP_MB` | preallocate this much VRAM before starting (default 0) |

## Measuring anything on this GPU

The card is shared with Chrome, which holds ~2.2 GB and can sit at 100 % util while nothing else
runs. Absolute timings are worthless across minutes: the same binary on the same file measured
36 s and 58 s, and with Chrome doing continuous GPU work a 60 s clip took 345 s (0.17x) where the
same clip took 36 s (1.67x) minutes later after `nvidia-smi --query-compute-apps` showed which
Chrome pids held CUDA contexts and those two were killed (GPU 100 %/2246 MiB -> 0 %/21 MiB). So
before any timing run: check `--query-compute-apps`, kill the pids holding contexts, and confirm
util is near 0 and memory near 0. Only compare runs that are back to back, and after sustained
load let it cool ~75 s first. VRAM headroom decides the chunk size: chunk 5 needs ~3.6 GB, so it
fits alone (6.1 GB) but not alongside Chrome, where the driver silently kills the process. Two hours
of near-continuous work took it to 88 °C and 3x slower on every GPU stage while the CPU-side mel
stage did not move at all — a stage that slows on GPU stages only is heat or contention, not a
regression. Prefer ratios inside one run over absolute ms.

## Memory

6 GB card. Peak VRAM is set by `--chunk-seconds`, not by file length: chunk 5 ≈ 3.2 GB,
chunk 10 ≈ 4.1 GB, chunk 15 ≈ 5.3 GB, chunk 20 OOMs. Past ~5.5 GB the driver kills the process
(exit 127, no message) or candle returns `CUDA_ERROR_OUT_OF_MEMORY`. Defaults are 5 s / 0.5 s
overlap: processed_seconds = audio_seconds * chunk / (chunk - overlap), so overlap is duplicated
work and 0.5 s keeps it at 11 % while staying well above the 2048-sample seam align window.

Host RAM, not VRAM, is the other limit on long files. `process_file` now streams: it drops the
source buffer right after resampling, cuts each 5 s window straight out of the resampled buffer,
enhances them 24 at a time, and hands each output to `ChunkMerger`, which keeps only the last
`overlap_len` samples and pushes everything older to the wav writer. Peak is one f32 per
input sample plus one batch (~21 MB) — about 640 MB for an hour of audio, where holding raw +
resampled + every chunk output + the merged buffer was ~2 GB and got a 58 min run killed.

## Where the time goes (15 s clip, chunk 5)

CFM solver ~62 %, vocoder ~38 %, mel well under 1 %. Total ≈ 1.6x realtime when the card is
cool, ~0.5x when it is hot (see above). Solver cost is ~0.8 s fixed plus ~2.9 ms per mel frame,
so one call covering several chunks amortises the fixed part: grouping 3 chunks per call beat
per-chunk solves 9.8 s vs 13.3 s on 15 s of audio. A batch of 3 files (88.8 s of audio) ran at
1.43x.

## Verifying output

Two runs of the same config are bit-identical (fixed randn sequence). A change that alters the op
sequence or tensor shapes changes every RNG draw, so waveform correlation between two different
configs is ~0.016 and means nothing. Compare instead: per-frame log-mel cosine (0.97 between
equivalent renders, 0.78 against the input on lecture speech) and average magnitude spectrum.
Seam clicks: max adjacent-sample delta inside the overlap window must stay below the file's global
max delta.

## Chunk seams

Each chunk's vocoder output is a fresh realization: the noise the vocoder draws per chunk decides
the waveform phase, so two renditions of the same audio correlate at ~0.1 at the sample level even
though their envelopes track at 0.78-0.85. That killed the old seam aligner, which cross-correlated
the previous chunk's tail against the next chunk's head: with no real lag to find it was picking
noise peaks, usually next to the +-2048-sample search edge, and then shifting the chunk by up to
46 ms — random time offsets, audio dropped or duplicated at every seam, and a file that ran out of
room before the last chunk. `ChunkMerger` now places every chunk at its nominal offset
(`placed() - overlap_len`) and only crossfades, which is exact by construction: the vocoder hop is
the mel hop (420), so output frame t belongs at sample t * 420.

The model does delay audio by ~120 ms (envelope best-lag 110-140 ms, the same for every chunk), a
constant of the mel framing, so it does not affect seams; the output is simply 120 ms late and the
last 120 ms of the source is never rendered.

## Model geometry

mel 128 ch, hop 420; AE latent 64; CFM WaveNet 30 layers, dilation cycle 5, hidden 512, nfe 64
gives 32 solver steps = 64 network evals; UnivNet cond 160, noise 128, channels 96,
strides [7,5,4,3], dilations [1,3,9,27].

## candle constraints

- No pooling allocator: every tensor is a `cudaMalloc`/`cudaFree`. Big short-lived buffers cost
  real time, not just bytes.
- `matmul` needs equal ranks; 2D @ 3D needs `broadcast_matmul`, which concretizes the broadcast
  (a copy). No beta-accumulate gemm is exposed.
- `set_gemm_reduced_precision_f32(true)` is TF32. f16 helped the CFM ~30 % and hurt the vocoder,
  so the default stays f32.

## Notes moved out of code

- `FastConv1d::im2col` stacks the dilated taps as `[channel, tap, frame]` so the flattened rows
  match the weight's own `(c_out, c_in * ksize)` layout and neither side needs a copy.
- `IM2COL_TILE_ELEMENTS` is the scratch budget for one tile; a tile is the widest run of frames
  whose `c_in * ksize * frames` fits. Tiling multiplies the op count, so the budget trades speed
  for memory directly.
- `FastConvTranspose1d` is one gemm per output phase: output sample `u * stride + r` collects
  every tap whose `j - padding` lands on phase `r`, and the phases are interleaved by a reshape
  instead of a scatter.
- `depthwise_correlate` is `out[c, t] = sum_j src[c, left + t * stride + j] * w[c, j]` with `src`
  already carrying the caller's padding. `depthwise_transpose` is its transpose:
  `out[t * stride + j - padding] += src[t] * w[c, j]`. Both come from aliasfree's Kaiser-sinc
  filters, which are per-channel and so cannot be a grouped convolution.
- `forward_frame_major` exists because UnivNet's kernel predictor slices per frame.
- `MAX_SOLVE_FRAMES` caps the stacked local conditioning buffer: it costs
  `frames * layers * 2 * hidden` floats (1680 frames ≈ 210 MB).
- `WaveNet::stacked_local_conditioning` is constant across the whole ODE, so the solver builds it
  once instead of once per step.
