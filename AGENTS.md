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

The card is shared (Chrome + dwm hold ~0.9-1.5 GB and can sit at 100 % util). Absolute timings
are worthless across minutes: the same binary on the same file measured 36 s and 58 s. Only
compare runs that are back to back, and after sustained load let it cool ~75 s first. Two hours
of near-continuous work took it to 88 °C and 3x slower on every GPU stage while the CPU-side mel
stage did not move at all — a stage that slows on GPU stages only is heat or contention, not a
regression. Prefer ratios inside one run over absolute ms.

## Memory

6 GB card. Peak VRAM is set by `--chunk-seconds`, not by file length: chunk 5 ≈ 3.2 GB,
chunk 10 ≈ 4.1 GB, chunk 15 ≈ 5.3 GB, chunk 20 OOMs. Past ~5.5 GB the driver kills the process
(exit 127, no message) or candle returns `CUDA_ERROR_OUT_OF_MEMORY`. Defaults are 5 s / 0.5 s
overlap: processed_seconds = audio_seconds * chunk / (chunk - overlap), so overlap is duplicated
work and 0.5 s keeps it at 11 % while staying well above the 2048-sample seam align window.

Host RAM, not VRAM, is the other limit on long files: a 58 min input holds the resampled wav,
every chunk's output and the merged buffer at once, ~2 GB. `process_file` drops the resampled wav
as soon as the chunks are cut, which is what keeps that from being ~2.6 GB.

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
equivalent renders, 0.73 against the input) and average magnitude spectrum. Seam clicks: max
adjacent-sample delta inside the overlap window must stay below the file's global max delta.

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
