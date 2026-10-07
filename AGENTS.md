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
| `RESOUND_DUMP_MEL` | write the conditioning mel and the decoded 160 ch to this dir as `.f32` |
| `RESOUND_COND_FILE` | skip mel/AE/CFM and vocode this `1x160xT` little-endian f32 instead |

## Measuring anything on this GPU

The card is shared with Chrome, which holds ~2.2 GB and can sit at 100 % util while nothing else
runs. Absolute timings are worthless across minutes: the same binary on the same file measured
36 s and 58 s, and with Chrome doing continuous GPU work a 60 s clip took 345 s (0.17x) where the
same clip took 36 s (1.67x) minutes later after `nvidia-smi --query-compute-apps` showed which
Chrome pids held CUDA contexts and those two were killed (GPU 100 %/2246 MiB -> 0 %/21 MiB). So
before any timing run: check `--query-compute-apps`, kill the pids holding contexts, and confirm
util is near 0 and memory near 0. Only compare runs that are back to back, and after sustained
load let it cool ~75 s first. Sharing costs ~10x, not 3x: the same 60 s clip at nfe 16 took
17.8 s alone and 183.9 s with one headless Chrome context beside it, and nfe 32 measured 2.72x
and 0.33x in the same hour.

Every Chrome on this box is headless gm automation (no window: check `MainWindowTitle`), and a new
one comes back within minutes of being killed, so one kill per piece is not enough —
`testsound/lectures/gpuwatch.sh` kills every non-resound CUDA client every 10 s while
`resound.exe` is alive. It kills Chrome's `--type=gpu-process` and not the browser, because Chrome
counts those as GPU crashes and drops to software compositing after a few — the one state in which
it stops asking for the card at all. `nvidia-smi -pl` and `-c` both answer Insufficient
Permissions here, so the power cap and exclusive compute mode are not available; the card runs
82-83 C at ~60 W with clocks pinned 1267/2100 MHz (SW Thermal Slowdown) and still delivers ~2.5x,
so heat is not worth waiting on, contention is. VRAM headroom decides the chunk size: chunk 5
needs ~3.6 GB, so it fits alone (6.1 GB) but not alongside Chrome, where the driver silently kills
the process. Prefer ratios inside one run over absolute ms.

Time A/B/A, never A/B: baseline, candidate, baseline again, with the same fixed rest between runs
(`testsound/lectures/ab*.sh` use 15-20 s). Drift then reads as a gap between the two baselines
instead of as a speedup.

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

nfe is the one knob that buys real time. On a 60 s clip with the card free, tf32 gave nfe 64
1.67x, nfe 32 2.72x, nfe 24 2.98x, nfe 16 3.17-3.37x (nfe 8 was not faster than 16). Quality,
each measured against nfe 32 on the same clip (per-frame log-mel cosine, mean / 1st percentile /
fraction of frames below 0.9, plus band-energy delta per decade from 0 to 22 kHz):

| nfe | cosine | p1 | frac < 0.9 | band delta | speed |
|---|---|---|---|---|---|
| 64 | 0.9609 | 0.871 | 3.0 % | within 0.17 dB | 1.0x |
| 24 | 0.9653 | — | — | within 0.39 dB (top band -0.11) | 1.10x |
| 16 | 0.9587 | 0.872 | 2.9 % | within 1.04 dB, all of it above 8 kHz | 1.24x |

nfe 16's deviation from nfe 32 is the same size and shape as nfe 32's from the upstream default of
64, and its HF noise floor over the input's 60 quietest frames is 0.7 dB *lower* than nfe 32's, so
it is not under-converged hiss — the extra ~1 dB sits in the band the model invents anyway (the
input has nothing above 8 kHz; the render is bandwidth extension). Chunk 10 was not faster than
chunk 5 (1.60x vs 1.67x) and costs ~900 MiB more, so the default stays 5.

## Verifying output

Two runs of the same config are bit-identical (fixed randn sequence). A change that alters the op
sequence or tensor shapes changes every RNG draw, so waveform correlation between two different
configs is ~0.016 and means nothing. Compare instead: per-frame log-mel cosine (0.97 between
equivalent renders, 0.78 against the input on lecture speech) and average magnitude spectrum.
Seam clicks: max adjacent-sample delta inside the overlap window must stay below the file's global
max delta. Folding the output's own RMS over the 4.5 s hop is not a seam test — speech dynamics
dominate it (the *source* folds to -6.7 dB at frame 0 where our render of a different file folds to
+5.9 dB). Fold the RMS of out/src instead, which removes the content: on a 60 s batch render the
overlap window averages -5.62 dB and the rest of the period -5.24 dB, with no dip at the boundary.

## Quality: how "damaged" was found and fixed

Upstream ground truth lives in `C:\dev\refenv` (Python 3.12, torch+cpu, `resemble-enhance` 0.0.1,
HF repo checked out at `C:\dev\refenv\Lib\site-packages\resemble_enhance\model_repo`). Run it with
`testsound/lectures/probe.py src.wav out.wav lambd nfe` (needs the `pathlib.PosixPath =
pathlib.WindowsPath` shim). It also monkeypatches `UnivNet.forward` and `IRMAE.encode` to dump the
conditioning mel (`out.wav.cond.f32`, 160 ch) and the latent, which is what makes module-level
comparison possible.

The metric that caught the damage is spectral flatness over 100-4000 Hz (geometric/arithmetic mean
of the power spectrum, `crest.py`) plus `hf.py`'s loud-vs-quiet band levels: real enhancement gates
HF with speech, hiss does not. On 60 s of lecture audio, source 0.0439 -> render 0.0153 flatness,
crest 14.35 -> 15.66 dB, 8-16 kHz from -69 dB (absent) to -29 dB loud but -58 dB quiet, and
`corr(log 300-3k, log 8-16k)` from 0.049 to 0.732. Upstream renders 0.0207-0.0211 flatness on the
same material, so that range is the target.

Localising it took dumping our own tensors and diffing against upstream's: `RESOUND_DUMP_MEL=<dir>`
writes the conditioning mel and the decoded 160 ch; `RESOUND_COND_FILE=<f32>` feeds a 160-ch cond
straight to the vocoder, skipping mel/AE/CFM. That last hook settled it — with upstream's own cond
mel our vocoder still produced flatness 0.1927, so the whole mel->IRMAE->CFM->decode path was fine
(our cond statistics match upstream's to 3 digits: std 1.244 vs 1.250, jitter 0.3217 vs 0.3215,
cos(cond128, mel) 0.5564 vs 0.5575) and the vocoder was not.

Two vocoder bugs were behind it, both invisible structurally:

1. `UpSample1d` scaled its Kaiser filter by `ratio²` where upstream (`univnet/alias_free_torch/
resample.py`) applies `ratio` once, as a gain on the transposed-conv output. Every AMP branch in
every LVC block therefore ran 2x too hot, and `SnakeBeta` is nonlinear so it did not cancel.
2. `KernelPredictor`'s trunk used LeakyReLU slope 0.1. Upstream's `KernelPredictor` defaults to
0.1, but `LVCBlock` overrides it with `lReLU_slope=0.2`, so 0.2 is what actually runs. This one
was the whole of the residual gap: same cond, before/after/upstream 8-16 kHz loud -32.4 / -24.7 /
-24.7 dB, quiet -58.6 / -72.7 / -72.1 dB, 16-22 kHz flatness 0.151 / 0.088 / 0.088. Our vocoder
now matches upstream's to rms 1.6e-4, corr 0.999999, on identical cond and noise. `conv_pre` and
`conv_post` also had to become reflect-padded (upstream `padding_mode="reflect"`).

On 60 s of lecture audio the whole pipeline now lands on upstream (nfe 32 / upstream nfe 32):
crest 15.32 / 15.31 dB, flatness 0.0205 / 0.0197, 8-16 kHz loud -19.50 / -18.58 dB and quiet
-68.79 / -68.12 dB, 0-100 Hz loud -7.62 / -7.59 dB, corr(log 300-3k, log 8-16k) 0.811 / 0.819.
The batch's nfe 16 is a hair milder (flatness 0.0217) and still 3.2x realtime. Confirmed on a
second lecture too (`Love Series 3B`, source flatness 0.0383): upstream nfe 32 renders 0.0282 and
our nfe 16 0.0294, crest 15.19 / 15.16 dB, 8-16 kHz loud -24.8 / -25.0 dB, corr 0.787 / 0.736 — so
flatness near 0.029 is that material, not a defect. Ours gates the pauses harder (300-3k quiet
-73.4 vs -64.4 dB, 8-16k quiet -75.4 vs -68.6 dB), which is the nfe-16 direction already measured. Two related port errors fixed at the same time: the Kaiser rolloff denominator (upstream is
`torch.kaiser_window(K, periodic=False)`, i.e. divide by `(K-1)/2`, not `K/2`) and the sinusoidal
time-embedding grid (`linspace(0, 4, 64)` has step `4/63`, not `4/64`). The `filter` buffers *are*
in the checkpoint (24 of them), so the Kaiser code itself is only a fallback.

`z_scale` is 6, not the upstream dataclass default of 5: `model_repo/enhancer_stage2/hparams.yaml`
says `lcfm_z_scale: 6` and `HParams.load` reads the yaml.

Still missing versus upstream: the denoiser stage (`lambd` in `enhance()`), which is a separate
model we do not ship, and `vocoder(npad=10)` (pad cond 10 frames, trim 4200 samples) — that one is
equivalent, both yield `T * 420` samples. Ours also does not drop the last mel frame the way
upstream's `to_mel(drop_last=True)` does; empirically both give the same frame count.

## Batch

`testsound/lectures/run_batch.sh` enhances the 588 lecture mp3s (447.8 h) into
`D:\Downloads\Manly_P.Hall_Enhanced` as 192 kbps mono mp3 (~39 GB, vs 89 GB flac / 142 GB wav).
Config: `GEMM=tf32 NFE=16 CHUNK=5 OVERLAP=0.5`, about 3.2x realtime on a free card, so ~6 days
plus ~20 h of decode/encode. Run it behind `gpuwatch.sh` (see above) or contention takes it to
0.33x. The 192 kbps mp3 encode is transparent: the same 60 s slice measures flatness 0.0290 taken
from a batch mp3 and 0.0294 from a wav render of the same source.

Ordering is shortest first (from `durations.tsv`), so the first files prove decode, enhance, piece
join and encode end to end in minutes. Every file is cut into equal 450 s pieces (0.5 s overlap)
and rejoined with `join.py`, which streams and keeps only the overlap tail. Output lands in
`C:\D\Downloads\Manly_P.Hall_Enhanced` — a `C:` path, like the source; `D:\temp_resound` is only
scratch.

Each file: decode to 44.1 kHz mono wav, enhance, encode to `$dest.part`, rename. An existing
non-empty output means done, so the run is resumable and reruns only pick up failures. Things
that bit once and are now guarded:

- `ffmpeg` reads stdin and will eat the file list out from under a `while read` loop; every call
  is `-nostdin` with `</dev/null`. `ffprobe` has no `-nostdin` (it errors on the flag), so it only
  gets `</dev/null`.
- The encode needs an explicit `-f mp3`: a `.part` suffix hides the extension `ffmpeg` guesses the
  muxer from.
- An empty `ffprobe` duration silently collapses the plan to one piece, which on a long file means
  one 1.2 GB enhance; the plan is validated and the file is skipped with `FAIL_PLAN`.
- Stopping a background task kills the wrapper, not the script, and two live instances share one
  temp dir and corrupt each other's pieces; the script now takes a `$TMP/.lock` pid lock.
- A silent resound death is VRAM or a driver reset: the script waits for the card under 85 C and
  under 1.5 GB used before each piece — capped at 12 polls x 20 s, so never more than 4 min — then
  retries twice at chunk 3. The cost is real: 5 of the first 6 files died once. The deaths are
  silent — rc=127, no stderr from candle, no Windows application or `nvlddmkm` event — and land
  mid-piece (6/24, 12/14 chunks) with the card otherwise at 0 % and host RAM at 7 GB free. Killing
  Chrome does not cause them (a chunk-5 run survived four `chrome.exe` GPU kills in 40 s), and
  chunk 5 peaks around 3 GB against a ~3.3 GB room once another client is on the card: a chunk-5
  run beside a chunk-3 one measured 5234 MiB and the chunk-3 process died as the other released.
  Working reading: a transient second CUDA client inside the peak, with the window set by how often
  the watcher polls — `gpuwatch.sh` and `gpuwatch_guard.sh` both poll (10 s and 3 s) rather than
  block, so a client can hold the card for up to that long.
  The temperature gate used to be 78 C, which bought minutes of idle per piece for nothing — at
  83 C and 1267 MHz a 539 s file still ran 2.57x. Pieces are 450 s because a longer one dies: candle has no pooling allocator, so
  every chunk is thousands of cudaMalloc/cudaFree cycles and a chunk-5 run over a whole 30 min
  piece dies past ~100 chunks, while the same audio in 450 s pieces does not.
- A run that prints `device=Cpu` has lost CUDA and is ~1000x slower while looking perfectly
  healthy. It happened once when the machine suspended mid-run and came back with
  `cuda:0 unavailable`; `run_enhance` now checks every piece for that line and stops the batch.
  `stayawake.ps1` holds an `ES_CONTINUOUS|ES_SYSTEM_REQUIRED|ES_DISPLAY_REQUIRED` request for the
  life of the run because `powercfg /change` and `/setacvalueindex` both answer Invalid Parameters
  here; the flags are written as the decimal 2147483651, since PowerShell parses `0x80000003` as a
  negative int and refuses the cast.
- Start it detached: `start_batch.bat` (and `start_watch.bat`) launched with `Start-Process`, since
  a `run_in_background` task dies with the session. `Start-Process -ArgumentList '-c','cmd'` does
  not quote the second element, so bash sees `-c cmd` and silently runs the first word — put the
  whole command in the .bat instead.

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
- `KernelPredictor`'s trunk slope is 0.2, not the 0.1 in `KernelPredictor.__init__`: `LVCBlock`
  passes `kpnet_nonlinear_activation_params={"negative_slope": lReLU_slope}` and overrides the
  default. Getting this wrong costs 8 dB of HF and 13 dB of noise floor, not an error.
