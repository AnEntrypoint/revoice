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
before any timing run: read `--query-compute-apps` and note who else is resident, because we no
longer kill them (see below), and take the number as a ratio between two runs made back to back
under the same residency. Only compare runs that are back to back, and after sustained
load let it cool ~75 s first. Sharing costs ~10x, not 3x: the same 60 s clip at nfe 16 took
17.8 s alone and 183.9 s with one headless Chrome context beside it, and nfe 32 measured 2.72x
and 0.33x in the same hour.

The machine is in use: **never kill Chrome.** `gpuwatch.sh`, `gpuwatch_guard.sh` and
`gpuwatch_event.ps1` all killed its `--type=gpu-process` on a timer; that is off now, and the batch
just shares the card. Sharing sets the chunk size instead: the other client holds ~2.2 GB and
chunk 5 needs ~3.2 GB, so chunk 5 only fits when the card is otherwise free, and the batch asks
`nvidia-smi` per file (see Batch). `nvidia-smi -pl` and `-c` both answer Insufficient Permissions
here, so the power cap and exclusive compute mode are not available; the card runs
82-83 C at ~60 W with clocks pinned 1267/2100 MHz (SW Thermal Slowdown) and still delivers ~2.5x,
so heat is not worth waiting on, contention is. Prefer ratios inside one run over absolute ms.

Time A/B/A, never A/B: baseline, candidate, baseline again, with the same fixed rest between runs
(`testsound/lectures/ab*.sh` use 15-20 s). Drift then reads as a gap between the two baselines
instead of as a speedup.

## Memory

6 GB card. Peak VRAM is set by `--chunk-seconds`, not by file length: chunk 5 ≈ 3.2 GB,
chunk 10 ≈ 4.1 GB, chunk 15 ≈ 5.3 GB, chunk 20 OOMs. Past ~5.5 GB the driver kills the process
(exit 127, no message) or candle returns `CUDA_ERROR_OUT_OF_MEMORY`. Defaults are 5 s / 0.5 s
overlap: processed_seconds = audio_seconds * chunk / (chunk - overlap), so overlap is duplicated
work. The batch runs 0.25 s, which duplicates 5 % of the audio instead of 11 % and is measurably
faster: 180 s of lecture at chunk 5 took 53.3 s at 0.25 s against 55.7 s at 0.5 s, A/B/A/B on an
idle card, with flatness 0.0133 vs 0.0127 and no seam delta above the file's global max. The
window no longer has to clear a 2048-sample search window — that was the cross-correlation seam
aligner, which is gone — only to be long enough to crossfade, and the only offset it has to absorb
is the model's constant ~120 ms of delay, which is why 0.25 s and not 0.15 s: below that the
margin over the delay is down to 30 ms.

Host RAM, not VRAM, is the other limit on long files. `process_file` now streams: it drops the
source buffer right after resampling, cuts each 5 s window straight out of the resampled buffer,
enhances them 24 at a time, and hands each output to `ChunkMerger`, which keeps only the last
`overlap_len` samples and pushes everything older to the wav writer. Peak is one f32 per
input sample plus one batch (~21 MB) — about 640 MB for an hour of audio, where holding raw +
resampled + every chunk output + the merged buffer was ~2 GB and got a 58 min run killed.

## Where the time goes (15 s clip, chunk 5)

CFM solver ~62 %, vocoder ~38 %, mel well under 1 %. Total ≈ 1.6x realtime when the card is
cool, ~0.5x when it is hot (see above); on an otherwise idle card a single 180 s file at chunk 5,
nfe 16, tf32 measures 3.2-3.4x, which is the ceiling the batch is measured against. Solver cost is ~0.8 s fixed plus ~2.9 ms per mel frame,
so one call covering several chunks amortises the fixed part: grouping 3 chunks per call beat
per-chunk solves 9.8 s vs 13.3 s on 15 s of audio. A batch of 3 files (88.8 s of audio) ran at
1.43x. Two knobs that looked like levers and are not: f16 for the CFM alone (`RESOUND_GEMM=tf32`
with every CFM conv cast to f16) measured 50.7 s against 46.9 s for plain tf32 on 180 s at chunk 5
— the ~30 % that f16 wins over f32 does not survive tf32 already being on — and
`RESOUND_SOLVE_FRAMES=3360` against 1680 was 52.8 s vs 53.1 s, i.e. the ~0.8 s per solver call is
already amortised at three chunks per call.

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
input has nothing above 8 kHz; the render is bandwidth extension). Chunk 10 is only 2 % faster
than chunk 5 — 378 s of lecture, A/B/A/B on an idle card, 112.9 and 114.0 s against 114.8 and
117.1 s — and costs ~900 MiB more, so the default stays 5, and the batch does not take it either
(see Batch): nothing reaches the wav until all 24 chunks of a batch are rendered, so its 234 s of
audio per batch doubles the window in which a death loses the whole piece. The older note that
chunk 10 was slower (1.60x vs 1.67x) was measured on a shared card and does not hold.

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

`testsound/lectures/qa.py` is that check as a sweep: it takes finished mp3s, decodes 60 s at 300 s
from each and prints flatness, corr and the loud-vs-quiet drop, aggregated the way `crest.py` does
it — per frame, then the median across frames. Averaging the spectra first instead weights the loud
frames and reads ~2x lower, so the two are not interchangeable. Over the first 28 finished lectures:
flatness 0.0157-0.0375, corr 0.634-0.889, drop 44-273 dB, nothing flagged (thresholds flat < 0.040,
corr > 0.50, drop > 25 dB). The spread is the material, not damage: the least clean render (0.0375)
comes from a 0.0887 source and the cleanest (0.0157) from a 0.0363 one, a 2.3-3.2x collapse either
way. It deletes its slice before decoding, because a decode that fails silently would otherwise be
measured again under the next file's name.

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
`C:\D\Downloads\Manly_P.Hall_Enhanced` as 192 kbps mono mp3 (~39 GB, vs 89 GB flac / 142 GB wav).
Config: `GEMM=tf32 NFE=16 OVERLAP=0.25`, with chunk and piece length picked per file by
`pick_chunk()`: chunk 5 with 450 s pieces above `CHUNK_FREE_MIB=4000` MiB free, else chunk 3 with
250 s pieces — and chunk 3 is also the retry chunk under a 5 (2 under a 3). Overlap is 0.25 s because
it is 5 % duplicated audio instead of 11 % and measures 4.5 % faster (see Memory). Chunk 10 was
tried as a third tier above 5000 MiB and reverted: 2 % faster but 234 s of audio per flush batch
instead of 114 s, so it doubles the window in which a death loses the whole piece, and it died
38 s into its first production piece. It is worth taking the bigger window when it fits: in a
45 min window chunk 5 rendered ~4000 s of audio in 340/349/401/405 s (2.67x) against chunk 3's
baseline of 390/486/519 s for ~2755 s (1.97x), and it duplicates 11 % of the audio instead of 20 %.
Deaths scale with process starts as well as time under load — 225 s pieces lost 11.5 per
audio-hour, chunk 3 at 450 s 6.5, chunk 5 at 450 s 4.5 — which is why the 225 s experiment was
reverted. Pieces measure 3.26-3.38x and a whole ~1122 s file lands at 3.0x end to end — 371 s for
decode, three pieces, join and encode — so ~6 days of enhancing for the 447.8 h rather than ~9,
plus ~20 h of decode/encode.
The 192 kbps mp3 encode is transparent: the same
60 s slice measures flatness 0.0290 taken from a batch mp3 and 0.0294 from a wav render of the same
source.

Ordering is shortest first (from `durations.tsv`), so the first files prove decode, enhance, piece
join and encode end to end in minutes. Every file is cut into equal pieces (0.25 s overlap; 450 s at
chunk 5, 250 s at chunk 3) and rejoined with `join.py`, which streams and keeps only the overlap
tail. Output lands in
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
  temp dir and corrupt each other's pieces; the script now takes a `$TMP/.lock` pid lock. Never
  edit `run_batch.sh` while a run is live: bash holds a byte offset into the script, so a rewrite
  of a different length gets re-read from the wrong place. Stop the whole `bash.exe` tree (not just
  the wrapper), `rm -rf $TMP/.lock`, edit, restart.
- A silent resound death is the GPU falling over, not contention: rc=127, no message, and an
  nvlddmkm Event ID 153 ("Error occurred on GPUID: 100") beside it in the System log (12 in 3 h;
  09:10:13 second-exact with a death timed from the log). Query it by `ProviderName`, not
  `Message` — the message is null there, so a message-match filter finds nothing. `TdrDelay` and
  `TdrLevel` are unset, so WDDM's 2 s default applies and raising them would need admin and a
  reboot. It is not VRAM and not Chrome: it happens at chunk 3 (~2.3 GB) with nothing else on the
  card, and killing Chrome's gpu-process at creation (`gpuwatch_event.ps1`, 40 kills) did not stop
  it. What varies is the rate — roughly one per 10-15 min of load, about one per file — and they
  arrive in bursts, so it is not worth preventing, only worth making cheap.
  The cost of one death is the wasted render plus a re-render of the piece plus whatever wait
  precedes the retry. Nothing reaches the wav until every chunk of a batch is rendered — killing a
  180 s render 42 s in left 5025770 samples, exactly 24 chunks' worth of 4.75 s hop, and nothing of
  the batch in flight — so `CHUNKS_PER_BATCH` is the knob on the wasted part: 24 risks 114 s of
  audio, 8 risks 38 s. It is not worth spending: 8 measured 52.96/53.30 s against 48.59/52.84 s
  for 24 on 180 s, but the card drifted 9 % across those four runs, so the pair is not separable
  and 24 stays. So on failure `resume_piece` goes first: hound writes the header when the
  file is created and only rewrites the sizes on finalize, so a killed run leaves a wav whose
  header claims 0 frames while every flushed sample is there — `wavefix.py` prints the true length
  off the file size, the remainder is cut from `(n - OV) / 44100` s and rendered, and `join.py`
  crossfades it onto the tail with the same window the pieces use (`OV`, 0.25 s). Measured on 120 s:
  resumed 120.00 s against a clean 120.00 s (6 samples out of 5.29 M), seam max delta 0.049
  against a global 0.352, and the audio before the seam bit-identical to the clean render. First
  production use: a 312 s piece died 60 s in, its 252.58 s remainder rendered at 2.90x and joined
  to 312.08 s against the sibling piece's 312.05 s, seam max delta 0.0396 against a global 0.2917,
  finished mp3 933.83 s against a 933.67 s source.
  A resume render can die too, and it leaves its own partial in `resume.enhance.wav`, which a retry
  used to delete — one death threw away 205 s of already-rendered audio that way. `salvage_resume`
  joins it onto the piece before retrying, so up to three attempts accumulate progress and only the
  last resort re-renders the whole piece at chunk 2.
  Waiting is what used to make a death expensive, and it was unconditional: `settle()` waited for
  two minutes of quiet in the event log before every piece. `ready()` asks the card instead — one
  1 s render at nfe 8 over `probe1s.wav`, ~4 s — and settles only when that fails, so a healthy
  card costs 4 s a piece instead of 126 s. Bursts are still real (six events in four minutes once)
  and a retry landing inside one loses the file rather than the piece — `Love Series 1B - Human
  Love` went exactly that way and recovered on the next run — so the wait stays, behind the probe.
- The encode is pure CPU and the enhance is GPU, so one file's encode no longer sits in front of
  the next one: `$TMP/in.enhance.wav` is renamed to `enc.wav` and handed to ffmpeg in the
  background, and `finish_encode()` settles the `.part` rename a whole file later, at the point the
  next encode would be launched (and once more after the loop ends) — 8-18 s per file the card used
  to idle through. Settling it at the top of the next iteration instead only moves the wait, it does
  not hide it: `WAIT_encode` read 19 s both ways.
- Pieces are 450 s because a longer one dies: candle has no pooling allocator, so every chunk is
  thousands of cudaMalloc/cudaFree cycles and a chunk-5 run over a whole 30 min piece dies past
  ~100 chunks, while the same audio in 450 s pieces does not. The gate before a piece is under
  85 C and under 1.5 GB used, capped at 12 polls x 20 s so never more than 4 min; it used to be
  78 C, which bought minutes of idle per piece for nothing — at 83 C and 1267 MHz a 539 s file
  still ran 2.57x.
- A run that prints `device=Cpu` has lost CUDA and is ~1000x slower while looking perfectly
  healthy. It happened once when the machine suspended mid-run and came back with
  `cuda:0 unavailable`; `run_enhance` now checks every piece for that line and stops the batch.
  `stayawake.ps1` holds an `ES_CONTINUOUS|ES_SYSTEM_REQUIRED|ES_DISPLAY_REQUIRED` request for the
  life of the run because `powercfg /change` and `/setacvalueindex` both answer Invalid Parameters
  here; the flags are written as the decimal 2147483651, since PowerShell parses `0x80000003` as a
  negative int and refuses the cast.
- Start it detached: `start_batch.bat` launched with `Start-Process`, since a `run_in_background`
  task dies with the session. `Start-Process -ArgumentList '-c','cmd'` does not quote the second
  element, so bash sees `-c cmd` and silently runs the first word — put the whole command in the
  .bat instead. `start_watch.bat`, `start_guard.bat` and `start_event.bat` launch the Chrome
  killers, which are switched off (see above); do not start them.

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
