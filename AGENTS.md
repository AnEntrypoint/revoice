# resound

Rust port of the resemble-enhance inference pipeline (candle 0.10.2, CUDA). `resound enhance`
and `resound denoise` over wav files; long files are split into chunks and crossfaded back.

## Run

```
cargo build --release --bin resound
./target/release/resound.exe enhance --out-dir out --chunk-seconds 5 --overlap-seconds 0.5 in.wav...
```

Inputs are resampled to 44.1 kHz mono first. A file that fails (OOM) is skipped and the batch
continues. Run from `C:\dev\resound`: weights load from a relative `weights/`, and paths must be
`D:/...` — `/d/...` and a foreign cwd both fail.

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

The machine is in use and the card is shared with Chrome, which holds ~2.2 GB and can sit at 100 %
util while nothing else runs: **never kill Chrome.** `gpuwatch.sh`, `gpuwatch_guard.sh` and
`gpuwatch_event.ps1` (which killed its `--type=gpu-process` on a timer) are switched off. Absolute
timings are worthless across minutes — the same binary on the same file measured 36 s and 58 s, and
a 60 s clip took 345 s while Chrome was busy and 36 s minutes later. Sharing costs ~10x, not 3x:
17.8 s alone against 183.9 s beside one headless context at nfe 16, and nfe 32 measured 2.72x and
0.33x in the same hour. A context that is only *resident*, not rendering, costs ~25 %: three
consecutive pieces of one file measured 2.68x, 2.57x and 3.22x, the two slow ones being the ones
`gpustate` logged a chrome.exe beside (13:40:03, 13:42:44).

So: read `nvidia-smi --query-compute-apps` before a timing run and note who is resident, take the
number as a ratio between two runs back to back under the same residency, and let it cool ~75 s
after sustained load. Time A/B/A, never A/B, with the same fixed rest between runs
(`testsound/lectures/ab*.sh` use 15-20 s); drift then reads as a gap between the two baselines
instead of as a speedup. `nvidia-smi -pl` and `-c` both answer Insufficient Permissions, so there is
no power cap and no exclusive compute mode; the card runs 82-83 C at ~60 W with clocks pinned
1267/2100 MHz (SW Thermal Slowdown) and still delivers ~2.5x, so heat is not worth waiting on,
contention is. Prefer ratios inside one run over absolute ms.

## Memory

6 GB card. Peak VRAM is set by `--chunk-seconds`, not file length: chunk 5 ≈ 3.2 GB, 10 ≈ 4.1,
15 ≈ 5.3, 20 OOMs; past ~5.5 GB the driver kills the process (exit 127, no message) or candle
returns `CUDA_ERROR_OUT_OF_MEMORY`. Since processed_seconds = audio_seconds * chunk /
(chunk - overlap), overlap is duplicated work: the batch's 0.25 s duplicates 5 % instead of 11 % and
is 4.5 % faster (180 s at chunk 5: 53.3 s against 55.7 s, A/B/A/B on an idle card, flatness 0.0133
vs 0.0127, no seam delta above the file's global max). 0.25 s is what was measured against 0.5 s;
0.15 s is untested rather than ruled out.

Host RAM, not VRAM, is the other limit on long files. `process_file` streams: it drops the source
after resampling, cuts each 5 s window from the resampled buffer, enhances them 24 at a time, and
hands each output to `ChunkMerger`, which keeps only the last `overlap_len` samples and pushes
everything older to the wav writer. Peak is one f32 per input sample plus one batch (~21 MB) —
~640 MB for an hour, where holding raw + resampled + every chunk output + the merged buffer was
~2 GB and got a 58 min run killed.

## Where the time goes (15 s clip, chunk 5)

CFM solver ~62 %, vocoder ~38 %, mel well under 1 %. Total ≈ 1.6x realtime cool, ~0.5x hot; on an
otherwise idle card a single 180 s file at chunk 5, nfe 16, tf32 measures 3.2-3.4x — the ceiling
the batch is measured against. Solver cost is ~0.8 s fixed plus ~2.9 ms per mel frame, so one call
covering several chunks amortises it (3 chunks per call beat per-chunk solves 9.8 s vs 13.3 s on
15 s of audio). Two knobs that looked like levers and are not: f16 for the CFM alone (50.7 s vs
46.9 s plain tf32 on 180 s — the ~30 % f16 wins over f32 does not survive tf32 already being on)
and `RESOUND_SOLVE_FRAMES=3360` vs 1680 (52.8 vs 53.1 s, i.e. ~0.8 s per call is already amortised
at three chunks per call).

nfe is the one knob that buys real time. On a 60 s clip, card free, tf32: nfe 64 1.67x, 32 2.72x,
24 2.98x, 16 3.17-3.37x (8 was not faster than 16). Quality against nfe 32 on the same clip
(per-frame log-mel cosine, mean / 1st percentile / fraction of frames below 0.9, plus band-energy
delta per decade from 0 to 22 kHz):

| nfe | cosine | p1 | frac < 0.9 | band delta | speed |
|---|---|---|---|---|---|
| 64 | 0.9609 | 0.871 | 3.0 % | within 0.17 dB | 1.0x |
| 24 | 0.9653 | — | — | within 0.39 dB (top band -0.11) | 1.10x |
| 16 | 0.9587 | 0.872 | 2.9 % | within 1.04 dB, all of it above 8 kHz | 1.24x |

nfe 16's deviation from nfe 32 is the same size and shape as nfe 32's from the upstream default of
64, and its HF noise floor over the input's 60 quietest frames is 0.7 dB *lower* than nfe 32's, so
it is not under-converged hiss — the extra ~1 dB sits in the band the model invents anyway (the
input has nothing above 8 kHz; the render is bandwidth extension). Chunk 10 is only 2 % faster than
chunk 5 — 378 s of lecture, A/B/A/B idle, 112.9 and 114.0 s against 114.8 and 117.1 s — and costs
~900 MiB, so 5 stays; an older note that chunk 10 was slower (1.60x vs 1.67x) was measured on a
shared card and does not hold.

## Verifying output

Two runs of the same config are bit-identical (fixed randn sequence). A change that alters the op
sequence or tensor shapes changes every RNG draw, so waveform correlation between two different
configs is ~0.016 and means nothing. Compare instead: per-frame log-mel cosine (0.97 between
equivalent renders, 0.78 against the input on lecture speech) and average magnitude spectrum. Seam
clicks: max adjacent-sample delta inside the overlap window must stay below the file's global max
delta. Folding the output's own RMS over the 4.5 s hop is not a seam test — speech dynamics dominate
it (the *source* folds to -6.7 dB at frame 0 where our render of a different file folds to +5.9 dB);
fold the RMS of out/src instead, which removes the content: on a 60 s batch render the overlap
window averages -5.62 dB and the rest of the period -5.24 dB, with no dip at the boundary.

`testsound/lectures/qa.py` is that check as a sweep: it decodes 60 s at 300 s from each finished
mp3 and prints flatness, corr and the loud-vs-quiet drop, aggregated per frame then the median
across frames — the way `crest.py` does it, so the numbers compare with the targets here (averaging
the spectra first weights the loud frames and reads ~2x lower). It deletes its slice before
decoding, because a decode that fails silently would otherwise be measured again under the next
file's name. Over the first 28 finished lectures: flatness 0.0157-0.0375, corr 0.634-0.889, drop
44-273 dB, nothing flagged (thresholds flat < 0.040, corr > 0.50, drop > 25 dB). The spread is the
material, not damage: the least clean render (0.0375) comes from a 0.0887 source and the cleanest
(0.0157) from a 0.0363 one, a 2.3-3.2x collapse either way.

## Quality: how "damaged" was found and fixed

Upstream ground truth lives in `C:\dev\refenv` (Python 3.12, torch+cpu, `resemble-enhance` 0.0.1,
HF repo at `C:\dev\refenv\Lib\site-packages\resemble_enhance\model_repo`). Run it with
`testsound/lectures/probe.py src.wav out.wav lambd nfe` (needs the `pathlib.PosixPath =
pathlib.WindowsPath` shim); it monkeypatches `UnivNet.forward` and `IRMAE.encode` to dump the
conditioning mel (`out.wav.cond.f32`, 160 ch) and the latent, which is what makes module-level
comparison possible. `testsound/lectures/crest.py` (crest + flatness) and `hf.py` (per-band level
split into loud vs quietest 5 % of frames, plus corr(log 300-3k, log 8-16k)) are the metrics.

A damaged render is not quieter or duller, it is *noisy*. The metric that caught it is spectral
flatness over 100-4000 Hz (geometric/arithmetic mean of the power spectrum) plus the loud-vs-quiet
band split: real enhancement gates HF with speech, hiss does not. On 60 s of lecture audio, source
0.0439 -> render 0.0153 flatness, crest 14.35 -> 15.66 dB, 8-16 kHz from -69 dB (absent) to -29 dB
loud but -58 dB quiet, corr(log 300-3k, log 8-16k) from 0.049 to 0.732 — upstream renders
0.0207-0.0211 on the same material, so ~0.02 is the target and damage reads as ~0.19.

Localising it took dumping our own tensors and diffing against upstream's: `RESOUND_DUMP_MEL=<dir>`
writes the conditioning mel and the decoded 160 ch; `RESOUND_COND_FILE=<f32>` feeds a 160-ch cond
straight to the vocoder, skipping mel/AE/CFM. That hook settled it — with upstream's own cond mel
our vocoder still produced flatness 0.1927, so mel->IRMAE->CFM->decode was fine (cond statistics
match to 3 digits: std 1.244 vs 1.250, jitter 0.3217 vs 0.3215, cos(cond128, mel) 0.5564 vs 0.5575)
and the vocoder was not.

Two vocoder bugs, both invisible structurally:

1. `UpSample1d` scaled its Kaiser filter by `ratio²` where upstream (`univnet/alias_free_torch/
resample.py`) applies `ratio` once, as a gain on the transposed-conv output. Every AMP branch in
every LVC block ran 2x too hot, and `SnakeBeta` is nonlinear so it did not cancel.
2. `KernelPredictor`'s trunk used LeakyReLU slope 0.1. `KernelPredictor.__init__` defaults to 0.1,
but `LVCBlock` passes `lReLU_slope=0.2`, so 0.2 is what runs. This was the whole residual gap:
same cond, before/after/upstream 8-16 kHz loud -32.4 / -24.7 / -24.7 dB, quiet -58.6 / -72.7 /
-72.1 dB, 16-22 kHz flatness 0.151 / 0.088 / 0.088. `conv_pre`/`conv_post` also had to become
reflect-padded (upstream `padding_mode="reflect"`). Our vocoder now matches upstream's to rms
1.6e-4, corr 0.999999, on identical cond and noise.

Port errors of that shape are invisible in code review and only show up as audio quality, so diff
constants against upstream source, not just structure. Two more found that way: the Kaiser rolloff
denominator (upstream is `torch.kaiser_window(K, periodic=False)`, i.e. divide by `(K-1)/2`, not
`K/2`) and the time-embedding grid (`linspace(0, 4, 64)` has step `4/63`, not `4/64`). The `filter`
buffers *are* in the checkpoint (24), so the Kaiser code is only a fallback.

Whole pipeline against upstream on 60 s of lecture (nfe 32 / upstream nfe 32): crest 15.32 / 15.31
dB, flatness 0.0205 / 0.0197, 8-16 kHz loud -19.50 / -18.58 dB and quiet -68.79 / -68.12 dB, corr
0.811 / 0.819; the batch's nfe 16 is a hair milder (0.0217) and still 3.2x. Second lecture (`Love
Series 3B`, source 0.0383): upstream nfe 32 renders 0.0282, our nfe 16 0.0294, 8-16 kHz loud -24.8 /
-25.0 dB, corr 0.787 / 0.736 — flatness near 0.029 is that material, not a defect. Ours gates pauses
harder (300-3k quiet -73.4 vs -64.4 dB, 8-16k quiet -75.4 vs -68.6 dB), the nfe-16 direction.

`z_scale` is 6, not the upstream dataclass default of 5: `model_repo/enhancer_stage2/hparams.yaml`
says `lcfm_z_scale: 6` and `HParams.load` reads the yaml. Still missing versus upstream: the
denoiser stage (`lambd` in `enhance()`), a separate model we do not ship, and `vocoder(npad=10)`
(pad cond 10 frames, trim 4200 samples) — equivalent, both yield `T * 420` samples. We also do not
drop the last mel frame the way `to_mel(drop_last=True)` does; empirically both give the same frame
count.

## Batch

`testsound/lectures/run_batch.sh` enhances the 588 lecture mp3s (447.8 h) from
`C:\D\Downloads\Manly_P.Hall_Digitally_Restored_Audio_Lectures` into
`C:\D\Downloads\Manly_P.Hall_Enhanced` as 192 kbps mono mp3 (~39 GB, vs 89 GB flac / 142 GB wav);
`D:\temp_resound` is scratch. Config `GEMM=tf32 NFE=16 OVERLAP=0.25`, chunk and piece length per
file from `pick_chunk()`: chunk 5 with 450 s pieces above `CHUNK_FREE_MIB=4000` MiB free, else
chunk 3 with 250 s, and chunk 3 (2 under a 3) is the retry. Chunk 10 was tried as a third tier
above 5000 MiB and reverted: 2 % faster but 234 s of audio per flush batch instead of 114 s, so it
doubles the window in which a death loses the whole piece, and it died 38 s into its first
production piece. Taking the bigger window when it fits is worth it: in a 45 min window chunk 5
rendered ~4000 s in 340/349/401/405 s (2.67x) against chunk 3's 390/486/519 s for ~2755 s (1.97x),
and duplicates 11 % instead of 20 %. The batch therefore does not dodge Chrome: chunk 3 is the only
size that fits beside it and measures 1.97x, worse than 2.57x, so run throughput ranges 2.6-3.3x
depending on who else is resident. Deaths scale with process starts as well as time under load —
225 s pieces lost 11.5 per audio-hour, chunk 3 at 450 s 6.5, chunk 5 at 450 s 4.5 — which is why
the 225 s experiment was reverted. Pieces measure 3.26-3.38x with the card to themselves; files land
2.35-3.0x end to end (three consecutive: 2.68x, 2.52x, 2.35x = 3580 s in 1429 s, a window that held
two deaths, a resident Chrome for two of nine pieces and a CPU-heavy sweep beside it; a file with
none of those is 3.0x, 371 s for ~1122 s). So 6-7.5 days of enhancing for 447.8 h, plus ~20 h of
decode/encode. The 192 kbps encode is transparent: the same 60 s slice measures 0.0290 from a batch
mp3 and 0.0294 from a wav render of the same source.

Ordering is shortest first (from `durations.tsv`), so the first files prove decode, enhance, piece
join and encode end to end in minutes. Each file is cut into equal pieces (0.25 s overlap) and
rejoined with `join.py`, which streams and keeps only the overlap tail. Each file: decode to 44.1 kHz
mono wav, optional rnnoise prefilter (see below), enhance, encode to `$dest.part`, rename. An
existing non-empty output means done, so the run is resumable. Things that bit once and are now
guarded:

- `ffmpeg` reads stdin and will eat the file list out from under a `while read` loop, so every call
  is `-nostdin </dev/null` (`ffprobe` has no `-nostdin`, so it only gets `</dev/null`), and the
  encode needs an explicit `-f mp3`: a `.part` suffix hides the extension the muxer is guessed from.
- An empty `ffprobe` duration silently collapses the plan to one piece — one 1.2 GB enhance on a long
  file — so the plan is validated and the file skipped with `FAIL_PLAN`.
- Stopping a background task kills the wrapper, not the script, and two live instances share one temp
  dir and corrupt each other's pieces; the script takes a `$TMP/.lock` pid lock. Never edit
  `run_batch.sh` while a run is live: bash holds a byte offset into the script, so a rewrite of a
  different length gets re-read from the wrong place. Stop the whole `bash.exe` tree (not just the
  wrapper), `rm -rf $TMP/.lock`, edit, restart.
- A silent resound death is the GPU falling over, not contention: rc=127, no message, with an
  nvlddmkm Event ID 153 ("Error occurred on GPUID: 100") beside it in the System log (12 in 3 h,
  09:10:13 second-exact with a death timed from the log). Query by `ProviderName`, not `Message` —
  the message is null there. `TdrDelay`/`TdrLevel` are unset, so WDDM's 2 s default applies. It is
  not VRAM and not Chrome: it happens at chunk 3 (~2.3 GB) with nothing else on the card, and
  killing Chrome's gpu-process at creation did not stop it. The rate is what varies — roughly one per
  10-15 min of load, about one per file, arriving in bursts — so it is not worth preventing, only
  worth making cheap.
  The cost of one death is the wasted render plus a re-render plus whatever wait precedes the retry.
  Nothing reaches the wav until every chunk of a batch is rendered (killing a 180 s render 42 s in
  left 5025770 samples, exactly 24 chunks' worth of 4.75 s hop), so `CHUNKS_PER_BATCH` is the knob on
  the wasted part: 24 risks 114 s of audio, 8 risks 38 s. Not worth spending: 8 measured 52.96/53.30 s
  against 48.59/52.84 s for 24 on 180 s, but the card drifted 9 % across those four runs. So on
  failure `resume_piece` goes first: hound writes the header at creation and only rewrites the sizes
  on finalize, so a killed run leaves a wav whose header claims 0 frames while every flushed sample
  is there — `wavefix.py` prints the true length off the file size, the remainder is cut from
  `(n - OV) / 44100` s and rendered, and `join.py` crossfades it on with the same 0.25 s window.
  Measured on 120 s: resumed 120.00 s against a clean 120.00 s, seam max delta 0.049 against a
  global 0.352, audio before the seam bit-identical. Production: a 312 s piece died 60 s in, its
  252.58 s remainder rendered at 2.90x and joined to 312.08 s against the sibling's 312.05 s, seam
  max delta 0.0396 against a global 0.2917.
  A resume render can die too and leaves its own partial in `resume.enhance.wav`, which a retry used
  to delete — one death threw away 205 s of rendered audio that way. `salvage_resume` joins it onto
  the piece before retrying, so up to three attempts accumulate progress and only the last resort
  re-renders the whole piece at chunk 2.
  Waiting used to make a death expensive and was unconditional: `settle()` waited for two quiet
  minutes in the event log before every piece. `ready()` asks the card instead — one 1 s render at
  nfe 8 over `probe1s.wav`, ~4 s — and settles only when that fails, so a healthy card costs 4 s a
  piece instead of 126 s. Bursts are still real (six events in four minutes) and a retry landing
  inside one loses the file rather than the piece (`Love Series 1B - Human Love` did, and recovered
  on the next run), so the wait stays, behind the probe.
- Pieces are 450 s because a longer one dies: candle has no pooling allocator, so every chunk is
  thousands of cudaMalloc/cudaFree cycles and a chunk-5 run over a whole 30 min piece dies past
  ~100 chunks, while the same audio in 450 s pieces does not. The gate before a piece is under 85 C
  and under 1.5 GB used, capped at 12 polls x 20 s; it used to be 78 C, which bought minutes of idle
  per piece for nothing — at 83 C and 1267 MHz a 539 s file still ran 2.57x.
- A run that prints `device=Cpu` has lost CUDA and is ~1000x slower while looking healthy (once, when
  the machine suspended mid-run); `run_enhance` checks every piece for that line and stops the batch.
  `stayawake.ps1` holds an `ES_CONTINUOUS|ES_SYSTEM_REQUIRED|ES_DISPLAY_REQUIRED` request for the
  life of the run because `powercfg /change` and `/setacvalueindex` both answer Invalid Parameters
  here; the flags are written as the decimal 2147483651, since PowerShell parses `0x80000003` as a
  negative int.
- Start it detached: `start_batch.bat` launched with `Start-Process`, since a `run_in_background`
  task dies with the session. `Start-Process -ArgumentList '-c','cmd'` does not quote the second
  element, so bash sees `-c cmd` and runs the first word — put the whole command in the .bat.
  `start_watch.bat`, `start_guard.bat` and `start_event.bat` launch the Chrome killers; do not start
  them.
- The encode is pure CPU, the enhance GPU, so one file's encode no longer sits in front of the next:
  `$TMP/in.enhance.wav` is renamed to `enc.wav` and handed to ffmpeg in the background, and
  `finish_encode()` settles the `.part` rename a whole file later (and once more after the loop) —
  8-18 s per file the card used to idle through; settling at the top of the next iteration only moves
  the wait (`WAIT_encode` read 19 s both ways). Reading the log: the `OK wall_s=` line for a file is
  written when the *next* file's pieces are done, so completions appear one file behind `PLAN`.

## Chunk seams

Each chunk's vocoder output is a fresh realization: the noise the vocoder draws per chunk decides
the waveform phase, so two renditions of the same audio correlate at ~0.1 at sample level even
though their envelopes track at 0.78-0.85. That killed the old seam aligner, which cross-correlated
the previous chunk's tail against the next chunk's head: with no real lag to find it picked noise
peaks, usually next to the ±2048-sample search edge, then shifted the chunk by up to 46 ms — random
time offsets, audio dropped or duplicated at every seam, and a file that ran out of room before the
last chunk. `ChunkMerger` now places every chunk at its nominal offset (`placed() - overlap_len`)
and only crossfades, which is exact by construction: the vocoder hop is the mel hop (420), so
output frame t belongs at sample t * 420.

There is no delay to compensate: measured three ways on three input/render pairs (log-mel shift at
9.5 ms per frame, RMS-envelope lag, onset-flux lag), the render lines up with its own input to
within one mel frame. The old 110-140 ms figure came from a noisy source, where removing the noise
leaves the correlation weak and its peak a broad hump rather than a point — the same measurement on
cleaner material peaks sharply at 0. Nothing of the source goes unrendered: output length equals
input length and the last frames carry signal.

## rnnoise prefilter

`run_batch.sh` feeds resound rnnoise's output instead of the source, `PREFILTER=1` by default:

```
ffmpeg -af "arnndn=m=testsound/lectures/models/bd.rnnn:mix=0.618,aresample=44100"
```

i.e. 61.8 % denoised with 38.2 % of the original left in. `arnndn`'s own `mix` does that blend in
one filter (mix=1 denoised, mix=0 untouched) and it has to be that filter rather than a hand-built
`amix`: rnnoise's output runs 448 samples (10.2 ms) behind its input, so blending the two
externally combs them — 3 dB down with the flatness halved. `arnndn` also outputs 48 kHz whatever
it is fed, so the `aresample` back to 44100 is not optional. The model is `bd`, *beguiling-drafter*
from `GregorR/rnnoise-models` (voice over recording noise), from `testsound/lectures/models/`; not
`sh`, the speech/recording model the names point at: on a noisy 60 s slice `sh` ducked the whole
signal 10.8 dB, 9.8 dB of it in the speech band, where `bd` took 23 dB out of the pauses and left
the voice 1.4 dB down. The model path must stay relative: a `:` in a filter option value ends the
option list, and MSYS mangles `/d/...` into `C:\Program Files\Git\...`.

Level has to be put back. rnnoise removes noise and some voice with it — 0.4 dB on the cleaner
slice, 1.3 dB on the noisier — and resound's output level follows its input, so the render would
come out that much quieter for no reason. The script reads both `mean_volume`s with `volumedetect`
(prints at `-v info`), clamps the gain so the peak stays under -1 dBFS and to ±12 dB, and applies it
at the encode. With the level matched, on two 60 s lecture slices at nfe 16 / chunk 5:

| | flatness | pauses (300-3k quiet) | 16-22 kHz quiet | loud 0.3-1k / 1-3k / 3-6k / 8-16k |
|---|---|---|---|---|
| A plain | 0.0202 | -63.29 dB | -73.72 dB | +25.84 / +8.83 / -3.37 / -26.13 |
| A prefiltered | 0.0203 | -62.41 dB | -73.70 dB | +25.81 / +9.21 / -3.66 / -25.53 |
| B plain | 0.0270 | -59.59 dB | -71.54 dB | +26.46 / +17.20 / -1.84 / -16.08 |
| B prefiltered | 0.0247 | -60.78 dB | -72.37 dB | +26.40 / +17.41 / -1.67 / -17.27 |

so the voice lands within 0.2-0.4 dB of the plain render in every speech band and what moves is the
part that was noise: flatness on the noisier slice (0.0270 -> 0.0247) and the pause floor. Above
8 kHz it costs ~1 dB, bandwidth the model was inventing anyway. End to end on a real 447.66 s
lecture: prefilter 5 s, render 133.78 s (3.35x), encode 6 s, `OK wall_s=149`, and `qa.py` on the
result reads flat=0.0271 corr=+0.901 (passes) against the source's 0.0355 / +0.193. Listening A/B
for the four variants on both slices: `C:\D\Downloads\resound_ab_prefilter\`.

The gate is last (`agate=threshold=0.00316:ratio=4:range=0.1:attack=20:release=200:detection=rms`)
and on our own renders it is a no-op: the render's pauses already sit at -87 dBFS (20 ms frame RMS,
p10), so a -50 dBFS threshold only zeroes what is inaudible and file RMS and loud frames are
unchanged to the last digit. It is in the chain because it is what makes the *other* reading of the
61.8 % listenable — see below.

Mixing the denoised copy into the *output* instead of the input measures worse, and the reason is
worth keeping: the render is a fresh realization (sample-correlation 0.008-0.018 with its own
input), so a 61.8/38.2 mix of the two is a doubled voice, not a blend — they add incoherently
(power sum, not amplitude sum), flatness climbs back to the source's (0.0328 / 0.0438 against
0.0202 / 0.0270), 8-16 kHz drops 7 dB and the level 4-13 dB. Gating recovers its pauses (-33 dB to
-62/-76) but not the doubling.

Cost: `arnndn` does 378 s of audio in 6 s, and the batch logged `WAIT_prefilter s=5` on a 447 s
lecture — roughly 3 % of a file's render.

## Model geometry

mel 128 ch, hop 420; AE latent 64; CFM WaveNet 30 layers, dilation cycle 5, hidden 512, nfe 64 gives
32 solver steps = 64 network evals; UnivNet cond 160, noise 128, channels 96, strides [7,5,4,3],
dilations [1,3,9,27].

## candle constraints

- No pooling allocator: every tensor is a `cudaMalloc`/`cudaFree`. Big short-lived buffers cost real
  time, not just bytes.
- `matmul` needs equal ranks; 2D @ 3D needs `broadcast_matmul`, which concretizes the broadcast (a
  copy). No beta-accumulate gemm is exposed.
- `set_gemm_reduced_precision_f32(true)` is TF32. f16 helped the CFM ~30 % and hurt the vocoder, so
  the default stays f32.

## Notes moved out of code

- `FastConv1d::im2col` stacks the dilated taps as `[channel, tap, frame]` so the flattened rows match
  the weight's own `(c_out, c_in * ksize)` layout and neither side needs a copy.
- `IM2COL_TILE_ELEMENTS` is the scratch budget for one tile; a tile is the widest run of frames whose
  `c_in * ksize * frames` fits. Tiling multiplies the op count, so the budget trades speed for memory
  directly.
- `FastConvTranspose1d` is one gemm per output phase: output sample `u * stride + r` collects every
  tap whose `j - padding` lands on phase `r`, and the phases are interleaved by a reshape instead of
  a scatter.
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
