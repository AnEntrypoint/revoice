# resound

Rust port of the resemble-enhance inference pipeline (candle 0.10.2, CUDA). `resound enhance` /
`resound denoise` over wav files; long files are split into chunks and crossfaded back.

## Run

```
cargo build --release --bin resound
./target/release/resound.exe enhance --out-dir out --chunk-seconds 5 --overlap-seconds 0.5 in.wav...
```

Resamples to 44.1 kHz mono; a failing file (OOM) is skipped. Run from
`C:\dev\resound` (relative `weights/`) with `D:/...` paths — `/d/...` and a foreign
cwd both fail.

| env | meaning |
|---|---|
| `RESOUND_GEMM` | `f32` (default) / `tf32` / `f16` gemm precision |
| `RESOUND_SOLVE_FRAMES` | mel frames per CFM solver call (default 1680) |
| `RESOUND_PROFILE` | `1` stage ms, `2` +gpu tick, `3` +per-layer tick |
| `RESOUND_WARMUP_MB` | preallocate this much VRAM before starting (default 0) |
| `RESOUND_DUMP_MEL` | write the conditioning mel and the decoded 160 ch to this dir as `.f32` |
| `RESOUND_COND_FILE` | skip mel/AE/CFM and vocode this `1x160xT` little-endian f32 instead |

## Measuring on this GPU

Shared 6 GB card; Chrome holds ~2.2 GB and can sit at 100 % util while nothing else runs: **never kill
Chrome** (`gpuwatch*.sh` and `gpuwatch_event.ps1` are off).

- Absolute ms are worthless across minutes: same binary and file, 36 s and 58 s; a 60 s clip 345 s
  with Chrome busy, 36 s later. Sharing costs ~10x, not 3x (17.8 s alone vs 183.9 s beside one
  headless context at nfe 16); a merely *resident* context ~25 % (pieces of one file: 2.68x, 2.57x,
  3.22x; the slow two had a chrome.exe beside).
- Method: `nvidia-smi --query-compute-apps` first, note who is resident, take ratios between two runs
  back to back under the same residency, cool ~75 s after load, A/B/A never A/B with a fixed rest, so
  drift reads as a gap between baselines.
- Heat is not worth waiting on: 82-83 C, 1267 MHz (SW Thermal Slowdown), still ~2.5x. `nvidia-smi -pl`
  / `-c` answer Insufficient Permissions: no power cap, no exclusive compute mode.

## Memory

Peak VRAM is set by `--chunk-seconds`, not file length; past ~5.5 GB the driver kills the process
(exit 127, no message) or candle returns `CUDA_ERROR_OUT_OF_MEMORY`.

| chunk s | 3 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| VRAM | ~2.3 GB | ~3.2 GB | ~4.1 GB | ~5.3 GB | OOM |

Chunk 3 is the only size that fits beside Chrome. processed = audio *
chunk / (chunk - overlap), so overlap is duplicated work: 0.25 s duplicates 5 % against 0.5 s's 11 %
and is 4.5 % faster (180 s at chunk 5, A/B/A/B idle: 53.3 vs 55.7 s; flatness 0.0133 vs 0.0127; no
seam delta above the file's global max). 0.15 s is untested, not ruled out.

Host RAM is the other limit. `process_file` streams (drops the source after resampling,
cuts each window from the resampled buffer, enhances 24 at a time) and `ChunkMerger` keeps only the
last `overlap_len` samples: peak = one f32 per input sample + one batch (~21 MB) ≈ 640 MB per hour,
against ~2 GB (and a 58 min run killed) for holding everything.

## Where the time goes

CFM solver ~62 %, vocoder ~38 %, mel <1 %. ≈1.6x realtime cool, ~0.5x hot; a 180 s file at chunk 5 /
nfe 16 / tf32 on an idle card is 3.2-3.4x — the ceiling the batch is measured against. The solver is
~0.8 s fixed + ~2.9 ms per mel frame, so one call covering several chunks amortises it (3 chunks per
call: 9.8 s vs 13.3 s per-chunk on 15 s).

nfe is the only knob that buys real time (60 s clip, card free, tf32):

| nfe | speed | cosine vs 32 | p1 | frac<0.9 | band delta vs 32 |
|---|---|---|---|---|---|
| 64 | 1.67x | 0.9609 | 0.871 | 3.0 % | within 0.17 dB |
| 32 | 2.72x | — | — | — | — |
| 24 | 2.98x | 0.9653 | — | — | within 0.39 dB (top band -0.11) |
| 16 | 3.17-3.37x | 0.9587 | 0.872 | 2.9 % | within 1.04 dB, all above 8 kHz |

cosine = per-frame log-mel; band delta per decade 0-22 kHz; nfe 8 was not faster than 16. nfe 16's
deviation from 32 is the same size and shape as 32's from upstream's default of 64, and its HF noise
floor over the input's 60 quietest frames is 0.7 dB *lower* than 32's — not under-converged hiss; the
extra ~1 dB sits in the band the model invents anyway (input has nothing above 8 kHz).

Measured and discarded: f16 for the CFM alone (50.7 s vs 46.9 s plain tf32 on 180 s — the ~30 % f16
win over f32 does not survive tf32 being on); `RESOUND_SOLVE_FRAMES=3360` vs 1680 (52.8 vs 53.1 s);
chunk 10 vs 5 (378 s, A/B/A/B idle: 112.9 and 114.0 s vs 114.8 and 117.1 s, at ~900 MiB more).

## Verifying output

Two runs of one config are bit-identical (fixed randn). A change to the op sequence or tensor shapes
changes every RNG draw, so sample correlation between two configs (~0.016) means nothing. Use
per-frame log-mel cosine (0.97 between equivalent renders, 0.78 against the input on lecture speech).

Seams: max adjacent-sample delta inside the overlap window must stay below the file's global max
delta. Folding the output's own RMS over the 4.5 s hop is not a seam test (content dominates: the
*source* folds to -6.7 dB at frame 0); fold RMS of out/src instead — a 60 s batch render averages
-5.62 dB in the overlap window against -5.24 dB over the rest of the period, no dip at the boundary.

`testsound/lectures/qa.py` decodes 60 s at 300 s from each finished mp3 and prints flatness (100-4000
Hz, geometric/arithmetic mean of the power spectrum), corr(log 300-3k, log 8-16k) and the loud-vs-
quiet drop, per frame then the median across frames — `crest.py`'s aggregation, so the numbers compare
with the targets here (averaging spectra first reads ~2x lower). Thresholds flat < 0.040, corr > 0.50,
drop > 25 dB. First 28 finished lectures: flatness 0.0157-0.0375, corr 0.634-0.889, drop 44-273 dB,
nothing flagged; the spread is the material (0.0375 from a 0.0887 source, 0.0157 from a 0.0363 one).
**A 200-300 dB drop is material plus the gate, not damage** — the gate drives already-inaudible pauses
to the 16-bit floor or to exact zero, and both pass.

## Upstream comparison and port traps

Ground truth in `C:\dev\refenv` (Python 3.12, torch+cpu, resemble-enhance 0.0.1; HF repo at
`C:\dev\refenv\Lib\site-packages\resemble_enhance\model_repo`), driven by
`testsound/lectures/probe.py src.wav out.wav lambd nfe` (needs a `pathlib.PosixPath =
pathlib.WindowsPath` shim); it monkeypatches `UnivNet.forward` and `IRMAE.encode` to dump the
conditioning mel (`out.wav.cond.f32`, 160 ch) and the latent, which is what makes module-level
comparison possible. `crest.py` gives crest + flatness, `hf.py` the loud-vs-quietest-5 % band split
plus corr(log 300-3k, log 8-16k).

Damage is *noise*, not dullness — flatness over 100-4000 Hz plus the loud-vs-quiet split catch it,
because real enhancement gates HF with speech and hiss does not. On 60 s of lecture: source 0.0439 ->
render 0.0153 flatness, crest 14.35 -> 15.66 dB, 8-16 kHz -69 dB (absent) -> -29 dB loud but -58 dB
quiet, corr 0.049 -> 0.732; upstream renders 0.0207-0.0211 there, so ~0.02 is the target and damage
~0.19. With upstream's own cond mel our vocoder still gave 0.1927, so mel->IRMAE->CFM->decode was fine
(cond stats match to 3 digits: std 1.244 vs 1.250, jitter 0.3217 vs 0.3215, cos(cond128, mel) 0.5564 vs
0.5575) and the vocoder was not. Two vocoder bugs, both invisible structurally:

1. `UpSample1d` scaled its Kaiser filter by `ratio²` where upstream (`univnet/alias_free_torch/
resample.py`) applies `ratio` once, as a gain on the transposed-conv output. Every AMP branch in every
LVC block ran 2x too hot through a nonlinear `SnakeBeta`.
2. `KernelPredictor`'s trunk used LeakyReLU 0.1; `LVCBlock` passes `lReLU_slope=0.2`, so 0.2 runs —
the whole residual gap. Same cond, before/after/upstream 8-16 kHz loud -32.4 / -24.7 / -24.7 dB,
quiet -58.6 / -72.7 / -72.1 dB, 16-22 kHz flatness 0.151 / 0.088 / 0.088. `conv_pre` / `conv_post`
also needed `padding_mode="reflect"`. Our vocoder now matches upstream's to rms 1.6e-4, corr 0.999999,
on identical cond and noise.

Diff constants against upstream source, not just structure:

| constant | upstream truth | trap |
|---|---|---|
| Kaiser rolloff denominator | `torch.kaiser_window(K, periodic=False)` | `(K-1)/2`, not `K/2` |
| time-embedding grid | `linspace(0, 4, 64)` | step `4/63`, not `4/64` |
| `z_scale` | `model_repo/enhancer_stage2/hparams.yaml` `lcfm_z_scale: 6`, read by `HParams.load` | not the dataclass default of 5 |
| KernelPredictor trunk slope | `LVCBlock` passes `kpnet_nonlinear_activation_params={"negative_slope": lReLU_slope}` = 0.2 | costs 8 dB HF, 13 dB noise floor |

The `filter` buffers *are* in the checkpoint (24), so the Kaiser code is only a fallback. Missing vs
upstream: the denoiser stage (`lambd` in `enhance()`), a model we do not ship, and `vocoder(npad=10)`
— equivalent, both give `T * 420` samples. We also skip `to_mel(drop_last=True)`'s last-frame drop and
get the same frame count.

Whole pipeline vs upstream on 60 s of lecture (ours nfe 32 / upstream nfe 32): crest 15.32 / 15.31 dB,
flatness 0.0205 / 0.0197, 8-16 kHz loud -19.50 / -18.58 dB and quiet -68.79 / -68.12 dB, corr 0.811 /
0.819. Second lecture (`Love Series 3B`, source 0.0383): upstream nfe 32 0.0282, our nfe 16 0.0294,
8-16 kHz loud -24.8 / -25.0 dB, corr 0.787 / 0.736 — ~0.029 is that material, not a defect. Ours gates
pauses harder (300-3k quiet -73.4 vs -64.4 dB, 8-16k quiet -75.4 vs -68.6 dB), the nfe-16 direction;
the batch's nfe 16 is a hair milder (0.0217) and still 3.2x.

## Batch

`testsound/lectures/run_batch.sh`: 588 lecture mp3s (447.8 h) from
`C:\D\Downloads\Manly_P.Hall_Digitally_Restored_Audio_Lectures` into `D:\Manly_P.Hall_Enhanced` as
192 kbps mono mp3 (~39 GB vs 89 GB flac / 142 GB wav); `D:\temp_resound` is scratch; on D: because C:
sits at 99 % with ~36 GB free.

**Live config: `GEMM=tf32 NFE=16 OVERLAP=0.25`, prefilter on, gate at 38.2 %; 35 files delivered.**
A stop costs only the in-flight piece.

Per file: decode to 44.1 kHz mono, rnnoise prefilter, enhance, encode to `$dest.part`, rename; a
non-empty output means done, so the run is resumable. Shortest first (`durations.tsv`). Equal pieces
(0.25 s overlap) rejoined by `join.py` (streams, keeps the overlap tail). `pick_chunk()`: chunk 5 with
450 s pieces above `CHUNK_FREE_MIB=4000` MiB free, else chunk 3 with 250 s; chunk 3 (2 under a 3) is
the retry. 450 s pieces because longer ones die: no pooling allocator means thousands of
cudaMalloc/cudaFree per chunk, and a chunk-5 run over a whole 30 min piece dies past ~100 chunks.
Pieces with the card to themselves run 3.26-3.38x.

| window | chunk 5, 450 s pieces | chunk 3, 450 s pieces |
|---|---|---|
| 45 min | ~4000 s of audio in 340/349/401/405 s (2.67x) | ~2755 s in 390/486/519 s (1.97x) |
| end to end | 2.35-3.0x: three consecutive 2.68x, 2.52x, 2.35x = 3580 s in 1429 s; a file with nothing else in its window is 3.0x | |

Deaths per audio-hour: 11.5 at 225 s pieces, 6.5 at chunk 3 / 450 s, 4.5 at chunk 5 / 450 s — they
scale with process starts as well as time under load, why the 225 s experiment was reverted. So 6-7.5
days for 447.8 h plus ~20 h of decode/encode. The 192 kbps encode is transparent: one 60 s slice
measures 0.0290 from a batch mp3 and 0.0294 from a wav render.

Gotchas, each of which has bitten:

- `ffmpeg` reads stdin and eats the file list from under a `while read` loop: every call is
  `-nostdin </dev/null` (`ffprobe` has no `-nostdin`, so it only gets `</dev/null`). The encode needs
  `-f mp3`: a `.part` suffix hides the extension the muxer is guessed from.
- **Apostrophes in paths**: MSYS will not convert a `/c/...` path to Windows form for a native exe when
  the path contains `'`, so ffmpeg answers "No such file or directory" for a file `ls` plainly sees.
  18 of 588 names have one (`Esoteric, Metaphysical 18B - Pandora's Box - the Mystery of Memory`).
  Fixed: `SRC` is now the Windows form `C:/D/Downloads/Manly_P.Hall_Digitally_Restored_Audio_Lectures`,
  which its three uses (`FILES`, `[ -f "$f" ]`, `ff -i "$f"`) all take. Commas and ampersands are fine;
  only `'` breaks it.
- An empty `ffprobe` duration collapses the plan to one piece (one 1.2 GB enhance), so the plan is
  validated and the file skipped with `FAIL_PLAN`.
- **Never edit `run_batch.sh` while a run is live**: bash holds a byte offset into the script, so a
  rewrite of a different length is re-read from the wrong place. Stop the whole `bash.exe` tree (not
  just the wrapper), `rm -rf $TMP/.lock`, edit, restart.
- Stopping a background task kills the wrapper, not the script; two live instances share one temp dir
  and corrupt each other's pieces, so the script takes a `$TMP/.lock` pid lock.
- Start detached: `start_batch.bat` with `Start-Process` (a `run_in_background` task dies with the
  session). `Start-Process -ArgumentList '-c','cmd'` does not quote the second element, so bash sees
  `-c cmd` — put the whole command in the .bat. `start_watch.bat`, `start_guard.bat`, `start_event.bat`
  launch the Chrome killers; do not start them.
- `device=Cpu` means CUDA was lost and it is ~1000x slower while looking healthy (once, when the
  machine suspended mid-run); `run_enhance` checks every piece and stops the batch. `stayawake.ps1`
  holds `ES_CONTINUOUS|ES_SYSTEM_REQUIRED|ES_DISPLAY_REQUIRED` for the run, because `powercfg /change`
  and `/setacvalueindex` answer Invalid Parameters; flags are decimal 2147483651, since PowerShell
  parses `0x80000003` as a negative int.
- The encode is CPU, the enhance GPU: a file's encode runs in the background and `finish_encode()`
  settles the `.part` a whole file later (and once more after the loop) — 8-18 s per file the card
  used to idle through; settling at the top of the next iteration only moves the wait (`WAIT_encode`
  read 19 s both ways). So a file's `OK wall_s=` is written when the *next* file's pieces are done:
  completions appear one file behind `PLAN`.
- **FAIL_QUIET**: after the join the render's mean volume is compared with the decoded source's after
  the prefilter's gain correction; `render + gain < source - 6 dB` withholds the file — logs
  `FAIL_QUIET base src=..dB render=..dB gain=..dB`, appends the base to `skip_quiet.tsv`, delivers
  nothing. Delivered files sit within -0.9..+2.2 dB of their source; what fails is music — two of 588,
  both withheld: `Art and Aesthetics 4A - Music of Comte de St. Germain` (14 dB down, 94 % of samples
  below -60 dBFS, silencedetect flags 1489 s of 1590 s) and `Art and Aesthetics 3A - Music of Comte de
  St. Germain` (`src=-24.8dB render=-41.5dB gain=4.00dB`, 16.7 dB under before its correction). Both do
  it with and without the prefilter, so it is the model collapsing on music, not the chain; a rerun
  reproduces it, which is what the skiplist is for. Checked live: flags both (-14.4, -10.1 dB), passes
  three known-good files (+1.5, +0.5, +0.7 dB).
- **bash itself can die** with a Cygwin fork failure (`child_copy: user heap read copy failed ...
  Win32 error 299`, `*** fatal error in forked process - WFSO timed out after longjmp`,
  `bash.exe.stackdump`): MSYS instability, not our script. It killed a run mid-file with no
  `BATCH_DONE` and nothing to restart it, so a run can stop with no `FAIL_*` and its only trace in
  `testsound/lectures/batch.out` (stderr), not `batch.log`. It leaves `$TMP/.lock` behind, harmlessly —
  the check is `kill -0` on the pid. `start_batch.bat` now records the log's line count and restarts
  (30 s, up to 50x) if no `BATCH_DONE` appears in the lines that run appended; scoping matters, the log
  holds every earlier run's `BATCH_DONE`.

### Making a GPU death cheap

A silent death is the GPU falling over, not contention: rc=127, no message, with an nvlddmkm Event ID
153 beside it in the System log (12 in 3 h). Query by `ProviderName`, not `Message` — the message is
null. `TdrDelay`/`TdrLevel` unset, so WDDM's 2 s default applies. Not VRAM, not Chrome: it happens at
chunk 3 (~2.3 GB) with nothing else on the card, and killing Chrome's gpu-process did not stop it. The
rate is what varies — ~one per 10-15 min of load, ~one per file, in bursts (six events in four
minutes) — so prevent nothing, just make it cheap.

- Nothing reaches the wav until every chunk of a batch is rendered (a 180 s render killed 42 s in
  left 5025770 samples = 24 chunks of 4.75 s hop), so `CHUNKS_PER_BATCH` sets the waste: 24
  risks 114 s of audio, 8 risks 38 s. Not worth spending — 8 measured 52.96/53.30 s against
  48.59/52.84 s for 24 on 180 s, but the card drifted 9 % across them. So `resume_piece` goes
  first: hound writes the header at creation and the sizes only on finalize, so a killed run leaves a
  wav claiming 0 frames with every flushed sample present — `wavefix.py` prints the true length off
  the file size, the remainder is cut from `(n - OV) / 44100` s and rendered, and `join.py` crossfades
  it on over the same 0.25 s. Production: a 312 s piece died 60 s in, its 252.58 s remainder rendered
  at 2.90x, joined to 312.08 s against the sibling's 312.05 s, seam max delta 0.0396 against a global
  0.2917.
- A resume render can die too and leaves its own partial in `resume.enhance.wav`, which a retry used
  to delete — one death threw away 205 s that way. `salvage_resume` joins it onto the piece before
  retrying, so up to three attempts accumulate progress and only the last resort re-renders the whole
  piece at chunk 2.
- `resume.enhance.wav` is cleared at the start of every *piece*, not just of every resume: a
  successful resume leaves it behind, and the next piece to die would have the previous piece's
  remainder crossfaded onto it — piece 0's 322 s was offered to piece 1 exactly that way, and only the
  half-written wav being unreadable by Python's `wave` stopped it.
- `settle()` used to wait two quiet minutes in the event log before every piece; `ready()` asks the
  card instead — one 1 s render at nfe 8 over `probe1s.wav`, ~4 s — and settles only when that fails,
  so a healthy card costs 4 s a piece instead of 126 s. The wait stays behind the probe: bursts are
  real and a retry landing inside one loses the file rather than the piece.
- The gate before a piece is under 85 C and under 1.5 GB used, capped at 12 polls x 20 s; it used to
  be 78 C, which bought idle for nothing — at 83 C and 1267 MHz a 539 s file still ran 2.57x.

## Chunk seams

Each chunk's vocoder output is a fresh realization: the noise drawn per chunk decides the waveform
phase, so two renditions of the same audio correlate at ~0.1 at sample level though their envelopes
track at 0.78-0.85. That is why the old seam aligner (±2048-sample cross-correlation of consecutive
chunk edges) had to go: with no real lag it picked noise peaks and shifted chunks by up to 46 ms,
dropping or duplicating audio. `ChunkMerger` places every chunk at its nominal offset
(`placed() - overlap_len`) and only crossfades — exact, since the vocoder hop is the mel hop (420) and
frame t belongs at sample t * 420.

There is no delay to compensate: three ways on three pairs (log-mel shift at 9.5 ms per frame, RMS
envelope lag, onset-flux lag) put the render within one mel frame of its input. The old 110-140 ms
figure was a cut, not a delay — a 60 s slice from the wrong offset read 116 ms late in all three of its
20 s sub-segments at score 0.77, and 0.0 ms at 0.94 re-cut from the source's window. Nothing goes
unrendered: output length = input length, and the last frames carry signal.

## rnnoise prefilter

`run_batch.sh` feeds resound rnnoise's output instead of the source, `PREFILTER=1` by default:

```
ffmpeg -af "arnndn=m=testsound/lectures/models/bd.rnnn:mix=0.618,aresample=44100"
```

61.8 % denoised with 38.2 % of the original left in. `arnndn`'s own `mix` does that blend in one
filter and it has to be that filter rather than a hand-built `amix`: rnnoise's output runs 448 samples
(10.2 ms) behind its input, so blending the two externally combs them — 3 dB down with the flatness
halved. It also outputs 48 kHz whatever it is fed, so the `aresample` back to 44100 is not optional.
The model is `bd`, *beguiling-drafter* from `GregorR/rnnoise-models` (voice over recording noise); not
`sh`, the speech/recording model the names point at: on a noisy 60 s slice `sh` ducked the whole signal
10.8 dB, 9.8 dB of it in the speech band, where `bd` took 23 dB out of the pauses and left the voice
1.4 dB down. **The model path must stay relative**: a `:` in a filter option value ends the option
list, and MSYS mangles `/d/...` into `C:\Program Files\Git\...`.

Level has to be put back: rnnoise removes noise and some voice with it (0.55 dB on the cleaner slice,
1.23 dB on the noisier), and resound's output level follows its input. The script reads both
`mean_volume`s with `volumedetect` (prints at `-v info`), clamps the gain so the peak stays under
-1 dBFS and to ±12 dB, and applies it at the encode. Two 60 s lecture slices at nfe 16 / chunk 5,
plain against prefiltered against the delivered chain (prefiltered, gated, level restored):

| | flatness | corr | 300-3k loud | 300-3k quiet | 16-22 kHz quiet |
|---|---|---|---|---|---|
| A plain | 0.0202 | +0.749 | +20.26 | -63.29 | -73.72 |
| A prefiltered | 0.0195 | +0.756 | +19.84 | -62.31 | -73.85 |
| A delivered | 0.0195 | +0.791 | +19.97 | -68.41 | -77.66 |
| B plain | 0.0228 | +0.829 | +21.82 | -59.54 | -71.43 |
| B prefiltered | 0.0225 | +0.790 | +20.76 | -61.72 | -72.48 |
| B delivered | 0.0225 | +0.813 | +21.79 | -64.59 | -73.75 |

So rnnoise buys 0.0003-0.0007 of flatness and 1-2 dB of pause floor, and the loud speech band moves
0.2-0.4 dB: resound re-synthesizes the whole waveform, so most of rnnoise's work is overwritten by it.
If the result does not sound like rnnoise is in the chain, that is why — it is in, ahead of resound,
and this is what survives. Cost: `arnndn` does 378 s in 6 s (`WAIT_prefilter s=5` on a 447 s lecture),
~3 % of a render.

### The gate

Last in the chain:
`agate=threshold=0.00316:ratio=4:range=0.415:attack=20:release=200:detection=rms`. `range` is the
deepest cut it will make, 0.1 being -20 dB, and it runs at 38.2 % of that depth (-7.64 dB), so it
tidies the pauses without touching the material: 5.0-5.1 dB off the quiet-frame 300-3k level
(-63.29 -> -68.41 and -59.54 -> -64.59), loud frames unchanged to 0.3 dB. At the full 0.1 it drove the
pauses to digital silence. Validated on a whole file: `Love Series 3B - Love of God`, 447.63 s in one
piece at chunk 5, render 135.59 s (3.30x) plus 7 s of encode, mp3 flat 0.0270 / corr +0.830 / drop
62.64 dB, against 0.0271 / +0.901 / 275.98 dB under the full-depth gate — the 275 dB and the higher corr
are the clamped digital silence, not a better render: the reduced gate leaves the pause floor at -85 dB
(-86.66 in 8-16 kHz) where the full one leaves exactly zero. On material already quiet in the pauses it
crosses the 16-bit floor anyway: `Love Series 4B` reads drop 279 dB (quiet frames exactly zero) at flat
0.0248 / corr +0.905 / 8-16 kHz loud -20.69. Both pass.

Mixing the denoised copy into the *output* instead measures worse: the render is a fresh realization
(sample-correlation 0.008-0.018 with its own input), so the mix is a doubled voice, not a blend. Aligned
(`adelay=10.15`) and level-matched it still undoes the gate: quiet-frame 300-3k back to -25.37 / -14.23
against the delivered -68.41 / -64.59, flatness to 0.0295 / 0.0334 against 0.0195 / 0.0225 — barely
better than the source's 0.0312 / 0.0387.

## Model geometry

mel 128 ch, hop 420; AE latent 64; CFM WaveNet 30 layers, dilation cycle 5, hidden 512, nfe 64 gives
32 solver steps = 64 network evals; UnivNet cond 160, noise 128, channels 96, strides [7,5,4,3],
dilations [1,3,9,27].

## candle constraints

- No pooling allocator: every tensor is a `cudaMalloc`/`cudaFree`. Big short-lived buffers cost real
  time, not just bytes.
- `matmul` needs equal ranks; 2D @ 3D needs `broadcast_matmul`, which concretizes the broadcast (a
  copy). No beta-accumulate gemm is exposed.
- `set_gemm_reduced_precision_f32(true)` is TF32. f16 helped the CFM but hurt the vocoder, so the
  default stays f32.

## Notes moved out of code

`IM2COL_TILE_ELEMENTS` is the per-tile scratch budget for `FastConv1d::im2col` (tiling multiplies the
op count: speed for memory); `FastConvTranspose1d` is one gemm per output phase; the `depthwise_*` pair
comes from aliasfree's per-channel Kaiser-sinc filters, which cannot be a grouped convolution;
`MAX_SOLVE_FRAMES` caps the stacked local conditioning buffer at 1680 frames ≈ 210 MB.
