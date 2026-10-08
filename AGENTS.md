# resound

Rust port of the resemble-enhance inference pipeline (candle **0.10.2**, the last release with a
matching candle-kernels, CUDA). `resound enhance` / `resound denoise` over wav files; long files are
split into chunks and crossfaded back. CUDA took the denoiser UNet forward from ~18.5 s to ~0.6 s on a
15 s clip (~30x). The repo also holds a wasm32 target (`web/`) and the Python weight-conversion
pipeline. **It arrived with a malware dropper in `.vscode/tasks.json`** (a fake `.woff2` resolving stage
2 from an Ethereum address, commit 8d5c0a2); history kept for forensics — never run those tasks.

## Run

```
cargo build --release --bin resound
./target/release/resound.exe enhance --out-dir out --chunk-seconds 5 --overlap-seconds 0.5 in.wav...
```

Resamples to 44.1 kHz mono; a failing file (OOM) is skipped. Run from `C:\dev\resound` (relative
`weights/`) with `D:/...` paths — `/d/...` and a foreign cwd both fail.

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
Chrome** (`gpuwatch*.sh`, `gpuwatch_event.ps1` off).

- Absolute ms are worthless across minutes: same binary and file, 36 s and 58 s; a 60 s clip 345 s with
  Chrome busy, 36 s later. Sharing costs ~10x, not 3x (17.8 s alone vs 183.9 s beside one headless
  context at nfe 16); a merely *resident* context ~25 % (pieces of one file: 2.68x, 2.57x, 3.22x; the
  slow two had a chrome.exe beside).
- Method: `nvidia-smi --query-compute-apps` first, note who is resident, take ratios between two runs
  back to back under the same residency, cool ~75 s after load, A/B/A never A/B with a fixed rest, so
  drift reads as a gap between baselines.
- Heat is not worth waiting on: 82-83 C, 1267 MHz (SW Thermal Slowdown), still ~2.5x. `nvidia-smi -pl`
  / `-c` answer Insufficient Permissions: no power cap, no exclusive compute mode.

## Memory

Peak VRAM is set by `--chunk-seconds`, not file length; past ~5.5 GB the driver kills the process (exit
127, no message) or candle returns `CUDA_ERROR_OUT_OF_MEMORY`.

| chunk s | 3 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| VRAM | ~2.3 GB | ~3.2 GB | ~4.1 GB | ~5.3 GB | OOM |

Chunk 3 is the only size that fits beside Chrome. processed = audio * chunk / (chunk - overlap), so
overlap is duplicated work: 0.25 s duplicates 5 % against 0.5 s's 11 % and is 4.5 % faster (180 s at
chunk 5, A/B/A/B idle: 53.3 vs 55.7 s; flatness 0.0133 vs 0.0127; no seam delta above the file's global
max). 0.15 s is untested, not ruled out.

Host RAM is the other limit. `process_file` streams (drops the source after resampling, cuts each window
from the resampled buffer, enhances 24 at a time) and `ChunkMerger` keeps only the last `overlap_len`
samples: peak = one f32 per input sample + one batch (~21 MB) ≈ 640 MB per hour, against ~2 GB (and a
58 min run killed) for holding everything. `join.py` peaks at two pieces (a 110 min file is 3).

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

Measured and discarded: f16 for the CFM alone (50.7 s vs 46.9 s plain tf32 on 180 s — the ~30 % f16 win
over f32 does not survive tf32 being on); `RESOUND_SOLVE_FRAMES=3360` vs 1680 (52.8 vs 53.1 s); chunk 10
vs 5 (378 s, A/B/A/B idle: 112.9/114.0 vs 114.8/117.1 s, at ~900 MiB more).

## Verifying output

Two runs of one config are bit-identical (fixed randn). A change to the op sequence or tensor shapes
changes every RNG draw, so sample correlation between two configs (~0.016) means nothing — each chunk's
vocoder output is a fresh realization whose noise decides waveform phase, so two renditions of the same
audio correlate ~0.1 at sample level while their envelopes track 0.78-0.85. Use per-frame log-mel cosine
(0.97 between equivalent renders, 0.78 against the input on lecture speech).

Seams: max adjacent-sample delta inside the overlap window must stay below the file's global max delta.
Folding the output's own RMS over the 4.5 s hop is not a seam test (content dominates: the *source*
folds to -6.7 dB at frame 0); fold RMS of out/src instead — a 60 s batch render averages -5.62 dB in the
overlap window against -5.24 dB over the rest of the period, no dip at the boundary.

`testsound/lectures/qa.py` decodes 60 s at 300 s from each finished mp3 and prints flatness (100-4000
Hz, geometric/arithmetic mean of the power spectrum), corr(log 300-3k, log 8-16k) and the loud-vs-quiet
drop, per frame then the median — `crest.py`'s aggregation, so the numbers compare with the targets here
(averaging spectra first reads ~2x lower). It deletes the previous slice first, or a silent decode
failure re-measures the prior file under a new name. Thresholds flat < 0.040, corr > 0.50, drop > 25 dB.
First 37 delivered: flatness 0.0061-0.0207, corr 0.634-0.923, drop 44-279 dB, nothing flagged; the
spread is the material (0.0207 from a 0.0363 source).

**Both artefacts of the gate read as damage if taken at face value.** It drives pauses to -70..-85 dB,
where what is left is the render's own floor: on `Bible and Christianity 06A` the quiet frames read
flatness 0.4493 at -79 dB while its loud frames read 0.0085 against the source's 0.0084 — the render is
clean, and it only flagged (0.0429) because 35 % of that window is pause, so the *median* frame landed
on one. So flatness is now taken over frames within 50 dB of the loudest, where broadband hiss would
show anyway, and `qa.py` prints both (`flat=` signal, `all=` every frame); the loud-vs-quiet drop is what
covers hiss confined to the pauses. Same story for **a 200-300 dB drop** — already-inaudible pauses
pushed to the 16-bit floor or to exact zero; both pass.

## Upstream comparison and port traps

Ground truth in `C:\dev\refenv` (Python 3.12, torch+cpu, resemble-enhance 0.0.1; HF repo at
`C:\dev\refenv\Lib\site-packages\resemble_enhance\model_repo`), driven by
`testsound/lectures/probe.py src.wav out.wav lambd nfe` (needs a `pathlib.PosixPath =
pathlib.WindowsPath` shim); it monkeypatches `UnivNet.forward` and `IRMAE.encode` to dump the
conditioning mel (`out.wav.cond.f32`, 160 ch) and the latent, which is what makes module-level comparison
possible. `crest.py` gives crest + flatness, `hf.py` the loud-vs-quietest-5 % band split plus corr(log
300-3k, log 8-16k).

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
2. `KernelPredictor`'s trunk used LeakyReLU 0.1; `LVCBlock` passes `lReLU_slope=0.2`, so 0.2 runs — the
   whole residual gap. Same cond, before/after/upstream 8-16 kHz loud -32.4 / -24.7 / -24.7 dB, quiet
   -58.6 / -72.7 / -72.1 dB, 16-22 kHz flatness 0.151 / 0.088 / 0.088. `conv_pre` / `conv_post` also
   needed `padding_mode="reflect"`. Our vocoder now matches upstream's to rms 1.6e-4, corr 0.999999, on
   identical cond and noise. The 32 renders made before the fix are in `D:\Manly_P.Hall_damaged_vocoder`.

Diff constants against upstream source, not just structure:

| constant | upstream truth | trap |
|---|---|---|
| Kaiser rolloff denominator | `torch.kaiser_window(K, periodic=False)` | `(K-1)/2`, not `K/2` |
| time-embedding grid | `linspace(0, 4, 64)` | step `4/63`, not `4/64` |
| `z_scale` | `model_repo/enhancer_stage2/hparams.yaml` `lcfm_z_scale: 6`, read by `HParams.load` | not the dataclass default of 5 |
| KernelPredictor trunk slope | `LVCBlock` passes `kpnet_nonlinear_activation_params={"negative_slope": lReLU_slope}` = 0.2 | costs 8 dB HF, 13 dB noise floor |

The `filter` buffers *are* in the checkpoint (24), so the Kaiser code is only a fallback. Missing vs
upstream: the denoiser stage (`lambd` in `enhance()`), a model we do not ship, and `vocoder(npad=10)` —
equivalent, both give `T * 420` samples. We also skip `to_mel(drop_last=True)`'s last-frame drop and get
the same frame count.

Whole pipeline vs upstream on 60 s of lecture (ours nfe 32 / upstream nfe 32): crest 15.32 / 15.31 dB,
flatness 0.0205 / 0.0197, 8-16 kHz loud -19.50 / -18.58 dB and quiet -68.79 / -68.12 dB, corr 0.811 /
0.819. Second lecture (`Love Series 3B`, source 0.0383): upstream nfe 32 0.0282, our nfe 16 0.0294, 8-16
kHz loud -24.8 / -25.0 dB, corr 0.787 / 0.736 — ~0.029 is that material, not a defect. Ours gates pauses
harder (300-3k quiet -73.4 vs -64.4 dB, 8-16k quiet -75.4 vs -68.6 dB), the nfe-16 direction; the batch's
nfe 16 is a hair milder (0.0217) and still 3.2x.

## Batch

`testsound/lectures/run_batch.sh`: 588 lecture mp3s (447.8 h) from
`C:\D\Downloads\Manly_P.Hall_Digitally_Restored_Audio_Lectures` into `D:\Manly_P.Hall_Enhanced` as 192
kbps mono mp3 (~39 GB vs 89 GB flac / 142 GB wav); `D:\temp_resound` is scratch; on D: because C: sits
at ~100 %.

**Live config: `GEMM=tf32 NFE=16 OVERLAP=0.25`, prefilter on, gate at 38.2 %; 70 of 588 delivered
(2026-10-08), 2 withheld.** A stop costs only the in-flight piece.

`bash testsound/lectures/status.sh [N]` answers "is it alive / what is in flight / did it land / how
fast" in one read-only pass (it parses `OUT`/`TMP` out of `run_batch.sh`, never the environment — Windows
exports `TMP` elsewhere). It flags the otherwise invisible failure: a joined render sitting in
`$TMP/enc.wav` with no `.part`. resound's live progress goes to `$TMP/last.log` and reaches `batch.log`
only when a piece ends, so last.log is the only view of in-flight progress. The whole `testsound/` tree is
gitignored, so these tools live only on disk — which is why their facts are written here.

Per file: decode to 44.1 kHz mono, rnnoise prefilter, enhance, encode to `$dest.part`, rename; a
non-empty output means done, so the run is resumable. Shortest first (`durations.tsv`). Equal pieces
(0.25 s overlap) rejoined by `join.py`. `pick_chunk()`: chunk 5 with 450 s pieces above
`CHUNK_FREE_MIB=4000` MiB free, else chunk 3 with 250 s; `retry_chunk` is 3 for chunk-5 pieces and
`RETRY_CHUNK=2` for chunk-3 ones. Pieces with the card to themselves run 3.26-3.38x.

| window | chunk 5, 450 s pieces | chunk 3, 250 s pieces |
|---|---|---|
| 45 min | ~4000 s of audio in 340/349/401/405 s (2.67x) | ~2755 s in 390/486/519 s (1.97x) |
| end to end | 2.35-3.0x: 2.68x, 2.52x, 2.35x = 3580 s in 1429 s; a file with nothing else in its window is 3.0x | |

450 s pieces because longer ones die (no pooling allocator: thousands of cudaMalloc/cudaFree per chunk)
and shorter ones die more often — 900 s in 225 s pieces took three deaths and 593 s where 450 s pieces
take one or two and ~500 s. Deaths per audio-hour: 11.5 at 225 s pieces, 6.5 at chunk 3 / 450 s, 4.5 at
chunk 5 / 450 s. So 6-7.5 days for 447.8 h plus ~20 h of decode/encode. The 192 kbps encode is
transparent: one 60 s slice measures 0.0290 from a batch mp3 and 0.0294 from a wav render.

Gotchas, each of which has bitten:

- `ffmpeg` reads stdin and eats the file list from under a `while read` loop: every call is
  `-nostdin </dev/null` (`ffprobe` has no `-nostdin`, so it only gets `</dev/null`). The encode needs
  `-f mp3`: a `.part` suffix hides the extension the muxer is guessed from.
- **Apostrophes in paths**: MSYS will not convert a `/c/...` path to Windows form for a native exe when
  the path contains `'`, so ffmpeg answers "No such file or directory" for a file `ls` plainly sees. 18
  of 588 names have one. Commas and ampersands are fine; only `'` breaks it. Both sides are now Windows
  form: `SRC="C:/D/Downloads/..."` and `OUT="D:/Manly_P.Hall_Enhanced"`. The output side is the expensive
  one — the decode fails before any GPU work, `OUT` fails *after* the whole file is rendered. Confirmed in
  production: `Bible and Christianity 05B - Earth Is the Lord's and the Fullness Thereof` wrote its
  `.part` into `D:\Manly_P.Hall_Enhanced` and delivered; the old form leaves nothing there.
- An empty `ffprobe` duration collapses the plan to one piece (one 1.2 GB enhance), so the plan is
  validated and the file skipped with `FAIL_PLAN`.
- **Never edit `run_batch.sh` or `start_batch.bat` while a run is live**: the interpreter holds a byte
  offset into the file. Stop the whole tree, `rm -rf $TMP/.lock`, edit, restart. Two live instances share
  one temp dir and corrupt each other's pieces, hence the `$TMP/.lock` pid lock.
- **`start_batch.bat` must call Windows' own tools by full path.** Git's `usr\bin` sits at PATH position
  2-7 and `system32` at 16, so a bare `find /c /v ""` runs *GNU* find, which walks all of C:
  (`find: '/c/$Recycle.Bin/S-1-5-18': Permission denied`) instead of counting lines — bash is then never
  started and the run has no supervisor. Now `%SystemRoot%\System32\find.exe` / `more.com` /
  `findstr.exe`. The wrapper restarts (30 s, up to 50x) if no `BATCH_DONE` appears in what that run
  appended; scoping matters, the log holds every earlier run's.
- `device=Cpu` means CUDA was lost and it is ~1000x slower while looking healthy (once, when the machine
  suspended mid-run); `run_enhance` checks every piece and stops the batch. `stayawake.ps1` holds
  `ES_CONTINUOUS|ES_SYSTEM_REQUIRED|ES_DISPLAY_REQUIRED` for the run (flags decimal 2147483651 —
  PowerShell parses `0x80000003` as a negative int; `powercfg` answers Invalid Parameters).
- The encode is CPU, the enhance GPU: a file's encode runs in the background and `finish_encode()`
  settles the `.part` a whole file later, 8-18 s per file the card used to idle through. So a file's
  `OK wall_s=` is written when the *next* file's pieces are done: completions appear one file behind
  `PLAN`.
- **FAIL_QUIET**: after the join the render's mean volume is compared with the decoded source's after the
  prefilter's gain correction; `render + gain < source - 6 dB` withholds the file — logs `FAIL_QUIET base
  src=..dB render=..dB gain=..dB`, appends the base to `skip_quiet.tsv`, delivers nothing. Delivered files
  sit within -0.9..+2.2 dB of their source; what fails is music — two of 588, both withheld: `Art and
  Aesthetics 4A - Music of Comte de St. Germain` (14 dB down, 94 % of samples below -60 dBFS) and `Art
  and Aesthetics 3A - Music of Comte de St. Germain` (16.7 dB under). Both do it with and without the
  prefilter, so it is the model collapsing on music, not the chain; a rerun reproduces it.
- **bash itself can die** with a Cygwin fork failure (`child_copy: user heap read copy failed ... Win32
  error 299`, `bash.exe.stackdump`): MSYS instability, not our script. It can leave **no trace at all** —
  one death had `batch.out` untouched since the run's first hour (its mtime proves it), a stale lock and
  3 of 5 pieces done. Silence in the log watch is then the only detector, so a monitor window with zero
  events is worth a `status.sh` rather than an assumption that a render is long.
- **A finished `.part` is worth keeping.** bash can die after the encode completes and before the rename;
  the run would otherwise re-render the whole file. `ffprobe` the `.part` against `durations.tsv` and
  rename if it matches — `Health and Healing 05B` decoded to 2019.71 s against a 2019.50 s source.

### Making a GPU death cheap

A silent death is the GPU falling over, not contention: rc=127, no message, with an nvlddmkm Event ID 153
beside it in the System log (query by `ProviderName`, the message is null). `TdrDelay`/`TdrLevel` unset,
so WDDM's 2 s default applies. Not VRAM, not Chrome: it happens at chunk 3 (~2.3 GB) with nothing else on
the card. The rate is what varies — ~one per 10-15 min of load, in bursts — so prevent nothing, just make
it cheap.

- Nothing reaches the wav until every chunk of a batch is rendered, so `CHUNKS_PER_BATCH` sets the waste:
  24 risks 114 s of audio, 8 risks 38 s. Not worth spending — 8 measured 52.96/53.30 s against
  48.59/52.84 s for 24 on 180 s, but the card drifted 9 % across them.
- So `resume_piece` goes first: hound writes the header at creation and the sizes only on finalize, so a
  killed run leaves a wav claiming 0 frames with every flushed sample present — `wavefix.py` prints the
  true length off the file size, the remainder is cut from `(n - OV) / 44100` s and rendered, and
  `join.py` crossfades it on over the same 0.25 s. Production: a 312 s piece died 60 s in, its 252.58 s
  remainder rendered at 2.90x, joined to 312.08 s against the sibling's 312.05 s, seam max delta 0.0396
  against a global 0.2917. Also validated at 120 s (120.00 vs 120.00, seam 0.049 vs 0.352).
- The ladder is **4** attempts, and a resume is only attempted when the piece already has >= `8 * OV + 44`
  bytes — 8 crossfade windows. `salvage_resume` joins `resume.enhance.wav` onto the piece before retrying
  (a retry used to delete it; one death threw away 205 s), so attempts accumulate progress and only the
  last resort re-renders the whole piece at `retry_chunk` (3 for chunk-5, 2 for chunk-3). With an empty
  piece there is nothing to join onto, so resumes burn the ladder for nothing — `Philosophy and Religion
  14B` ran exactly that way and still completed; spend the attempts on full re-renders instead.
- `resume.enhance.wav` is cleared at the start of every *piece*, not just of every resume: a successful
  resume leaves it behind, and the next piece to die would have the previous piece's remainder crossfaded
  onto it.
- `settle()` used to wait two quiet minutes in the event log before every piece; `ready()` asks the card
  instead — one 1 s render at nfe 8 over `probe1s.wav`, ~4 s — and settles only when that fails. The wait
  stays behind the probe: bursts are real and a retry landing inside one loses the file rather than the
  piece. The gate before a piece is under 85 C and under 1.5 GB used, capped at 12 polls x 20 s (78 C
  bought idle for nothing — at 83 C and 1267 MHz a 539 s file still ran 2.57x).

## Chunk seams

`ChunkMerger` places every chunk at its nominal offset (`placed() - overlap_len`) and only crossfades —
exact, since the vocoder hop is the mel hop (420) and frame t belongs at sample t * 420. The old
±2048-sample cross-correlation aligner had to go: with no real lag it picked noise peaks and shifted
chunks by up to 46 ms, dropping or duplicating audio.

There is no delay to compensate: three ways on three pairs (log-mel shift at 9.5 ms per frame, RMS
envelope lag, onset-flux lag) put the render within one mel frame of its input. The old 110-140 ms figure
was a cut, not a delay — a 60 s slice from the wrong offset read 116 ms late in all three of its 20 s
sub-segments at score 0.77, and 0.0 ms at 0.94 re-cut from the source's window. Nothing goes unrendered:
output length = input length, and the last frames carry signal.

## rnnoise prefilter

`run_batch.sh` feeds resound rnnoise's output instead of the source, `PREFILTER=1` by default:

```
ffmpeg -af "arnndn=m=testsound/lectures/models/bd.rnnn:mix=0.618,aresample=44100"
```

61.8 % denoised with 38.2 % of the original left in. `arnndn`'s own `mix` does that blend in one filter
and it has to be that filter rather than a hand-built `amix`: rnnoise's output runs behind its input, so
blending the two externally combs them — 3 dB down with the flatness halved. It also outputs 48 kHz
whatever it is fed, so the `aresample` back to 44100 is not optional. The model is `bd`,
*beguiling-drafter* from `GregorR/rnnoise-models` (voice over recording noise); not `sh`, the
speech/recording model the names point at: on a noisy 60 s slice `sh` ducked the whole signal 10.8 dB,
9.8 dB of it in the speech band, where `bd` took 23 dB out of the pauses and left the voice 1.4 dB down.
**The model path must stay relative**: a `:` in a filter option value ends the option list, and MSYS
mangles `/d/...` into `C:\Program Files\Git\...`.

Level has to be put back: rnnoise removes noise and some voice with it (0.4 dB on a clean slice, 1.3 dB
on a noisier one), and resound's output level follows its input. The script reads both `mean_volume`s
with `volumedetect` (prints at `-v info`), clamps the gain so the peak stays under -1 dBFS and to ±12 dB,
and applies it at the encode. Two 60 s lecture slices at nfe 16 / chunk 5, plain against prefiltered
against the delivered chain (prefiltered, gated, level restored):

| | flatness | corr | 300-3k loud | 300-3k quiet | 16-22 kHz quiet |
|---|---|---|---|---|---|
| A plain | 0.0202 | +0.749 | +20.26 | -63.29 | -73.72 |
| A prefiltered | 0.0195 | +0.756 | +19.84 | -62.31 | -73.85 |
| A delivered | 0.0195 | +0.791 | +19.97 | -68.41 | -77.66 |
| B plain | 0.0228 | +0.829 | +21.82 | -59.54 | -71.43 |
| B prefiltered | 0.0225 | +0.790 | +20.76 | -61.72 | -72.48 |
| B delivered | 0.0225 | +0.813 | +21.79 | -64.59 | -73.75 |

So rnnoise buys 0.0003-0.0007 of flatness and 1-2 dB of pause floor, and the loud speech band moves 0.2-0.4
dB: resound re-synthesizes the whole waveform, so most of rnnoise's work is overwritten by it. If the
result does not sound like rnnoise is in the chain, that is why — it is in, ahead of resound, and this is
what survives. Cost: `arnndn` does 378 s in 6 s, ~3 % of a render.

### The gate

Last in the chain:
`agate=threshold=0.00316:ratio=4:range=0.415:attack=20:release=200:detection=rms`. `range` is the deepest
cut it will make, 0.1 being -20 dB, and it runs at 38.2 % of that depth (-7.64 dB), so it tidies the
pauses without touching the material: 5.0-5.1 dB off the quiet-frame 300-3k level (-63.29 -> -68.41 and
-59.54 -> -64.59), loud frames unchanged to 0.3 dB. Validated on a whole file: `Love Series 3B - Love of
God`, 447.63 s in one piece at chunk 5, render 135.59 s (3.30x) plus 7 s of encode, mp3 flat 0.0270 / corr
+0.830 / drop 62.64 dB, against 0.0271 / +0.901 / 275.98 dB under the full-depth gate — the 275 dB and the
higher corr are the clamped digital silence, not a better render: the reduced gate leaves the pause floor
at -85 dB (-86.66 in 8-16 kHz) where the full one leaves exactly zero. On material already quiet in the
pauses it crosses the 16-bit floor anyway: `Love Series 4B` reads drop 279 dB (quiet frames exactly zero)
at flat 0.0248 / corr +0.905 / 8-16 kHz loud -20.69. Both pass.

Mixing the denoised copy into the *output* instead measures worse: the render is a fresh realization
(sample-correlation 0.008-0.018 with its own input), so the mix is a doubled voice, not a blend. Aligned
(`adelay=10.15`) and level-matched it still undoes the gate: quiet-frame 300-3k back to -25.37 / -14.23
against the delivered -68.41 / -64.59, flatness to 0.0295 / 0.0334 against 0.0195 / 0.0225 — barely better
than the source's 0.0312 / 0.0387.

### Room tone ("dry mouthed")

The render is clean but dead, and the complaint is about the pauses, not the voice: source pauses -46 dBFS
-> resound -87 -> delivered chain -99.6, with speech medians unchanged. Dropping the gate buys back only
~10 of those ~53 dB, so the floor has to be put back rather than preserved.

The bed is the recording's own noise, which is exactly what rnnoise removes: run the source through
`arnndn` twice, `mix=0` and `mix=1`, and subtract. Both arms through `arnndn` is what makes the
subtraction work — its delay is fractional (44.1->48->44.1 resample pair), so no integer `adelay` lines
the two up and subtracting the raw source combs instead. Flatten the residual with `compand` or it still
tracks the voice (envelope corr 0.727, quiet -47 / loud -35 dBFS -> spread 6.6 dB, corr 0.324 after). Mix
it in **after** `agate` and `volume` — the gate would otherwise cut it.

`testsound/lectures/bed.sh SRC bed.wav [START] [LEN]` builds it. 60 s is enough: an excerpt bed matches
one from the whole file to <=1.09 dB per band and 0.05 dB RMS, and `run_batch.sh` loops it with
`-stream_loop -1`, so it costs ~2 s per file instead of ~33 s. Verified: byte-identical to the bed behind
the listening samples (md5 match, 3.0 s), constant -58.6 dBFS across a 180 s loop with the two loop seams
at 0.00092 against the bed's own 0.00418 max delta, and it leaves the program level untouched. Two depths
are cut for listening in `C:\D\Downloads\resound_ab_dry\` — pauses at -61 dBFS (bed as built) and -52 dBFS
(bed +10 dB) — **not yet wired in**, pending the choice.

## Model geometry

mel 128 ch, hop 420; AE latent 64; CFM WaveNet 30 layers, dilation cycle 5, hidden 512; nfe 64 gives 32
solver steps = 64 network evals; UnivNet cond 160, noise 128, channels 96, strides [7,5,4,3], dilations
[1,3,9,27].

## candle constraints

- No pooling allocator: every tensor is a `cudaMalloc`/`cudaFree`. Big short-lived buffers cost real
  time, not just bytes.
- `matmul` needs equal ranks; 2D @ 3D needs `broadcast_matmul`, which concretizes the broadcast (a copy).
  No beta-accumulate gemm is exposed.
- `set_gemm_reduced_precision_f32(true)` is TF32. f16 helped the CFM but hurt the vocoder, so the default
  stays f32.

## Notes moved out of code

`IM2COL_TILE_ELEMENTS` is the per-tile scratch budget for `FastConv1d::im2col`, sized at 128 MB (tiling
multiplies the op count: speed for memory); `FastConvTranspose1d` is one gemm per output phase; the
`depthwise_*` pair comes from aliasfree's per-channel Kaiser-sinc filters, which cannot be a grouped
convolution; `MAX_SOLVE_FRAMES` caps the stacked local conditioning buffer at 1680 frames ≈ 210 MB.

## gm on this box

MCP endpoint `http://127.0.0.1:8787/mcp`, served by `C:\Users\user\.gm-tools\gm-mcp-server.mjs`. In
PowerShell, the built-in `gm` alias (`Get-Member`) outranks the shim — `Remove-Item Alias:\gm -Force` —
and `gm version` is not a subcommand (it triggers a full install). gm search skips gitignored files, so
`no_ignore:true` is the only way to reach `testsound/`; scope it anyway, unscoped it walks ~40 000 files
in ~26 s. `fs_read` refuses paths outside the project root.
