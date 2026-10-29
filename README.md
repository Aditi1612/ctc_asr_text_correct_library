# ctc_asr_text_correct

In-process Korean **STT + transcript correction**: the fine-tuned
`Wav2Vec2ForCTC` model from `service_ctc` followed by Gemini correction. There's
**no HTTP server and no calls to the API**. You load the model once inside your
app, then transcribe and correct clips or a live mic stream directly.

This is a standalone package, combining two sibling libraries:
- the STT side is `ctc_asr_library`: audio normalization, CTC decode, jamo
  normalization and Hangul composition, copied from the serving code so output
  matches the API;
- the correction side is the `GeminiCorrector` from `asr_text_correct_library`:
  the same prompt and Gemini key rotation with fallback.

It does **not** modify or import `APIs/service_ctc/`. By default it only reads
the checkpoint files from there.

```
audio ──► Wav2Vec2-CTC (fine-tuned, ko) ──► transcript ──► Gemini ──► corrected_text
          jamo decode + Hangul compose                     (key rotation, fallback)
```

## Install

On this server (editable, from source):

```bash
conda activate ctc
pip install -e /data2/irfan/projects/APIs/ctc_asr_text_correct_library
```

On any other machine, install from GitHub:

```bash
pip install "ctc-asr-text-correct[hub] @ git+https://github.com/Aditi1612/ctc_asr_text_correct_library.git"
```

You can also build a wheel and hand over the file:

```bash
pip install build && python -m build      # -> dist/ctc_asr_text_correct-0.1.0-*.whl
pip install "ctc_asr_text_correct-0.1.0-py3-none-any.whl[hub]"
```

`torch` / `torchaudio` are heavy and hardware-specific. Install the right
CUDA/CPU build first, then install this.

Optional extras:
- `[hub]`: Hugging Face Hub download.
- `[vad]`: webrtcvad, which works better than the energy fallback.
- `[mic]`: sounddevice, needed for the mic demo.

## Gemini API key

Correction needs a Gemini key. You can supply it in any of these ways:

```bash
export GEMINI_API_KEY=...                  # single key
export GEMINI_API_KEYS=key1,key2,key3      # rotation (takes priority)
export GEMINI_MODEL=gemini-2.5-flash-lite  # optional, this is the default
```

You can also pass the key in code: `ASRCorrector(api_key=...)` or
`ASRCorrector(api_keys=[...])`.

How the corrector handles keys and failures:
- **Rotation:** with several keys it round-robins between them. On a 429 or any
  other error it tries the next key.
- **Fallback:** if every key fails, it returns the raw transcript with the
  whitespace collapsed and sets `correction_ok=False`.
- **Strict mode:** pass `fallback=False` to raise an error instead.

## Model assets

The checkpoint (~1.2 GB) is **not** in the package. Choose a source:

### A. Hugging Face Hub (use this off this server)

The checkpoint is published to the **gated** model repo
[`fluffy16/ctc-asr-ko`](https://huggingface.co/fluffy16/ctc-asr-ko), the same
one `ctc_asr` uses. Install with the `[hub]` extra:

```python
ASRCorrector(hf_repo="fluffy16/ctc-asr-ko")   # ~1.2 GB download on first use, then cached
```

The env-var form is `export CTC_HF_REPO=fluffy16/ctc-asr-ko`, with no code
change.

Anyone can see the repo, but downloading the weights requires approval:

1. Sign in to Hugging Face, open the model page, and click **Request access**.
2. Wait for the owner to approve the request.
3. Authenticate as that account, in any one of these ways:
   - `huggingface-cli login`
   - `export HF_TOKEN=hf_xxx`
   - `ASRCorrector(hf_repo=..., hf_token="hf_xxx")`

Without approval, or without a token, the download fails with `GatedRepoError`.
The owner reviews pending requests on the model page, under
**Settings → Gated user access**.

To pin an exact model version, pass `hf_revision="<commit-sha>"`.

To publish a new checkpoint (run once, from this server):

```bash
huggingface-cli login          # or: export HF_TOKEN=hf_xxx
python examples/upload_to_hf.py fluffy16/ctc-asr-ko \
  /data2/irfan/projects/APIs/service_ctc/transcribe_bundle/checkpoint \
  /data2/irfan/projects/APIs/service_ctc/transcribe_bundle/vocab.json
```

### B. Local files

```python
ASRCorrector("/abs/path/checkpoint", "/abs/path/vocab.json")
# or
export CTC_MODEL_DIR=/abs/path/checkpoint
export CTC_VOCAB_PATH=/abs/path/vocab.json
```

### C. This server only

With no arguments, the library auto-finds the sibling
`../service_ctc/transcribe_bundle/` checkpoint.

**Resolution priority:** explicit `model_dir`/`vocab_path` → `hf_repo` /
`CTC_HF_REPO` → `CTC_MODEL_DIR`/`CTC_VOCAB_PATH` → sibling checkpoint.

## One-shot: STT + correction

```python
from ctc_asr_text_correct import ASRCorrector

pipe = ASRCorrector()                 # loads the CTC model once
r = pipe.process("clip.wav")
r.transcript         # composed Hangul from the CTC model
r.corrected_text     # Gemini-corrected sentence
r.raw_transcription  # undecoded CTC jamo output
r.correction_ok      # False if the Gemini fallback was used
r.timings            # {'stt': ..., 'text_correction': ..., 'total': ...}
r.to_dict()
```

Options for `process()`:
- `reference_text=...` fills `r.cer`.
- `prompt=...` overrides the correction prompt for that call.

`process()` accepts a file path, encoded bytes (`wav/flac/ogg`, plus `mp3/m4a`
if ffmpeg is present), or a numpy/torch waveform passed with `input_sr=`.

### Only one half

```python
from ctc_asr_text_correct import Transcriber, GeminiCorrector

asr = Transcriber()                                  # STT only, no API key needed
asr.transcribe("clip.wav")                           # -> str
asr.transcribe("clip.wav", detailed=True)            # -> dict with raw_transcription / cer

fix = GeminiCorrector()                              # correction only, no model load
fix.correct("은 변원 앞에 있어요 약을 하우레 두 씩 드세요")
fix.correct_detailed(text)                           # Correction(text, raw_text, ok, model, error)

pipe = ASRCorrector(transcriber=asr, corrector=fix)  # combine existing objects
```

## Streaming (VAD-segmented)

Your app captures audio from any source and pushes mono float32 frames. Each
time the speaker pauses, the buffered utterance is transcribed, then corrected:

```python
pipe = ASRCorrector()
stream = pipe.stream(on_segment=lambda s: print(s.text, "->", s.corrected_text))

for chunk in your_mic_source():                    # any length, float32 mono
    for seg in stream.accept(chunk, input_sr=48000):
        ...                                        # seg.text, seg.corrected_text, seg.start_time

stream.flush()                                     # finalize trailing speech
```

- `pipe.stream(correct=False)` and `Transcriber().stream()` return raw STT
  only, and `corrected_text` stays `None`.
- By default `accept()` blocks while a finished utterance is transcribed and
  corrected: one short CTC pass plus one Gemini round-trip. Pass
  `on_correction=` to correct in the background instead (see below).

Runnable mic demo (low-latency setup): `python examples/stream_mic.py`

### Tuning `stream(...)`

| arg | default | meaning |
|---|---|---|
| `end_silence_ms` | 600 | pause length that ends an utterance |
| `min_speech_ms` | 250 | shorter blips are dropped |
| `max_speech_ms` | 15000 | force-cut a segment that never pauses |
| `start_speech_frames` | 3 | consecutive speech frames to trigger |
| `pre_roll_ms` | 200 | audio kept before the trigger point |
| `vad_aggressiveness` | 2 | webrtcvad 0–3 (higher = stricter) |
| `on_correction` | `None` | if set, correct in background threads and call this with each corrected segment |
| `correction_workers` | 4 | parallel Gemini calls in background mode |

## Low latency (CPU only)

After the speaker stops, latency is three parts: the `end_silence_ms` wait,
the CTC pass, and the Gemini round-trip. Gemini dominates (~0.6–0.9 s; it's
network-bound, and newer Flash-Lite models aren't faster), so the main lever is
to show the raw text immediately and swap in the correction when it arrives:

```python
pipe = ASRCorrector(force_cpu=True, quantize=True)   # int8: ~35% faster CPU STT
stream = pipe.stream(
    end_silence_ms=300,
    on_segment=lambda s: ui.show(s.start_time, s.text),               # raw, fast
    on_correction=lambda s: ui.replace(s.start_time, s.corrected_text),  # ~0.8 s later
)
```

Measured on this server (24-core Threadripper, CPU only, real-time-paced mic
simulation over 4 sample utterances, speech end → text shown):

| setup | raw text | corrected text |
|---|---|---|
| defaults (600 ms, fp32, blocking) | 1.6 s | 1.6 s |
| 300 ms, int8, `on_correction` | **0.42 s** | 1.2 s |
| 200 ms, int8, `on_correction` | 0.30 s | 1.1 s |

- `on_correction` callbacks run on worker threads and can arrive out of order;
  match them up by `seg.start_time`. `flush()` waits for all of them.
- `quantize=True` gave identical transcripts to fp32 on every sample. It is
  CPU-only (ignored on CUDA) and uses torch's dynamic quantization, which torch
  plans to remove in 2.10.
- CPU STT per utterance (24 threads, int8): ~75 ms for 2 s of audio, ~150 ms
  for 5 s, ~640 ms for 22 s. Don't raise torch's thread count above the
  physical core count: 48 threads on this 24-core CPU took ~2.9 s per clip.
- Short `end_silence_ms` cuts sentences at pauses. The samples here never
  split even at 150 ms, but dysarthric speech has longer mid-sentence pauses,
  so test on real users before going below ~300 ms.

## Notes / limits

- The CTC model is a full-utterance model, so there is no token-by-token
  streaming. Splitting on silence can cut a word at a boundary.
- Correction is an LLM rewrite. It may change verb endings and add punctuation,
  so keep `transcript` if you need the literal STT output.
- One `ASRCorrector` is fine for sequential calls but runs one inference at a
  time. If you need parallelism, use one per worker. `Transcriber` caches the
  loaded model per config, so creating it again is cheap.

## Layout

```
ctc_asr_text_correct/
  config.py        EngineConfig / CorrectorConfig + asset & API-key resolution
  hub.py           HF Hub download + one-time checkpoint upload
  audio.py         decode / resample / normalize, processor build
  postprocess.py   jamo normalize + Hangul compose  (vendored from service_ctc)
  engine.py        Wav2Vec2CTCEngine  (mirrors service_ctc/ctc_infer.py)
  corrector.py     GeminiCorrector: prompt, key rotation, fallback
  transcriber.py   Transcriber, ASRCorrector, StreamingTranscriber, Segment, Result
  vad.py           webrtcvad wrapper + energy fallback
docs/              user guide: guide.html + PDF (English), guide.ko.html + PDF (Korean)
examples/          transcribe_file.py, correct_text.py, stream_mic.py, upload_to_hf.py
tests/             test_smoke.py
```
