"""Smoke tests.

STT tests need the CTC checkpoint (CTC_HF_REPO, CTC_MODEL_DIR, or the sibling
service_ctc repo) and a sample wav. Gemini tests need GEMINI_API_KEY(S); the
fallback tests need neither network nor a valid key.

    pytest -q            # from the library root
"""

import glob
import os

import numpy as np
import pytest
import soundfile as sf

from ctc_asr_text_correct import ASRCorrector, GeminiCorrector, Transcriber
from ctc_asr_text_correct.config import default_model_dir, gemini_api_keys_from_env

_SAMPLE_GLOBS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "service_ctc", "samples", "*.wav"),
    os.path.join(os.path.dirname(__file__), "samples", "*.wav"),
]

needs_model = pytest.mark.skipif(
    default_model_dir() is None and not os.environ.get("CTC_HF_REPO"),
    reason="no checkpoint available (set CTC_HF_REPO or CTC_MODEL_DIR)",
)
needs_gemini = pytest.mark.skipif(not gemini_api_keys_from_env(), reason="no GEMINI_API_KEY")


def _find_sample():
    for pattern in _SAMPLE_GLOBS:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return None


@pytest.fixture(scope="session")
def asr():
    return Transcriber()


@pytest.fixture(scope="session")
def sample():
    path = _find_sample()
    if path is None:
        pytest.skip("no sample wav")
    return path


# --- correction ----------------------------------------------------------


def test_corrector_fallback_on_bad_key():
    corrector = GeminiCorrector(api_key="invalid-key")
    out = corrector.correct_detailed("  안녕   하세요  ")
    assert not out.ok
    assert out.text == "안녕 하세요"


def test_corrector_no_fallback_raises():
    corrector = GeminiCorrector(api_key="invalid-key", fallback=False)
    with pytest.raises(RuntimeError):
        corrector.correct("안녕 하세요")


def test_corrector_empty_passthrough():
    assert GeminiCorrector(api_key="unused").correct("   ") == "   "


@needs_gemini
def test_corrector_live():
    out = GeminiCorrector().correct_detailed("은 변원 앞에 있어요 약을 하우레 두 씩 드세요")
    assert out.ok and out.text


# --- STT -----------------------------------------------------------------


@needs_model
def test_transcribe_file(asr, sample):
    assert isinstance(asr.transcribe(sample), str)


@needs_model
def test_transcribe_waveform_matches_file(asr, sample):
    wav, sr = sf.read(sample, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    assert asr.transcribe(sample) == asr.transcribe(wav, input_sr=sr)


@needs_model
def test_streaming_segments(asr, sample):
    wav, sr = sf.read(sample, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    # 300ms lead/tail silence so the VAD sees a clean utterance boundary.
    pad = np.zeros(int(sr * 0.3), dtype=np.float32)
    wav = np.concatenate([pad, wav, pad])

    stream = asr.stream()
    got = []
    step = int(sr * 0.1)  # 100ms push chunks
    for i in range(0, len(wav), step):
        got.extend(stream.accept(wav[i : i + step], input_sr=sr))
    tail = stream.flush()
    if tail:
        got.append(tail)

    assert len(got) >= 1
    assert any(s.text for s in got)


# --- pipeline ------------------------------------------------------------


@needs_model
def test_pipeline_with_fallback(asr, sample):
    pipe = ASRCorrector(transcriber=asr, api_key="invalid-key")
    r = pipe.process(sample)
    assert r.correction_ok is False
    assert r.corrected_text == " ".join(r.transcript.split())


@needs_model
def test_streaming_with_fallback_correction(asr, sample):
    pipe = ASRCorrector(transcriber=asr, api_key="invalid-key")
    wav, sr = sf.read(sample, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    pad = np.zeros(int(sr * 0.3), dtype=np.float32)
    stream = pipe.stream()
    got = stream.accept(np.concatenate([pad, wav, pad]), input_sr=sr)
    tail = stream.flush()
    if tail:
        got.append(tail)
    assert got and all(s.corrected_text == " ".join(s.text.split()) for s in got)


@needs_model
def test_streaming_background_correction(asr, sample):
    pipe = ASRCorrector(transcriber=asr, api_key="invalid-key")
    wav, sr = sf.read(sample, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    pad = np.zeros(int(sr * 0.3), dtype=np.float32)
    corrected = []
    stream = pipe.stream(on_correction=corrected.append)
    got = stream.accept(np.concatenate([pad, wav, pad]), input_sr=sr)
    tail = stream.flush()  # waits for background corrections
    if tail:
        got.append(tail)
    assert got and sorted(id(s) for s in corrected) == sorted(id(s) for s in got)
    assert all(s.corrected_text == " ".join(s.text.split()) for s in got)


def test_quantized_matches_fp32(sample):
    if default_model_dir() is None:
        pytest.skip("no local checkpoint")
    fp32 = Transcriber(force_cpu=True, cache=False)
    int8 = Transcriber(force_cpu=True, quantize=True, cache=False)
    assert int8.transcribe(sample) == fp32.transcribe(sample)


@needs_model
@needs_gemini
def test_pipeline_live(asr, sample):
    r = ASRCorrector(transcriber=asr).process(sample)
    assert r.correction_ok and r.corrected_text
