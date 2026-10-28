"""ctc_asr_text_correct -- in-process Korean ASR (fine-tuned Wav2Vec2-CTC) + Gemini correction.

No HTTP server. Load the model once, transcribe + correct clips or stream mic frames.

    from ctc_asr_text_correct import ASRCorrector

    pipe = ASRCorrector()                      # CTC model + Gemini (GEMINI_API_KEY)
    r = pipe.process("clip.wav")
    print(r.transcript, "->", r.corrected_text)

    stream = pipe.stream(on_segment=lambda s: print(s.corrected_text))
    for chunk in mic_frames():                 # float32 mono @ pipe.sample_rate
        stream.accept(chunk)
    stream.flush()
"""

from .config import CorrectorConfig, EngineConfig, default_model_dir, default_vocab_path
from .corrector import DEFAULT_PROMPT, Correction, GeminiCorrector
from .engine import Wav2Vec2CTCEngine, get_engine, per_sample_cer
from .hub import resolve_from_hub, upload_checkpoint
from .transcriber import ASRCorrector, Result, Segment, StreamingTranscriber, Transcriber

__all__ = [
    "ASRCorrector",
    "Result",
    "Transcriber",
    "StreamingTranscriber",
    "Segment",
    "GeminiCorrector",
    "Correction",
    "DEFAULT_PROMPT",
    "EngineConfig",
    "CorrectorConfig",
    "Wav2Vec2CTCEngine",
    "get_engine",
    "per_sample_cer",
    "default_model_dir",
    "default_vocab_path",
    "resolve_from_hub",
    "upload_checkpoint",
]

__version__ = "0.1.0"
