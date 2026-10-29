"""Live mic streaming demo with per-utterance correction (requires: pip install sounddevice).

    export GEMINI_API_KEY=...
    python examples/stream_mic.py

Speak, pause, and each utterance's raw transcript is printed as soon as you stop
talking; the Gemini-corrected line follows when it arrives (background
correction, so the mic loop never waits on the network). Ctrl-C to quit. This file owns the microphone; the library itself only
consumes frames, so in your own app you can push audio from any source.
"""

import queue
import sys

import numpy as np

from ctc_asr_text_correct import ASRCorrector

try:
    import sounddevice as sd
except ImportError:
    sys.exit("pip install sounddevice to run this demo")


def show_raw(seg) -> None:
    print(f"  [{seg.start_time:6.2f}s] {seg.text}")


def show_corrected(seg) -> None:
    print(f"  [{seg.start_time:6.2f}s] -> {seg.corrected_text}")


def main() -> None:
    pipe = ASRCorrector(quantize=True)  # int8 speeds up CPU inference; ignored on CUDA
    print(f"[device={pipe.device}] loading done. Speak now (Ctrl-C to stop).")

    stream = pipe.stream(
        end_silence_ms=300, on_segment=show_raw, on_correction=show_corrected
    )
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    def cb(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            print(status, file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=pipe.sample_rate, channels=1, dtype="float32", callback=cb, blocksize=0
    ):
        try:
            while True:
                stream.accept(audio_q.get())
        except KeyboardInterrupt:
            stream.flush()  # prints the trailing utterance, waits for pending corrections
            print("\nstopped.")


if __name__ == "__main__":
    main()
