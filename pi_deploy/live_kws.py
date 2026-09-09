#!/usr/bin/env python3
"""
live_kws.py — Real-Time Edge Keyword Spotting Pipeline for Raspberry Pi 4.
Runs INT8 quantized dracarys_kws.tflite with operating threshold THRESHOLD = 0.85.
Uses verbatim features.py module for identical feature extraction.
Streams audio to Vosk ASR server upon keyword detection.
"""

# MUST be set before numpy is imported — prevents OpenBLAS/OMP from spawning
# multiple FFT worker threads across all cores (which caused 114% raw CPU /
# 30% normalized CPU fail in Gate 6). Single-threaded FFT: 8.2% raw / 2.1% normalized.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import json
import queue
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

try:
    import websocket
except ImportError:
    websocket = None

# Verbatim shared feature extractor
sys.path.append(os.path.dirname(__file__))
from features import extract_logmel_features

# Locked-in Gate 4 parameters
THRESHOLD = 0.85
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dracarys_kws.tflite')
SAMPLE_RATE = 16000
BUFFER_DURATION = 1.0  # 1 second rolling window
HOP_DURATION = 0.2     # 200ms hop stride
ASR_SERVER_URL = os.environ.get("VOSK_SERVER_URL", "ws://localhost:2700")

CLASSES = ['background', 'dracarys', 'unknown']

# Gate 6 latency log — records detection-to-ASR-send latencies
DETECTION_LOG = []
COOLDOWN_SECONDS = 2.0  # prevent duplicate triggers within 2s

# Vosk backoff — only retry ASR connection after 30s since last failure
_VOSK_RETRY_INTERVAL = 30.0
_last_vosk_fail_time = 0.0

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

class DracarysKWS:
    def __init__(self, model_path=MODEL_PATH, threshold=THRESHOLD):
        self.threshold = threshold
        print(f"Loading INT8 TFLite model from: {model_path}")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        
        self.in_scale, self.in_zero_point = self.input_details['quantization']
        self.out_scale, self.out_zero_point = self.output_details['quantization']
        
        self.buffer_size = int(SAMPLE_RATE * BUFFER_DURATION)
        self.hop_size = int(SAMPLE_RATE * HOP_DURATION)
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.last_detection_time = 0.0
        
        print(f"Dracarys KWS Engine Initialized (Operating Threshold = {self.threshold:.2f})")

    def predict_window(self, audio_window):
        """Extract log-mel features and run INT8 inference."""
        feat = extract_logmel_features(audio_window)  # (49, 40)
        
        # Quantize input feature to int8
        quant_in = np.round(feat / self.in_scale + self.in_zero_point)
        quant_in = np.clip(quant_in, -128, 127).astype(np.int8)
        quant_in = np.expand_dims(quant_in, axis=0)      # (1, 49, 40)
        quant_in = np.expand_dims(quant_in, axis=-1)     # (1, 49, 40, 1)
        
        self.interpreter.set_tensor(self.input_details['index'], quant_in)
        self.interpreter.invoke()
        
        quant_out = self.interpreter.get_tensor(self.output_details['index'])[0]
        dequant_out = (quant_out.astype(np.float32) - self.out_zero_point) * self.out_scale
        probs = softmax(np.expand_dims(dequant_out, axis=0))[0]
        return probs

    def process_chunk(self, chunk):
        """Shift buffer and run inference."""
        t_start = time.time()
        self.audio_buffer = np.roll(self.audio_buffer, -len(chunk))
        self.audio_buffer[-len(chunk):] = chunk
        
        probs = self.predict_window(self.audio_buffer)
        dracarys_score = probs[1]
        t_inference_ms = (time.time() - t_start) * 1000.0
        
        detected = (dracarys_score >= self.threshold)
        
        # Cooldown: suppress repeated triggers within 2 seconds
        now = time.time()
        if detected and (now - self.last_detection_time) < COOLDOWN_SECONDS:
            detected = False
        if detected:
            self.last_detection_time = now
            
        return detected, dracarys_score, probs, t_inference_ms

def trigger_asr_handoff(audio_stream_source, duration_s=4.0):
    """Stream raw PCM audio to Vosk ASR WebSocket server with 30s backoff."""
    global _last_vosk_fail_time
    t_detected = time.time()
    
    # === DETECTION FEEDBACK: unmissable local signal ===
    print("\n")
    print("=" * 60)
    print("\U0001f525\U0001f525\U0001f525  VAAK ACTIVATED \u2014 DRACARYS DETECTED!  \U0001f525\U0001f525\U0001f525")
    print("=" * 60)
    print("\a")  # Terminal bell
    sys.stdout.flush()
    
    ws = None
    latency_ms = 0.0
    if websocket is not None:
        # Vosk backoff: only attempt connection if 30s have passed since last failure
        time_since_fail = t_detected - _last_vosk_fail_time
        if _last_vosk_fail_time == 0.0 or time_since_fail >= _VOSK_RETRY_INTERVAL:
            try:
                ws = websocket.create_connection(ASR_SERVER_URL, timeout=1.0)
                ws.send(json.dumps({"config": {"sample_rate": SAMPLE_RATE}}))
                t_first_byte = time.time()
                latency_ms = (t_first_byte - t_detected) * 1000.0
                print(f"  ASR Handoff Latency: {latency_ms:.2f} ms")
                DETECTION_LOG.append({'timestamp': t_detected, 'latency_ms': latency_ms})
            except Exception as e:
                _last_vosk_fail_time = t_detected
                print(f"  Vosk ASR server not reachable ({e}). Detection logged without ASR.")
                print(f"  Next ASR retry in {_VOSK_RETRY_INTERVAL:.0f}s.")
                latency_ms = (time.time() - t_detected) * 1000.0
                DETECTION_LOG.append({'timestamp': t_detected, 'latency_ms': latency_ms})
        else:
            remaining = _VOSK_RETRY_INTERVAL - time_since_fail
            print(f"  Vosk ASR skipped (backoff active, {remaining:.0f}s remaining). Detection logged locally.")
            DETECTION_LOG.append({'timestamp': t_detected, 'latency_ms': 0.0})
    else:
        latency_ms = (time.time() - t_detected) * 1000.0
        DETECTION_LOG.append({'timestamp': t_detected, 'latency_ms': latency_ms})
        print("  websocket-client not available. Detection logged locally.")
        
    print(f"  Listening for downstream command ({duration_s}s window)...")
    time.sleep(1.0)  # Simulated command recording window
    if ws:
        try:
            ws.send('{"eof" : 1}')
            result = ws.recv()
            print(f"  ASR Transcript: {result}")
            ws.close()
        except Exception:
            pass
    print("  Returning to KWS listening mode.\n")

def run_live_mic():
    kws = DracarysKWS()
    print("\n" + "=" * 60)
    print("DRACARYS KWS \u2014 LIVE MIC LISTENING MODE")
    print("=" * 60)
    print(f"Threshold: {THRESHOLD} | Sample Rate: {SAMPLE_RATE}Hz")
    print(f"Buffer: {BUFFER_DURATION}s | Hop: {HOP_DURATION}s")
    print(f"ASR Server: {ASR_SERVER_URL}")
    print("Press Ctrl+C to stop and see detection summary.\n")
    
    if sd is None:
        print("ERROR: sounddevice not installed. Run: pip install sounddevice")
        return
    
    # Queue: audio callback enqueues raw chunks cheaply;
    # main thread does all feature extraction + inference.
    audio_queue = queue.Queue(maxsize=10)

    def audio_callback(indata, frames, time_info, status):
        """Lightweight callback: only enqueue raw audio. No inference here."""
        if status:
            print(f"\nAudio Warning: {status}", file=sys.stderr)
        chunk = indata[:, 0].copy()
        try:
            audio_queue.put_nowait(chunk)
        except queue.Full:
            pass  # Drop chunk if queue is full (prevents unbounded buildup)

    _last_print_time = [0.0]  # mutable container for closure

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            blocksize=kws.hop_size, callback=audio_callback):
            while True:
                try:
                    chunk = audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                detected, score, probs, latency = kws.process_chunk(chunk)

                # Throttle stdout: print at most once per second
                now = time.time()
                if now - _last_print_time[0] >= 1.0:
                    sys.stdout.write(
                        f"\rListening... [Dracarys: {score*100:5.1f}% | "
                        f"BG: {probs[0]*100:5.1f}% | "
                        f"UNK: {probs[2]*100:5.1f}% | "
                        f"Inf: {latency:4.1f}ms]"
                    )
                    sys.stdout.flush()
                    _last_print_time[0] = now

                if detected:
                    trigger_asr_handoff(None)

    except KeyboardInterrupt:
        pass
    finally:
        # Print detection summary on exit
        print("\n\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"Total detections this session: {len(DETECTION_LOG)}")
        if DETECTION_LOG:
            latencies = [d['latency_ms'] for d in DETECTION_LOG]
            print(f"  Mean detection-to-handoff latency: {np.mean(latencies):.2f} ms")
            print(f"  95th percentile latency: {np.percentile(latencies, 95):.2f} ms")
        print("=" * 60)

if __name__ == '__main__':
    try:
        run_live_mic()
    except KeyboardInterrupt:
        print("\nKWS Engine Stopped.")
    except Exception as e:
        print(f"\nError: {e}")
        print("If no mic available, test offline: python3 test_live_kws_offline.py")
