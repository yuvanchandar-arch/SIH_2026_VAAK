# VaaK — Complete System Specification & Architecture Plan

> **Note**: This document serves as the permanent, self-contained reference and execution specification for the **VaaK (SIH 2026)** system. It documents the complete repo state, top-level architecture, stage-by-stage build plan, strict build order, metric mapping, and locked novelty statement.

---

## 0. Current Verification & Hardware Status

- **Gate 0 to 5c**: **PASSED ✅**
  - Val Recall: **93.00%** / FA: **0.03%**
  - Test Recall: **92.00%** / FA: **0.15%**
  - Full Streaming Audit (Val/Test): **100.00% Recall**, FA **0.03% (Val)** / **0.17% (Test)**
  - Model Flash Footprint: **46.35 KB** (`dracarys_kws.tflite`)
  - Model Dynamic RAM Arena: **64.41 KB** (Ping-Pong Buffer Reuse, Headroom = 191.59 KB)
- **Gate 6 (On-Device Pi 4 Hardware Audit)**: **PASSED ✅** (Verified on physical Raspberry Pi 4 B + INMP441 I2S mic)
  - CPU Mean: **2.1%** normalized across 4 cores (**8.2% raw**, Peak **2.5%** normalized / **10.0% raw**) — comfortably under the $<10\%$ target.
  - Process RSS Memory: **60.3 MB** (Informational).
  - Inference Latency: **0.80 ms avg** ($< 2.0$ ms target).
  - OpenBLAS thread cap locked in `pi_deploy/live_kws.py` (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`).

---

## 1. Where The Repo Stands Today

### `pi_deploy/` (Existing & Verified):
- `dracarys_kws.tflite`: **DONE** — 46.35 KB flash, 64.41 KB tensor arena.
- `features.py`: **DONE** — verified byte-identical train/inference `(49, 40)` log-mel feature extractor.
- `live_kws.py`: **PARTIAL** — real-time keyword detection works; Vosk section opens a socket but needs true continuous streaming post-wake PCM.
- `gate6_measure.py`: **DONE** — CPU & RAM measurement harness verified on physical hardware.
- `setup_pi.sh`: **DONE** — consolidated Raspberry Pi 4 I2S overlay configuration script.
- `requirements_pi.txt`: **DONE** — Pi 4 environment dependencies.
- `test_live_kws_offline.py`: **DONE** — full val/test streaming audit passed.

### NOT YET IN REPO (To Be Built):
1. `pi_deploy/asr_client.py`: Real streaming WebSocket ASR handoff.
2. `capacity_controller.py`: Adaptive model-capacity logic (Acoustic Condition Estimator + Controller).
3. `dracarys_small.tflite`: Companion low-capacity INT8 model (2 DS-CNN blocks instead of 4).
4. `command_parser.py`: Intent / slot / 20-command whitelist parser.
5. `speaker_id` server code: Vosk `SpkModel` integration for technician speaker verification on server.
6. `qa_log.py`: Hash-chained Q&A audit record generator.

---

## 2. Full Target Architecture (Top Level)

```
EDGE (Raspberry Pi 4)
INMP441 -> 16kHz mono -> 1s rolling window (200ms hop) -> 49x40 Log-Mel
  │
  ▼
Acoustic Condition Estimator (noise_score 0-1)
  │
  ▼
Capacity Controller (deterministic + hysteresis)
  ├───────────────────────────────────┐
  ▼                                   ▼
Small INT8 DS-CNN                 Large INT8 DS-CNN
(resident interpreter)            (resident interpreter)
  │                                   │
  └─────────────────┬─────────────────┘
                    ▼
           Dracarys probability
                    │
        ┌───────────┴───────────┐
     < 0.85                   >= 0.85
(reject, no-op)           (WAKE CONFIRMED)
                            │
                            ▼
              Stream post-wake PCM audio (WebSocket)
                            │
                            ▼
SERVER / CLOUD
Incoming post-wake audio stream (WebSocket)
  ├───────────────────────────────────┐
  ▼ (parallel)                        ▼ (parallel)
Vosk ASR                            Vosk SpkModel
  │                                   │
  ▼                                   ▼
Transcript                          Speaker Embedding Vector
  │                                   │
  │                                   ▼ Compare vs enrolled vectors
  │                                   (TECH_01 / TECH_02 / TECH_03 / UNKNOWN)
  │                                   │
  ▼                                   ▼
Intent Parser <───────────────────── Speaker Identity
  │
  ▼
Slot / Parameter Parser (e.g. value=36.5, unit=Celsius)
  │
  ▼
20-Command Whitelist Check
  ├───────────────────────────────────┐
  ▼                                   ▼
UNAUTHORIZED / UNKNOWN             AUTHORIZED
(log + reject, no execution)          │
                                      ▼
                           Execute Command / App Logic
                                      │
                                      ▼
                           Hash-Chained QA Log Entry
                           {timestamp, speaker, command, value, result, prev_hash, entry_hash}
                                      │
                                      ▼
                           Return to STANDBY (edge)
```

---

## 3. Stage-By-Stage Build-Out

### Stage A — Feature Extraction (existing, unchanged)
- File: `features.py`
- Produces `(49, 40)` Log-Mel representation from 1-second rolling window.
- Unchanged. Both small and large models consume this exact same tensor, ensuring scientific validity.

### Stage B — Acoustic Condition Estimator (new)
- File: part of `capacity_controller.py`
- Input: 49x40 Log-Mel frame.
- Output: `noise_score` (0–1) calculated via noise-floor estimate (energy in non-speech-band mel bins or spectral flatness — cheap, deterministic, no ML).

### Stage C — Capacity Controller (new)
- File: `capacity_controller.py`
- Deterministic controller combining `noise_score` with recent KWS confidence history + hysteresis:
  - `noise_score` LOW **AND** recent confidence STABLE $\rightarrow$ **SMALL**
  - `noise_score` HIGH **OR** recent confidence UNSTABLE $\rightarrow$ **LARGE**
- **Hysteresis**: switching SMALL $\rightarrow$ LARGE uses a lower noise threshold than LARGE $\rightarrow$ SMALL to prevent thrashing/oscillation on borderline audio.
- Output: `SMALL` or `LARGE` decision per 1-second analysis window (aligned with 200ms hop).

### Stage D — Dual Resident INT8 Interpreters (new)
- Files:
  - `dracarys_small.tflite`: trained from scratch (2 DS-CNN blocks instead of 4, same 3-class output, same speaker-disjoint dataset discipline).
  - `dracarys_kws.tflite`: existing 46.35 KB model (referred to conceptually as "large").
- **Critical Rule**: Both interpreters allocated **once at startup** and held resident in RAM:
  ```python
  # Startup (once):
  small_interp = tflite.Interpreter("dracarys_small.tflite")
  small_interp.allocate_tensors()
  large_interp = tflite.Interpreter("dracarys_kws.tflite")
  large_interp.allocate_tensors()

  # Per window:
  interp = small_interp if controller.decide(noise_score, confidence_history) == "SMALL" else large_interp
  result = interp.invoke_on(features)
  ```
  *Never reload an interpreter mid-stream; reload overhead erases compute savings.*

### Stage E — Combined RAM Gate (new gate)
- Measure combined dynamic RAM of both resident interpreters + shared runtime directly (`small_arena + large_arena + shared runtime overhead`).
- Target: comfortably under **256 KB**.

### Stage F — Four-Configuration Evaluation (proves the adaptive idea)
- Run evaluation against validation/test sets across 4 configurations:
  - **Config A**: Always Large (baseline, current system)
  - **Config B**: Always Small (establishes accuracy/cost floor)
  - **Config C**: Noise-only controller (tests if acoustic difficulty alone is useful)
  - **Config D**: Noise + confidence controller (proposed adaptive system)
- Measured per config: CPU%, mean inference latency, combined RAM, recall, false-activation rate, and **large-model invocation rate** (% of windows using large model).

### Stage G — Real ASR Streaming
- File: `pi_deploy/asr_client.py`
- Upon wake keyword detection ($\ge 0.85$), open/reuse WebSocket to Vosk server and stream 16-bit mono PCM continuously in real-time (not batch-send).
- Auto-stop after fixed window or trailing silence.
- **Latency Timestamps**:
  - `T1`: keyword-end
  - `T2`: WebSocket connection established
  - `T3`: first audio byte received by server
  - `T4`: final transcript received
  - **$T3 - T1$** is the critical edge-to-server latency metric.

### Stage H — Server-Side ASR + Speaker ID (parallel)
- Two concurrent consumers of the post-wake audio stream on the server:
  1. **Vosk ASR** $\rightarrow$ transcript
  2. **Vosk SpkModel** $\rightarrow$ speaker embedding vector compared vs enrolled technician vectors (`TECH_01`, `TECH_02`, `TECH_03`, `UNKNOWN`).
- Rejects ambiguous matches.
- Keeps edge memory budget clean (ASR & Speaker ID run entirely server-side).

### Stage I — Command Parser (new)
- File: `command_parser.py`
- Intent Parser + Slot Parser + 20-command whitelist check.
- Deterministic parsing (normalizing paraphrases like *"please go to the next step"* / *"move to the next procedure step"* $\rightarrow$ `NEXT_STEP`).

### Stage J — QA / Audit Log (new)
- File: `qa_log.py`
- Hash-chained structured QA record:
  ```json
  {
    "timestamp": "10:42:18",
    "speaker": "TECH_02",
    "command": "LOG_TEMPERATURE",
    "value": "36.5 C",
    "result": "ACCEPTED",
    "prev_hash": "...",
    "entry_hash": "..."
  }
  ```

---

## 4. Strict Build Order

1. **Stage G (real ASR streaming)** — oldest open gap, blocks Stage F's latency numbers.
2. **Real Gate 6 on physical hardware** — **COMPLETED & VERIFIED** (2.1% CPU, 60.3 MB RSS, 0.80 ms latency).
3. **Stage D+E (small model + dual interpreters + combined RAM gate)**.
4. **Stage B+C (estimator + controller)**.
5. **Stage F (four-configuration experiment)**.
6. **Stage H (server-side ASR + speaker ID)**.
7. **Stage I+J (command parser, QA log)** — lowest technical risk, done last.

---

## 5. Locked Novelty Statement

> "VaaK implements acoustic-condition-driven model-capacity selection for a custom single-keyword edge detector: a deterministic noise/confidence controller selects between two independently trained and quantized INT8 DS-CNNs, executing exactly one per audio window on commodity hardware, with both interpreters held resident to avoid reload overhead. This differs from precision-adaptive KWS accelerators such as EERA-KWS — which reconfigure numerical precision inside custom 28nm silicon — by instead selecting computational capacity within a standard INT8 inference stack, making the technique reproducible on off-the-shelf edge hardware and empirically measured across a four-configuration comparison rather than asserted."
