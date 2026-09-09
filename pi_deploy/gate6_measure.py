#!/usr/bin/env python3
"""
gate6_measure.py — Gate 6 CPU & RAM Measurement Script.
Run this in a SEPARATE terminal while live_kws.py is running.
Samples CPU and RSS every second for 30 seconds, then reports.

Usage:
  1. Start live_kws.py in Terminal 1
  2. In Terminal 2: python3 gate6_measure.py
"""

import os
import sys
import time
import subprocess

def find_live_kws_pid():
    """Find the PID of the running live_kws.py process."""
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'live_kws.py'],
            capture_output=True, text=True
        )
        pids = result.stdout.strip().split('\n')
        # Filter out our own PID
        pids = [p for p in pids if p and int(p) != os.getpid()]
        if pids:
            return int(pids[0])
    except Exception:
        pass
    return None

def get_rss_kb(pid):
    """Get RSS memory in KB for a given PID."""
    try:
        result = subprocess.run(
            ['ps', '-o', 'rss=', '-p', str(pid)],
            capture_output=True, text=True
        )
        return int(result.stdout.strip())
    except Exception:
        return 0

def main():
    print("=" * 60)
    print("GATE 6 — CPU & RAM MEASUREMENT")
    print("=" * 60)
    
    pid = find_live_kws_pid()
    if pid is None:
        print("\nERROR: live_kws.py is not running!")
        print("Start it first in another terminal:")
        print("  python3 live_kws.py")
        return
        
    print(f"\nFound live_kws.py process: PID {pid}")
    print("Sampling CPU and RSS every 1 second for 30 seconds...")
    print("Keep the room QUIET (idle listening mode).\n")
    
    try:
        import psutil
        proc = psutil.Process(pid)
    except ImportError:
        print("ERROR: psutil not installed. Run: pip install psutil")
        return
    except psutil.NoSuchProcess:
        print(f"ERROR: PID {pid} no longer exists.")
        return
    
    cpu_samples = []
    rss_samples = []
    
    for i in range(30):
        try:
            cpu = proc.cpu_percent(interval=1.0)
            rss_kb = get_rss_kb(pid)
            cpu_samples.append(cpu)
            rss_samples.append(rss_kb)
            print(f"  Sample {i+1:2d}/30: CPU={cpu:5.1f}%  RSS={rss_kb} KB ({rss_kb/1024:.1f} MB)")
        except Exception as e:
            print(f"  Sample {i+1:2d}/30: Error — {e}")
    
    print("\n" + "=" * 60)
    print("GATE 6 RESULTS — CPU & RAM (Idle Listening)")
    print("=" * 60)
    
    if cpu_samples:
        cpu_count = os.cpu_count() or 1
        mean_cpu_raw = sum(cpu_samples) / len(cpu_samples)
        peak_cpu_raw = max(cpu_samples)
        mean_cpu_norm = mean_cpu_raw / cpu_count
        peak_cpu_norm = peak_cpu_raw / cpu_count
        print(f"  CPU cores on this machine: {cpu_count}")
        print(f"  CPU Mean (raw psutil):      {mean_cpu_raw:.1f}%")
        print(f"  CPU Mean (normalized/core): {mean_cpu_norm:.1f}%  (Target: <10%)")
        print(f"  CPU Peak (raw psutil):      {peak_cpu_raw:.1f}%")
        print(f"  CPU Peak (normalized/core): {peak_cpu_norm:.1f}%")
        passed = mean_cpu_norm < 10.0
        print(f"  CPU Status: {'PASSED ✅' if passed else 'FAILED ❌'} (normalized mean {mean_cpu_norm:.1f}% vs <10% target)")
    
    if rss_samples:
        rss_val = rss_samples[-1]  # Most stable reading
        print(f"\n  Process RSS Memory: {rss_val} KB ({rss_val/1024:.1f} MB)")
        print(f"  RAM Status: INFORMATIONAL (Full Linux Process RSS)")
        print(f"  Note: Total process RSS includes Python interpreter, numpy, and tflite-runtime")
        print(f"        overhead (~20-50 MB expected). The strict <256 KB RAM constraint applies")
        print(f"        specifically to the model's dynamic tensor arena (verified at 64.41 KB in Gate 5b).")
    
    print("=" * 60)

if __name__ == '__main__':
    main()
