import time
import numpy as np
import av
from djitellopy import Tello

def benchmark_tello(n_samples=100, warmup_frames=5):
    """
    Benchmark Tello telemetry RTT and video frame latency/FPS.

    Args:
        n_samples (int): Number of samples to measure for telemetry and video.
        warmup_frames (int): Number of initial frames to discard as warm-up.
    """
    # --- Connect to the drone ---
    tello = Tello()
    tello.connect()

    # --- Telemetry RTT via send_read_command ---
    telemetry_rtts = []
    for _ in range(n_samples):
        t0 = time.time()
        _ = tello.send_read_command("battery?")
        t1 = time.time()
        telemetry_rtts.append((t1 - t0) * 1000)  # in ms

    # --- Video via PyAV demux + decode ---
    tello.streamon()
    address = tello.get_udp_video_address()
    container = av.open(address, timeout=(5, None))

    # Warm-up: discard a few frames
    for _ in range(warmup_frames):
        for packet in container.demux(video=0):
            for _ in packet.decode():
                break
            break

    video_latencies = []
    timestamps = []
    t_prev = None

    # Measure n_samples frames
    for _ in range(n_samples):
        # Get next frame (blocks until one arrives)
        for packet in container.demux(video=0):
            frames = packet.decode()
            if not frames:
                continue
            _frame = frames[0]
            break

        t_now = time.time()
        if t_prev is not None:
            video_latencies.append((t_now - t_prev) * 1000)  # in ms
            timestamps.append(t_now)
        t_prev = t_now

    # --- Compute statistics ---
    tel_mean = np.mean(telemetry_rtts)
    tel_std  = np.std(telemetry_rtts)
    vid_mean = np.mean(video_latencies)
    vid_std  = np.std(video_latencies)
    fps      = 1.0 / np.mean(np.diff(timestamps))             # FPS via timestamps
    tel_freq = 1000.0 / tel_mean                              # Telemetry frequency in Hz
    vid_freq = 1000.0 / vid_mean                              # Video frequency in Hz

    # --- Print results ---
    print("=== Communication Latency Analysis ===")
    print(f"Telemetry RTT:        μ = {tel_mean:.1f} ms, σ = {tel_std:.1f} ms over {n_samples} samples")
    print(f"Telemetry frequency:           ≈ {tel_freq:.2f} Hz")
    print(f"Video latency:        μ = {vid_mean:.1f} ms, σ = {vid_std:.1f} ms")
    print(f"Video FPS (timestamps):        ≈ {fps:.1f}")
    print(f"Video frequency (1/mean latency): ≈ {vid_freq:.2f} Hz")

    # --- Cleanup ---
    tello.streamoff()
    tello.end()

if __name__ == "__main__":
    # Example: 100 samples, discard first 5 warm-up frames
    benchmark_tello(n_samples=250, warmup_frames=30)
