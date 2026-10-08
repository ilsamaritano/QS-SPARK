"""
Raw IQ -> spectrogram PNG pipeline. Mirrors the PLASyR repo's two-stage
process (pre_processingBurst.py + iqToSpectrograms.py) but keeps everything
in memory between stages instead of round-tripping burst-segmented IQ
through disk: this machine has ~29 GB free, and the intermediate burst
files would not fit alongside the spectrogram outputs and raw captures.
"""
import os
import sys
import numpy as np
from PIL import Image
from scipy.fft import fft, fftshift
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


def load_capped(path, max_samples=cfg.MAX_RAW_SAMPLES_PER_FILE):
    return np.fromfile(path, dtype=np.complex64, count=max_samples)


def magnitude_threshold(iq, alpha=cfg.ALPHA):
    mag = np.abs(iq)
    tau = alpha * (mag.mean() + 2 * mag.std())
    return iq[mag > tau]


def burst_segment(iq, w=cfg.BURST_WINDOW, t_min=cfg.BURST_T_MIN,
                   energy_thresh_factor=cfg.BURST_ENERGY_THRESH_FACTOR):
    """Identical logic to pre_processingBurst.burst_segment (PLASyR repo)."""
    if len(iq) == 0:
        return []
    energy = np.abs(iq) ** 2
    # O(N) sliding-window mean (np.convolve with a w=50k box kernel is
    # O(N*w) and does not finish in reasonable time on 10^8-sample arrays).
    smoothed = uniform_filter1d(energy, size=w, mode="nearest")
    threshold = energy_thresh_factor * np.mean(smoothed)
    mask = (smoothed > threshold).astype(np.int8)
    diff = np.diff(mask, prepend=mask[0], append=mask[-1])
    rising = np.where(diff == 1)[0]
    falling = np.where(diff == -1)[0]

    bursts = []
    for start, end in zip(rising, falling):
        if end - start < t_min:
            continue
        burst = iq[start:end]
        rms = np.sqrt(np.mean(np.abs(burst) ** 2))
        if rms > 0:
            burst = burst / rms
        bursts.append(burst)

    if not bursts:
        rms = np.sqrt(np.mean(np.abs(iq) ** 2))
        bursts = [iq / rms if rms > 0 else iq]
    return bursts


def compute_spectrum_frame(chunk, fft_size=cfg.FFT_SIZE):
    spectrum = np.abs(fftshift(fft(chunk, fft_size)))
    spectrum_db = 20 * np.log10(spectrum + 1e-6)
    spectrum_db -= spectrum_db.min()
    mx = spectrum_db.max()
    if mx > 0:
        spectrum_db /= mx
    return spectrum_db * 255


def burst_to_images(burst, fft_size=cfg.FFT_SIZE, num_stacks=cfg.NUM_STACKS,
                     image_size=cfg.IMAGE_SIZE, max_images=None):
    """Stack consecutive FFT frames row-wise into 224x224 grayscale images."""
    num_chunks = len(burst) // fft_size
    num_images = num_chunks // num_stacks
    if num_images == 0:
        return []
    if max_images is not None:
        num_images = min(num_images, max_images)

    images = []
    for i in range(num_images):
        canvas = np.zeros((num_stacks, fft_size), dtype=np.float32)
        for j in range(num_stacks):
            start = (i * num_stacks + j) * fft_size
            canvas[j, :] = compute_spectrum_frame(burst[start:start + fft_size], fft_size)
        canvas -= canvas.min()
        mx = canvas.max()
        if mx > 0:
            canvas /= mx
        canvas *= 255
        img = Image.fromarray(canvas.astype(np.uint8)).resize(image_size, Image.BILINEAR)
        images.append(img)
    return images


def process_file_to_images(raw_path, max_images=cfg.MAX_IMAGES_PER_FILE):
    """Full pipeline for one raw capture file -> list of PIL images."""
    iq = load_capped(raw_path)
    if len(iq) == 0:
        return []
    filtered = magnitude_threshold(iq)
    bursts = burst_segment(filtered)
    bursts = [b for b in bursts if len(b) >= cfg.MIN_BURST_SAMPLES_FOR_IMAGE]
    if not bursts:
        return []

    images = []
    # Spread the image budget across bursts so a single huge burst doesn't
    # starve the others, while still respecting the per-file cap.
    per_burst_cap = max(1, max_images // max(1, len(bursts)))
    for burst in bursts:
        images.extend(burst_to_images(burst, max_images=per_burst_cap))
        if len(images) >= max_images:
            break
    return images[:max_images]


def save_images(images, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(out_dir, f"{prefix}_img{i:04d}.png"))
    return len(images)
