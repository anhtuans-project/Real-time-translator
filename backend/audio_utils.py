import numpy as np
import asyncio

def resample_pcm16(data: bytes, src_sr: int, dst_sr: int) -> bytes:
    """
    Resamples PCM16 audio data from src_sr to dst_sr.
    Uses linear interpolation for simplicity if soxr is not available.
    """
    if src_sr == dst_sr:
        return data

    # Convert bytes to float32
    audio_float = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    # Linear interpolation
    duration = len(audio_float) / src_sr
    num_samples = int(duration * dst_sr)

    # Create a new time axis
    old_indices = np.arange(len(audio_float))
    new_indices = np.linspace(0, len(audio_float) - 1, num_samples)

    # Interpolate
    resampled_float = np.interp(new_indices, old_indices, audio_float)

    # Convert back to PCM16
    resampled_int16 = (resampled_float * 32767).astype(np.int16)
    return resampled_int16.tobytes()

async def float32_to_pcm16(float_array: np.ndarray) -> bytes:
    """
    Converts a float32 numpy array to PCM16 bytes.
    """
    # Clamp to [-1.0, 1.0]
    clamped = np.clip(float_array, -1.0, 1.0)
    return (clamped * 32767).astype(np.int16).tobytes()
