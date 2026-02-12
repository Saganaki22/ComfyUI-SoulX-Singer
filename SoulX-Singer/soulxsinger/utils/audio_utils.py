import torch
import soundfile as sf
import numpy as np
from scipy import signal


def load_wav(wav_path: str, sample_rate: int):
    """Load wav file and resample to target sample rate.

    Args:
        wav_path (str): Path to wav file.
        sample_rate (int): Target sample rate.

    Returns:
        torch.Tensor: Waveform tensor with shape (1, T).
    """
    # Load with soundfile
    waveform, sr = sf.read(wav_path, dtype='float32')
    
    # Convert to torch tensor
    waveform = torch.from_numpy(waveform)
    
    # Ensure shape is (channels, samples)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T
    
    # Resample if needed
    if sr != sample_rate:
        # Use scipy for resampling
        num_samples = int(waveform.shape[1] * sample_rate / sr)
        waveform_np = waveform.numpy()
        resampled = signal.resample(waveform_np, num_samples, axis=1)
        waveform = torch.from_numpy(resampled.astype(np.float32))
    
    # Convert to mono if needed
    if len(waveform.shape) > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    return waveform

        
