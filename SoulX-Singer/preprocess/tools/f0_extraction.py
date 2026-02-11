# https://github.com/Dream-High/RMVPE
import math
import time
import librosa
import numpy as np
from librosa.filters import mel
from scipy.interpolate import interp1d

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super(ConvBlockRes, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x):
        if not hasattr(self, "shortcut"):
            return self.conv(x) + x
        else:
            return self.conv(x) + self.shortcut(x)


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super(ResEncoderBlock, self).__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        else:
            return x


class Encoder(nn.Module):
    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01):
        super(Encoder, self).__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for i in range(self.n_encoders):
            self.layers.append(
                ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=momentum)
            )
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for layer in self.layers:
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super(Intermediate, self).__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList()
        self.layers.append(ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum))
        for i in range(self.n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super(ResDecoderBlock, self).__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for conv2 in self.conv2:
            x = conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for i in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum)
            )
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super(DeepUnet, self).__init__()
        self.encoder = Encoder(in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class E2E(nn.Module):
    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super(E2E, self).__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * 128, 256, n_gru),
                nn.Linear(512, 360),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * 128, 360),
                nn.Dropout(0.25),
                nn.Sigmoid()
            )

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x



class MelSpectrogram(torch.nn.Module):
    def __init__(self, is_half, n_mel_channels, sampling_rate, win_length, hop_length, 
                 n_fft=None, mel_fmin=0, mel_fmax=None, clamp=1e-5):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp
        self.is_half = is_half

    def forward(self, audio, keyshift=0, speed=1, center=True):
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        
        keyshift_key = str(keyshift) + "_" + str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(audio.device)
        
        fft = torch.stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=self.hann_window[keyshift_key],
            center=center,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new
        
        mel_output = torch.matmul(self.mel_basis, magnitude)
        if self.is_half:
            mel_output = mel_output.half()
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec



class RMVPE:
    def __init__(self, model_path: str, is_half, device=None):
        self.is_half = is_half
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device) if isinstance(device, str) else device
        
        self.mel_extractor = MelSpectrogram(
            is_half=is_half,
            n_mel_channels=128,
            sampling_rate=16000,
            win_length=1024,
            hop_length=160,
            n_fft=None,
            mel_fmin=30,
            mel_fmax=8000
        ).to(self.device)
        
        model = E2E(n_blocks=4, n_gru=1, kernel_size=(2, 2))
        ckpt = torch.load(model_path, map_location=self.device)
        model.load_state_dict(ckpt)
        model.eval()
        
        if is_half:
            model = model.half()
        else:
            model = model.float()
        
        self.model = model.to(self.device)
        
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))  # 368

    def mel2hidden(self, mel):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            if n_pad > 0:
                mel = F.pad(mel, (0, n_pad), mode="constant")
            mel = mel.half() if self.is_half else mel.float()
            hidden = self.model(mel)
            return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03):
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        return f0

    def infer_from_audio(self, audio, thred=0.03):
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)
        
        mel = self.mel_extractor(audio.float().to(self.device).unsqueeze(0), center=True)
        hidden = self.mel2hidden(mel)
        hidden = hidden.squeeze(0).cpu().numpy()
        
        if self.is_half:
            hidden = hidden.astype("float32")
        
        f0 = self.decode(hidden, thred=thred)
        return f0

    def to_local_average_cents(self, salience, thred=0.05):
        center = np.argmax(salience, axis=1)
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center += 4
        
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5
        
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[:, starts[idx]:ends[idx]][idx])
            todo_cents_mapping.append(self.cents_mapping[starts[idx]:ends[idx]])
        
        todo_salience = np.array(todo_salience)
        todo_cents_mapping = np.array(todo_cents_mapping)
        product_sum = np.sum(todo_salience * todo_cents_mapping, 1)
        weight_sum = np.sum(todo_salience, 1)
        devided = product_sum / weight_sum
        
        maxx = np.max(salience, axis=1)
        devided[maxx <= thred] = 0
        
        return devided

class F0Extractor:
    """Extract frame-level f0 from singing voice.

    Wrapper around an RMVPE network that:
        1) loads the checkpoint once in ``__init__``
        2) exposes a simple :py:meth:`process` API and optionally saves ``*_f0.npy``.
    """
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        *,
        is_half: bool = False,
        input_sr: int = 16000,
        target_sr: int = 24000,
        hop_size: int = 480,
        max_duration: float = 300,
        thred: float = 0.03,
        verbose: bool = True,
    ):
        """Initialize the f0 extractor.

        Args:
            model_path: Path to RMVPE checkpoint.
            device: Torch device string, e.g. ``"cuda:0"`` / ``"cpu"``.
            is_half: Whether to run the model in fp16.
            input_sr: Input resample rate used by RMVPE frontend.
            target_sr: Target sample rate for the output f0 grid.
            hop_size: Target hop size for the output f0 grid.
            max_duration: Max duration (seconds) for interpolation grid.
            thred: Voicing threshold used when decoding salience.
            verbose: Whether to print verbose logs.
        """
        self.model_path = model_path
        self.input_sr = input_sr
        self.target_sr = target_sr
        self.hop_size = hop_size
        self.max_duration = max_duration
        self.thred = thred

        self.verbose = verbose

        self.model = RMVPE(model_path, is_half=is_half, device=device)

        if self.verbose:
            print(
                "[f0 extraction] init success:",
                f"device={device}",
                f"model_path={model_path}",
                f"is_half={is_half}",
                f"input_sr={input_sr}",
                f"target_sr={target_sr}",
                f"hop_size={hop_size}",
                f"thred={thred}",
            )

    @staticmethod
    def interpolate_f0(
        f0_16k: np.ndarray,
        original_length: int,
        original_sr: int,
        *,
        target_sr: int = 48000,
        hop_size: int = 256,
        max_duration: float = 20.0,
    ) -> np.ndarray:
        """Interpolate f0 from RMVPE's 16k hop grid to target mel hop grid."""
        mel_target_sr = target_sr
        mel_hop_size = hop_size
        mel_max_duration = max_duration

        batch_max_length = int(mel_max_duration * mel_target_sr / mel_hop_size)
        duration_in_seconds = original_length / original_sr
        effective_target_length = int(duration_in_seconds * mel_target_sr)
        original_frames = math.ceil(effective_target_length / mel_hop_size)
        target_frames = min(original_frames, batch_max_length)

        rmvpe_hop = 160
        t_16k = np.arange(len(f0_16k)) * (rmvpe_hop / 16000.0)
        t_target = np.arange(target_frames) * (mel_hop_size / float(mel_target_sr))

        if len(f0_16k) > 0:
            f_interp = interp1d(
                t_16k,
                f0_16k,
                kind="linear",
                bounds_error=False,
                fill_value=0.0,
                assume_sorted=True,
            )
            f0 = f_interp(t_target)
        else:
            f0 = np.zeros(target_frames)

        if len(f0) != target_frames:
            f0 = (
                f0[:target_frames]
                if len(f0) > target_frames
                else np.pad(f0, (0, target_frames - len(f0)), "constant")
            )

        return f0

    def process(self, audio_path: str, *, f0_path: str | None = None, verbose: Optional[bool] = None) -> np.ndarray:
        """Run f0 extraction for a single wav.

        Args:
            audio_path: Path to the input wav file.
            f0_path: if is not None, save the f0 data to this path.
            verbose: Override instance-level verbose flag for this call.

        Returns:
            np.ndarray: shape ``[T]``, f0 in Hz (0 for unvoiced).
        """
        verbose = self.verbose if verbose is None else verbose
        if verbose:
            print(f"[f0 extraction] process: start: {audio_path}")
            t0 = time.time()

        audio, _ = librosa.load(audio_path, sr=self.input_sr)
        f0_16k = self.model.infer_from_audio(audio, thred=self.thred)
        f0 = self.interpolate_f0(
            f0_16k,
            original_length=audio.shape[-1],
            original_sr=self.input_sr,
            target_sr=self.target_sr,
            hop_size=self.hop_size,
            max_duration=self.max_duration,
        )

        if verbose:
            dt = time.time() - t0
            voiced_ratio = float(np.mean(f0 > 0)) if len(f0) else 0.0
            print(
                "[f0 extraction] process: done:",
                f"frames={len(f0)}",
                f"voiced_ratio={voiced_ratio:.3f}",
                f"time={dt:.3f}s",
            )
        if f0_path is not None:
            np.save(f0_path, f0)

        return f0


if __name__ == "__main__":
    model_path = (
        "pretrained_models/rmvpe/rmvpe.pt"
    )
    audio_path = "./outputs/transcription/test.wav"

    pe = F0Extractor(
        model_path,
        device="cuda",
    )
    f0 = pe.process(audio_path)
