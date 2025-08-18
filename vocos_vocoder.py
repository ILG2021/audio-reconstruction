import os

import torch
import torchaudio
from torch import nn
from vocos import Vocos


def get_vocos_mel_spectrogram(
        waveform,
        n_fft=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        hop_length=256,
        win_length=1024,
):
    mel_stft = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mel_channels,
        power=1,
        center=True,
        normalized=False,
        norm=None,
    ).to(waveform.device)
    if len(waveform.shape) == 3:
        waveform = waveform.squeeze(1)  # 'b 1 nw -> b nw'

    assert len(waveform.shape) == 2

    mel = mel_stft(waveform)
    mel = mel.clamp(min=1e-5).log()
    return mel


class MelSpec(nn.Module):
    def __init__(
            self,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=100,
            target_sample_rate=24_000,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate
        self.extractor = get_vocos_mel_spectrogram
        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, wav):
        if self.dummy.device != wav.device:
            self.to(wav.device)

        mel = self.extractor(
            waveform=wav,
            n_fft=self.n_fft,
            n_mel_channels=self.n_mel_channels,
            target_sample_rate=self.target_sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        return mel


def process_audio_folder(input_folder, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 加载预训练的 Vocos 模型
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

    # 支持的音频文件扩展名
    audio_extensions = ('.wav', '.mp3', '.flac')

    # 遍历输入文件夹中的所有音频文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(audio_extensions):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            print(f"Processing {filename}...")

            # 加载音频文件
            waveform, sample_rate = torchaudio.load(input_path)

            # 确保音频是单声道并重采样到模型期望的采样率（24kHz）
            if sample_rate != 24000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)
                waveform = resampler(waveform)
                sample_rate = 24000

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 从 Mel 频谱图重构音频
            reconstructed_waveform  = vocos.decode(MelSpec()(waveform))

            # 保存重构的音频
            torchaudio.save(output_path, reconstructed_waveform, sample_rate)
            print(f"Saved reconstructed audio to {output_path}")


if __name__ == "__main__":
    # 示例用法
    input_folder = "origin"  # 替换为你的输入音频文件夹路径
    output_folder = "vocos-output"  # 输出文件夹路径
    process_audio_folder(input_folder, output_folder)
