import os

import bigvgan
import librosa
import torch
import torchaudio
from bigvgan.meldataset import get_mel_spectrogram, mel_spectrogram
from torch import nn


class MelSpec(nn.Module):
    def __init__(
            self,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=100,
            target_sample_rate=24000,
            mel_spec_type="bigvgan",
    ):
        super().__init__()
        assert mel_spec_type in ["bigvgan", "bigvgan_44k"], print("Supported mel backends: bigvgan or bigvgan_44k")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate
        self.mel_spec_type = mel_spec_type

        if mel_spec_type == "bigvgan":
            self.n_fft = 1024
            self.hop_length = 256
            self.win_length = 1024
            self.n_mel_channels = 100
            self.target_sample_rate = 24000
            self.f_max = 12000
        elif mel_spec_type == "bigvgan_44k":
            self.n_fft = 2048
            self.hop_length = 512
            self.win_length = 2048
            self.n_mel_channels = 128
            self.target_sample_rate = 44100
            self.f_max = 22050

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, wav):
        if self.dummy.device != wav.device:
            self.to(wav.device)

        # Ensure wav is in the correct shape: [B, T_time]
        if len(wav.shape) == 3:
            wav = wav.squeeze(1)  # 'b 1 nw -> b nw'
        assert len(wav.shape) == 2, f"Expected 2D waveform tensor, got shape {wav.shape}"

        mel = mel_spectrogram(
            wav,
            n_fft=self.n_fft,
            hop_size=self.hop_length,
            win_size=self.win_length,
            num_mels=self.n_mel_channels,
            sampling_rate=self.target_sample_rate,
            fmax=self.f_max,
            fmin=0,
            center=True,
        ).to(wav.device)

        return mel


def process_audio_folder(input_folder, output_folder, model_type="bigvgan"):
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Load pretrained BigVGAN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "bigvgan":
        model_path = "nvidia/bigvgan_v2_24khz_100band_256x"
        target_sample_rate = 24000
    elif model_type == "bigvgan_44k":
        model_path = "nvidia/bigvgan_v2_44khz_128band_512x"
        target_sample_rate = 44100
    else:
        raise ValueError("Model type must be 'bigvgan' or 'bigvgan_44k'")

    # Load model
    model = bigvgan.BigVGAN.from_pretrained(model_path, use_cuda_kernel=False)
    model.remove_weight_norm()
    model.eval().to(device)

    # Supported audio file extensions
    audio_extensions = ('.wav', '.mp3', '.flac')

    # Process all audio files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(audio_extensions):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            print(f"Processing {filename}...")

            # Load audio file
            wav, sr = librosa.load(input_path, sr=target_sample_rate, mono=True)
            wav = torch.FloatTensor(wav).unsqueeze(0).to(device)  # [1, T_time]

            # Compute mel spectrogram
            mel_spec = MelSpec(mel_spec_type=model_type)(wav)

            # Generate waveform from mel
            with torch.no_grad():
                reconstructed_waveform = model(mel_spec)  # [1, 1, T_time]

            # Convert to int16 for saving
            wav_gen_int16 = (reconstructed_waveform.squeeze(0).cpu().numpy() * 32767.0).astype('int16')

            # Save reconstructed audio
            torchaudio.save(output_path, torch.tensor(wav_gen_int16), target_sample_rate)
            print(f"Saved reconstructed audio to {output_path}")


if __name__ == "__main__":
    # Example usage
    input_folder = "origin"  # Replace with your input audio folder path
    output_folder_24k = "bigvgan_24k_output"  # Output folder for 24kHz
    output_folder_44k = "bigvgan_44k_output"  # Output folder for 44kHz

    # Process with BigVGAN 24kHz
    process_audio_folder(input_folder, output_folder_24k, model_type="bigvgan")
    # Process with BigVGAN 44kHz
    process_audio_folder(input_folder, output_folder_44k, model_type="bigvgan_44k")
