import os

import bigvgan
import librosa
import torch
import torchaudio
from bigvgan import load_hparams_from_json, BigVGAN
from bigvgan.meldataset import get_mel_spectrogram, mel_spectrogram
from huggingface_hub import snapshot_download
from torch import nn

from vocos_vocoder import get_vocos_mel_spectrogram


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

def process_audio_folder(input_folder, output_folder, model_type="bigvgan"):
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Load pretrained BigVGAN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_sample_rate = 24000

    # Load model
    cache_dir = snapshot_download("cckm/bigvgan_melspec",
                                  allow_patterns=[
                                      "config.json",
                                      "g_00400000"
                                  ])
    h = load_hparams_from_json(os.path.join(cache_dir, "config.json"))
    model = BigVGAN(h, use_cuda_kernel=False)
    checkpoint_dict = torch.load(os.path.join(cache_dir, "g_00400000"), map_location='cpu')
    model.load_state_dict(checkpoint_dict["generator"])

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
            mel_spec = MelSpec()(wav)

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
    output_folder_24k = "bigvgan_decode_vocos_output"  # Output folder for 24kHz

    # Process with BigVGAN 24kHz
    process_audio_folder(input_folder, output_folder_24k, model_type="bigvgan")
    # Process with BigVGAN 44kHz
