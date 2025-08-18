import os
import torch
import torchaudio
from snac import SNAC
from audiotools import AudioSignal

def process_audio_folder(input_folder, output_folder, model_type="24khz"):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the specified SNAC model (24kHz or 44kHz)
    model_name = f"hubertsiuzdak/snac_{model_type}"
    model = SNAC.from_pretrained(model_name)
    model.eval()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Supported audio file extensions
    audio_extensions = ('.wav', '.mp3', '.flac')

    # Target sample rate based on model type
    target_sample_rate = 24000 if model_type == "24khz" else 44100

    # Iterate through all audio files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(audio_extensions):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            print(f"Processing {filename}...")

            # Load audio file
            waveform, sample_rate = torchaudio.load(input_path)

            # Ensure audio is mono and resampled to the target sample rate
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                waveform = resampler(waveform)
                sample_rate = target_sample_rate

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Ensure waveform has the shape [1, 1, T] as expected by SNAC
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)  # Add batch dimension

            # Move waveform to the same device as the model
            waveform = waveform.to(device)

            # Encode and decode the audio signal
            with torch.inference_mode():
                reconstructed_waveform, codes = model(waveform)

            # Save reconstructed audio
            torchaudio.save(output_path, reconstructed_waveform.squeeze(0).cpu(), sample_rate)
            print(f"Saved reconstructed audio to {output_path}")

if __name__ == "__main__":
    # Example usage
    input_folder = "origin"  # Replace with your input audio folder path
    output_folder_24khz = "snac_24khz_output"  # Output folder for SNAC-24k
    output_folder_44khz = "snac_44khz_output"  # Output folder for SNAC-44k

    # Process with SNAC-24k
    process_audio_folder(input_folder, output_folder_24khz, model_type="24khz")

    # Process with SNAC-44k
    process_audio_folder(input_folder, output_folder_44khz, model_type="44khz")