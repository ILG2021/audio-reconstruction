import os
import torch
import torchaudio
from transformers import MimiModel, AutoFeatureExtractor
from datasets import Audio

def process_audio_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the Mimi model and feature extractor
    model = MimiModel.from_pretrained("kyutai/mimi")
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    model.eval()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Supported audio file extensions
    audio_extensions = ('.wav', '.mp3', '.flac')

    # Target sample rate for Mimi
    target_sample_rate = feature_extractor.sampling_rate  # Typically 24 kHz for Mimi

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

            # Pre-process audio using the feature extractor
            inputs = feature_extractor(
                raw_audio=waveform.squeeze(0).numpy(),  # Convert to numpy array
                sampling_rate=target_sample_rate,
                return_tensors="pt"
            )

            # Move inputs to the same device as the model
            input_values = inputs["input_values"].to(device)
            padding_mask = inputs.get("padding_mask", None)
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)

            # Encode and decode the audio signal
            with torch.inference_mode():
                # Explicit encode and decode
                encoder_outputs = model.encode(input_values, padding_mask)
                reconstructed_waveform = model.decode(encoder_outputs.audio_codes, padding_mask)[0]
                # Alternatively, use forward pass: reconstructed_waveform = model(input_values, padding_mask).audio_values

            # Save reconstructed audio
            torchaudio.save(output_path, reconstructed_waveform.squeeze(0).cpu(), sample_rate)
            print(f"Saved reconstructed audio to {output_path}")

if __name__ == "__main__":
    # Example usage
    input_folder = "origin"  # Replace with your input audio folder path
    output_folder = "mimi_output"  # Output folder for Mimi
    process_audio_folder(input_folder, output_folder)