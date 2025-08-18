import os
import torch
import torchaudio
from encodec import EncodecModel

def process_audio_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load pre-trained Encodec model (24kHz bandwidth)
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)  # Set bandwidth, e.g., 6.0 kbps
    model.eval()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Supported audio file extensions
    audio_extensions = ('.wav', '.mp3', '.flac')

    # Iterate through all audio files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(audio_extensions):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            print(f"Processing {filename}...")

            # Load audio file
            waveform, sample_rate = torchaudio.load(input_path)

            # Ensure audio is mono and resampled to 24kHz (Encodec's expected sample rate)
            if sample_rate != 24000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)
                waveform = resampler(waveform)
                sample_rate = 24000

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Move waveform to the same device as the model
            waveform = waveform.to(device)

            # Add batch dimension if needed (Encodec expects [batch, channels, time])
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)

            # Encode audio to latent representation
            with torch.no_grad():
                encoded_frames = model.encode(waveform)

            # Decode back to waveform
            reconstructed_waveform = model.decode(encoded_frames)

            # Remove batch dimension and ensure proper shape
            reconstructed_waveform = reconstructed_waveform.squeeze(0)

            # Detach tensor and move to CPU for saving
            reconstructed_waveform = reconstructed_waveform.detach().cpu()

            # Save reconstructed audio
            torchaudio.save(output_path, reconstructed_waveform, sample_rate)
            print(f"Saved reconstructed audio to {output_path}")

if __name__ == "__main__":
    # Example usage
    input_folder = "origin"  # Replace with your input audio folder path
    output_folder = "encodec-output"  # Output folder path
    process_audio_folder(input_folder, output_folder)