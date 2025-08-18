import os
import torch
import torchaudio
import dac
from audiotools import AudioSignal

def process_audio_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Download and load the 44kHz DAC model
    model_path = dac.utils.download(model_type="44khz")
    model = dac.DAC.load(model_path)
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

            # Ensure audio is mono and resampled to 44.1kHz (DAC's expected sample rate for this model)
            target_sample_rate = 44100
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                waveform = resampler(waveform)
                sample_rate = target_sample_rate

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Create AudioSignal object
            signal = AudioSignal(waveform, sample_rate=sample_rate)

            # Move signal to the same device as the model
            signal = signal.to(device)

            # Compress (encode) the audio signal
            with torch.no_grad():
                compressed = model.compress(signal)

            # Decompress (decode) back to audio signal
            reconstructed_signal = model.decompress(compressed)

            # Get the reconstructed waveform tensor
            reconstructed_waveform = reconstructed_signal.samples

            # Detach tensor and move to CPU for saving

            # Save reconstructed audio
            torchaudio.save(output_path, reconstructed_waveform.squeeze(0).cpu(), sample_rate)
            print(f"Saved reconstructed audio to {output_path}")

if __name__ == "__main__":
    # Example usage
    input_folder = "origin"  # Replace with your input audio folder path
    output_folder = "dac-output"  # Output folder path
    process_audio_folder(input_folder, output_folder)