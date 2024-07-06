import torchaudio
import torch
import sys
import time

def rms_normalize(waveform, target_dB=-20.0):
    rms = (waveform ** 2).mean().sqrt()
    scalar = 10 ** (target_dB / 20) / rms
    return waveform * scalar

def amplitude_to_db(waveform):
    return 20.0 * torch.log10(torch.clamp(waveform, min=1e-10))

def db_to_amplitude(db_waveform):
    return torch.pow(10.0, db_waveform / 20.0)

def dynamic_range_compression(waveform, threshold_dB=-35.0, ratio=2.0):
    waveform_db = amplitude_to_db(waveform)
    
    # Apply compression
    compressed_db = torch.where(
        waveform_db > threshold_dB,
        threshold_dB + (waveform_db - threshold_dB) / ratio,
        waveform_db
    )
    
    compressed_waveform = db_to_amplitude(compressed_db)
    return compressed_waveform

def dynamic_range_expansion(waveform, threshold_dB=-50.0, ratio=2.0):
    waveform_db = amplitude_to_db(waveform)
    
    # Apply expansion
    expanded_db = torch.where(
        waveform_db < threshold_dB,
        threshold_dB - (threshold_dB - waveform_db) * ratio,
        waveform_db
    )
    
    expanded_waveform = db_to_amplitude(expanded_db)
    return expanded_waveform

def soft_clipping(waveform, threshold=0.8):
    return torch.tanh(waveform / threshold) * threshold

def apply_compression(input_path, output_path, target_dB=-20.0, compression_threshold_dB=-35.0, compression_ratio=2.0, expansion_threshold_dB=-50.0, expansion_ratio=2.0, limit=0.8):
    start_time = time.time()
    
    # Load the audio file
    waveform, sample_rate = torchaudio.load(input_path)
    
    # Normalize the RMS
    waveform_normalized = rms_normalize(waveform, target_dB)
    
    # Apply dynamic range compression
    waveform_compressed = dynamic_range_compression(waveform_normalized, compression_threshold_dB, compression_ratio)
    
    # Apply dynamic range expansion
    waveform_expanded = dynamic_range_expansion(waveform_compressed, expansion_threshold_dB, expansion_ratio)
    
    # Apply soft clipping to avoid excessive peaks
    waveform_clipped = soft_clipping(waveform_expanded, limit)
    
    # Save the result
    torchaudio.save(output_path, waveform_clipped, sample_rate)

    duration = time.time() - start_time
    print(f"Normalized processing time: {duration} seconds")

    return duration

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the input audio file as a command line argument.")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = input_file.replace(".wav", "_normalized.wav")

    # Adjust these parameters as needed
    target_dB = -10.0
    compression_threshold_dB = -25.0
    compression_ratio = 2.0
    expansion_threshold_dB = -40.0
    expansion_ratio = 2.0
    limit = 0.5

    # Apply compression and expansion to equalize volume
    duration = apply_compression(input_file, output_file, target_dB, compression_threshold_dB, compression_ratio, expansion_threshold_dB, expansion_ratio, limit)

    print(f"Total normalized processing time: {duration} seconds")
    print(f"Processed file saved to: {output_file}")
