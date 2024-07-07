import sys
import time
import torch
import torchaudio
import scipy.signal as signal
from df import enhance, init_df

# Normalize
def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

def correct_dc_offset(waveform):
    return waveform - waveform.mean(dim=-1, keepdim=True)

def peak_normalize(waveform, peak_value=0.95):
    return waveform / torch.max(torch.abs(waveform)) * peak_value

def adjust_volume(waveform, target_dB=-20.0):
    rms = waveform.pow(2).mean().sqrt()
    scalar = (10 ** (target_dB / 20)) / (rms + 1e-10)
    return waveform * scalar

def limit_peak(waveform, peak_limit=0.9):
    waveform = torch.clamp(waveform, -peak_limit, peak_limit)
    return waveform

def smooth_amplify(waveform, sample_rate, threshold, gain, sustain_time, fade_length):
    threshold_linear = 10 ** (threshold / 20)
    gain_factor = 10 ** (gain / 20)
    
    frame_size = int(sample_rate * 0.02)  # 20ms 프레임
    sustain_frames = int(sustain_time * sample_rate / frame_size)  # 증폭 유지 시간
    
    num_frames = waveform.size(1) // frame_size
    smoothed_waveform = waveform.clone()
    
    sustain_counter = 0
    
    for i in range(num_frames):
        frame_start = i * frame_size
        frame_end = frame_start + frame_size
        frame = waveform[:, frame_start:frame_end]
        rms = frame.pow(2).mean().sqrt()
        
        if rms < threshold_linear or sustain_counter > 0:
            frame_gain = gain_factor * (threshold_linear / (rms + 1e-10))
            frame_gain = min(frame_gain, gain_factor)  # 최대 gain_factor 이상 증폭하지 않도록 제한
            
            if sustain_counter == 0:
                # 페이드인 적용
                fade_in_len = min(fade_length, frame_size)
                fade_in = torch.linspace(0, 1, fade_in_len)
                frame[:, :fade_in_len] *= fade_in.unsqueeze(0)
                
            # 증폭 적용
            frame *= frame_gain
            
            if sustain_counter == sustain_frames:
                # 지수 감쇠 적용
                fade_out_len = min(fade_length, frame_size)
                fade_out = torch.exp(-torch.linspace(0, 5, fade_out_len))
                frame[:, -fade_out_len:] *= fade_out.unsqueeze(0)
                sustain_counter = 0  # 증폭 유지 시간 초기화
            else:
                sustain_counter += 1
            
            smoothed_waveform[:, frame_start:frame_end] = frame
        else:
            sustain_counter = 0  # 증폭 유지 시간 초기화
    
    # 최대 피크 값 제한
    smoothed_waveform = limit_peak(smoothed_waveform)
    
    return smoothed_waveform

def normalize_audio(file_path, target_dB=-20.0, threshold=-30.0, gain=5.0, sustain_time=0.03, fade_length=15, peak_limit=0.9):
    waveform, sample_rate = load_audio(file_path)
    
    # DC 오프셋 제거
    waveform = correct_dc_offset(waveform)
    
    # 피크 정규화
    waveform = peak_normalize(waveform)
    
    # 소리 크기 조정
    waveform = adjust_volume(waveform, target_dB)
    
    # 부드러운 증폭 적용
    waveform = smooth_amplify(waveform, sample_rate, threshold, gain, sustain_time, fade_length)
    
    # 최대 피크 값 제한
    waveform = limit_peak(waveform, peak_limit)
    
    return waveform, sample_rate

# Noise reduction
def remove_noise(input_waveform, sample_rate):
    # DeepFilterNet 모델 초기화
    model, df_state, _ = init_df()

    # 소음 제거
    denoised_waveform = enhance(model, df_state, input_waveform)

    return denoised_waveform, sample_rate

# Equalizer
def high_pass_filter(waveform, sample_rate, cutoff=100):
    nyquist = 0.5 * sample_rate
    norm_cutoff = cutoff / nyquist
    b, a = signal.butter(1, norm_cutoff, btype='high', analog=False)
    filtered_waveform = signal.lfilter(b, a, waveform, axis=-1)
    return filtered_waveform

def low_pass_filter(waveform, sample_rate, cutoff=10000):
    nyquist = 0.5 * sample_rate
    norm_cutoff = cutoff / nyquist
    b, a = signal.butter(1, norm_cutoff, btype='low', analog=False)
    filtered_waveform = signal.lfilter(b, a, waveform, axis=-1)
    return filtered_waveform

def band_stop_filter(waveform, sample_rate, low_cutoff, high_cutoff):
    nyquist = 0.5 * sample_rate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = signal.butter(1, [low, high], btype='bandstop', analog=False)
    filtered_waveform = signal.lfilter(b, a, waveform, axis=-1)
    return filtered_waveform

def band_pass_filter(waveform, sample_rate, low_cutoff, high_cutoff, gain=1.0):
    nyquist = 0.5 * sample_rate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = signal.butter(1, [low, high], btype='band', analog=False)
    filtered_waveform = signal.lfilter(b, a, waveform, axis=-1)
    filtered_waveform *= gain
    return filtered_waveform

def volume_up_peak(waveform, peak_value=0.8):
    peak = torch.max(torch.abs(waveform))
    scalar = peak_value / peak
    normalized_waveform = waveform * scalar
    return normalized_waveform

def equalize_audio(input_waveform, sample_rate):
    waveform_filtered = input_waveform.numpy()
    
    # 100Hz 이하 하이패스 필터
    waveform_filtered = high_pass_filter(waveform_filtered, sample_rate, 100)
    
    # 100~300Hz 영역 감소 (저역 필터)
    waveform_filtered += band_pass_filter(waveform_filtered, sample_rate, 100, 300, gain=1.0)
    
    # 300~500Hz 영역 감소 (울림/반사음 필터)
    waveform_filtered = band_stop_filter(waveform_filtered, sample_rate, 300, 500)

    # 4~6kHz 영역 부스트 (명료함 부스트)
    boosted_4_6kHz = band_pass_filter(waveform_filtered, sample_rate, 4000, 6000, gain=2.0)
    waveform_filtered += boosted_4_6kHz
   
    # 6~10kHz 영역 부스트 (치찰음 부스트)
    boosted_6_8kHz = band_pass_filter(waveform_filtered, sample_rate, 6000, 8000, gain=1.0)
    waveform_filtered += boosted_6_8kHz
    
    # 10~20kHz 로우패스 필터
    waveform_filtered = low_pass_filter(waveform_filtered, sample_rate, 10000)
    
    # 음량 조정
    waveform_filtered = volume_up_peak(torch.tensor(waveform_filtered, dtype=torch.float32))
    
    # tensor로 다시 변환 (float32 형식 유지)
    filtered_waveform = torch.tensor(waveform_filtered, dtype=torch.float32)
    
    return filtered_waveform, sample_rate

# 전체 프로세스
def process_audio(input_path, output_path):
    start_time = time.time()
    
    # Step 1: Normalize
    normalized_waveform, sample_rate = normalize_audio(input_path)
    
    # Step 2: Noise Reduction
    denoised_waveform, sample_rate = remove_noise(normalized_waveform, sample_rate)
    
    # Step 3: Equalize
    equalized_waveform, sample_rate = equalize_audio(denoised_waveform, sample_rate)
    
    duration = time.time() - start_time
    print(duration)
    
    # Save the final output
    torchaudio.save(output_path, equalized_waveform, sample_rate)
    print(f"Processed file saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the input audio file as a command line argument.")
        sys.exit(1)

    input_file = sys.argv[1]
    output_filename = input_file.replace(".wav", "_processed.wav")

    # 전체 오디오 처리
    process_audio(input_file, output_filename)
