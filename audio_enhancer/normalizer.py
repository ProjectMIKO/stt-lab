import sys
import time
import torch
import torchaudio

# 오디오 파일 로드
def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

# DC 오프셋 제거
def correct_dc_offset(waveform):
    return waveform - waveform.mean()

# 피크 정규화
def peak_normalize(waveform):
    return waveform / torch.max(torch.abs(waveform))

# 소리 크기 조정
def adjust_volume(waveform, target_dB=-20.0):
    rms = waveform.pow(2).mean().sqrt()
    scalar = (10 ** (target_dB / 20)) / (rms + 1e-10)
    return waveform * scalar

# 부드러운 증폭 적용
def smooth_amplify(waveform, sample_rate, threshold=-40.0, gain=10.0, sustain_time=0.2, fade_length=1000):
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
                # 페이드아웃 적용
                fade_out_len = min(fade_length, frame_size)
                fade_out = torch.linspace(1, 0, fade_out_len)
                frame[:, -fade_out_len:] *= fade_out.unsqueeze(0)
                sustain_counter = 0  # 증폭 유지 시간 초기화
            else:
                sustain_counter += 1
            
            smoothed_waveform[:, frame_start:frame_end] = frame
        else:
            sustain_counter = 0  # 증폭 유지 시간 초기화
    
    return smoothed_waveform

# 오디오 정규화 함수
def normalize_audio(file_path, target_dB=-20.0, threshold=-25.0, gain=30.0, sustain_time=0.5, fade_length=500):
    waveform, sample_rate = load_audio(file_path)
    
    # DC 오프셋 제거
    waveform = correct_dc_offset(waveform)
    
    # 피크 정규화
    waveform = peak_normalize(waveform)
    
    # 소리 크기 조정
    waveform = adjust_volume(waveform, target_dB)
    
    # 부드러운 증폭 적용
    waveform = smooth_amplify(waveform, sample_rate, threshold, gain, sustain_time, fade_length)
    
    return waveform, sample_rate

# 오디오 파일 처리 및 시간 측정
def process_with_torchaudio(input_file, output_file):
    start_time = time.time()
    
    normalized_waveform, sample_rate = normalize_audio(input_file)
    
    # 정규화된 오디오 저장
    torchaudio.save(output_file, normalized_waveform, sample_rate)
    
    duration = time.time() - start_time
    return duration

# 커맨드 라인 인자 처리
if len(sys.argv) < 2:
    print("Please provide the input audio file as a command line argument.")
    sys.exit(1)

input_file = sys.argv[1]
output_filename = input_file.replace(".wav", "_nr.wav")

# torchaudio를 사용하여 오디오 처리
duration = process_with_torchaudio(input_file, output_filename)

print(f"torchaudio processing time: {duration} seconds")
