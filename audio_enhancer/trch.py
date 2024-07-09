import time
import torchaudio
import scipy.signal as signal
import numpy as np
import torch

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

def process_with_torchaudio(input_path, output_path):
    start_time = time.time()
    
    # 오디오 파일 로드
    waveform, sample_rate = torchaudio.load(input_path)
    
    # numpy 배열로 변환
    waveform_np = waveform.numpy()
    
    # 100Hz 이하 하이패스 필터
    waveform_hp = high_pass_filter(waveform_np, sample_rate, 100)
    
    # 10000Hz 이상 로우패스 필터
    waveform_lp = low_pass_filter(waveform_hp, sample_rate, 10000)
    
    # tensor로 다시 변환 (float32 형식 유지)
    filtered_waveform = torch.tensor(waveform_lp, dtype=torch.float32)
    
    # 필터 적용된 오디오 파일 저장
    torchaudio.save(output_path, filtered_waveform, sample_rate)
    
    duration = time.time() - start_time
    print(f"torchaudio processing time: {duration} seconds")
    
    return duration

# 사용 예시
input_path = r"voice_file\음성테스트_맥북(노래).wav"
output_path_torchaudio = r"voice_file\output_torchaudio.wav"

torchaudio_duration = process_with_torchaudio(input_path, output_path_torchaudio)
