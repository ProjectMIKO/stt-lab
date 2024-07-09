import sys
import time
import torchaudio
import scipy.signal as signal
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

def process_with_torchaudio(input_path, output_path):
    start_time = time.time()
    
    # 오디오 파일 로드
    waveform, sample_rate = torchaudio.load(input_path)
    
    # numpy 배열로 변환
    waveform_filtered = waveform.numpy()
    
    # 100Hz 이하 하이패스 필터
    waveform_filtered = high_pass_filter(waveform_filtered, sample_rate, 100)
    
    # 100~300Hz 영역 
    waveform_filtered += band_pass_filter(waveform_filtered, sample_rate, 100, 300, gain=1.0)
    
    # 300~500Hz 영역 감소 (울림/반사음 필터)
    waveform_filtered = band_stop_filter(waveform_filtered, sample_rate, 300, 500)
    
    # 500~1kHz 영역 부스팅 (목소리의 풍부함 부스트)
    # waveform_filtered += band_pass_filter(waveform_filtered, sample_rate, 500, 1000, gain=0.5)
    
    # 1~4kHz 영역 부스트 (목소리의 존재감 부스트)
    # waveform_filtered -= band_pass_filter(waveform_filtered, sample_rate, 1000, 4000, gain=0.5)
    
    # 4~6kHz 영역 부스트 (명료함 부스트)
    waveform_filtered += band_pass_filter(waveform_filtered, sample_rate, 4000, 6000, gain=2)
    
    # 6~10kHz 영역 부스트 (치찰음)
    waveform_filtered += band_pass_filter(waveform_filtered, sample_rate, 6000, 8000, gain=1)
    
    # 10~20kHz 로우패스 필터
    waveform_filtered = low_pass_filter(waveform_filtered, sample_rate, 10000)
    
    # 음량 조정
    waveform_filtered = volume_up_peak(torch.tensor(waveform_filtered, dtype=torch.float32))
    
    # tensor로 다시 변환 (float32 형식 유지)
    filtered_waveform = torch.tensor(waveform_filtered, dtype=torch.float32)
    
    # 필터 적용된 오디오 파일 저장
    torchaudio.save(output_path, filtered_waveform, sample_rate)
    
    duration = time.time() - start_time
    print(f"torchaudio processing time: {duration} seconds")
    
    return duration
  
  
# 오디오 파일 로드
if len(sys.argv) < 2:
  print("Please provide the input audio file as a command line argument.")
  sys.exit(1)

input_file = sys.argv[1]
output_filename = input_file.replace(".wav", "_eq.wav")

# torchaudio를 사용하여 오디오 처리
duration = process_with_torchaudio(input_file, output_filename)

print(f"torchaudio processing time: {duration} seconds")
