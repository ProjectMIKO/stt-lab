import sys
import time
import torch
import torchaudio
import torchaudio.sox_effects as sox_effects

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

# 컴프레서 효과 적용
def apply_compressor(waveform, sample_rate, threshold=-20, ratio=2.0, attack=50, release=200):
    effects = [
        ["compand", str(attack), str(threshold) + ":" + str(threshold), str(ratio), str(attack) + ":" + str(release)]
    ]
    waveform, _ = sox_effects.apply_effects_tensor(waveform, sample_rate, effects)
    return waveform

# 오디오 정규화 함수
def normalize_audio(file_path, target_dB=-20.0, threshold=-20, ratio=2.0, attack=50, release=200):
    waveform, sample_rate = load_audio(file_path)
    
    # DC 오프셋 제거
    waveform = correct_dc_offset(waveform)
    
    # 피크 정규화
    waveform = peak_normalize(waveform)
    
    # 소리 크기 조정
    waveform = adjust_volume(waveform, target_dB)
    
    # 컴프레서 효과 적용
    waveform = apply_compressor(waveform, sample_rate, threshold, ratio, attack, release)
    
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
