from pydub import AudioSegment
import noisereduce as nr
import numpy as np
import time
import sys


def remove_noise(input_file, output_filename):
  # 오디오 파일 로드
  audio = AudioSegment.from_file(input_file)
  
  start_time = time.time()
  # 배열로 변환
  audio_array = np.array(audio.get_array_of_samples())

  # 잡음 감소
  reduced_noise = nr.reduce_noise(y=audio_array, sr=audio.frame_rate)

  duration = time.time() - start_time

  # 결과 저장
  denoised_audio = AudioSegment(reduced_noise.tobytes(), frame_rate=audio.frame_rate, sample_width=audio.sample_width, channels=audio.channels)
  denoised_audio.export(output_filename, format="wav")
  
  return duration

# 오디오 파일 로드
if len(sys.argv) < 2:
  print("Please provide the input audio file as a command line argument.")
  sys.exit(1)

input_file = sys.argv[1]
output_filename = input_file.replace(".wav", "_noise_reduced.wav")

# torchaudio를 사용하여 오디오 처리
duration = remove_noise(input_file, output_filename)

print(f"noisereduce processing time: {duration} seconds")