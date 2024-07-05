from pydub import AudioSegment
import noisereduce as nr
import numpy as np

# 오디오 파일 로드
audio = AudioSegment.from_file("voice_file/recording (1).wav")

# 배열로 변환
audio_array = np.array(audio.get_array_of_samples())

# 잡음 감소
reduced_noise = nr.reduce_noise(y=audio_array, sr=audio.frame_rate)

# 결과 저장
denoised_audio = AudioSegment(reduced_noise.tobytes(), frame_rate=audio.frame_rate, sample_width=audio.sample_width, channels=audio.channels)
denoised_audio.export("denoised_audio(1).wav", format="wav")
