from pydub import AudioSegment
from pydub.generators import Sine
from pydub.effects import high_pass_filter, low_pass_filter
import numpy as np
import scipy.signal as signal

def apply_equalization(audio):
    # 100Hz 이하 하이패스 필터
    audio = high_pass_filter(audio, 100)
    
    # 300Hz 이하 영역 감소
    audio = audio.low_pass_filter(300)
    
    # 300~500Hz 영역 감소 (환경 울림/반사음)
    audio = audio.band_stop_filter(300, 500)
    
    # 1~4kHz 영역 부스트 (목소리의 존재감)
    audio = audio.apply_gain_stereo(1000, 4000, 5)
    
    # 4~6kHz 영역 부스트 (명료함)
    audio = audio.apply_gain_stereo(4000, 6000, 5)
    
    # 6~10kHz 영역 부스트 (치찰음)
    audio = audio.apply_gain_stereo(6000, 10000, 3)
    
    # 10~20kHz 로우패스 필터
    audio = low_pass_filter(audio, 10000)
    
    return audio

# 오디오 파일 로드
audio = AudioSegment.from_file("input.wav")

# 이퀄라이징 적용
equalized_audio = apply_equalization(audio)

# 수정된 오디오 파일 저장
equalized_audio.export("output.wav", format="wav")
