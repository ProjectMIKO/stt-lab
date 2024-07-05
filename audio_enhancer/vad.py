import torch
import torchaudio
from IPython.display import Audio

# Silero VAD 모델 다운로드
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
(get_speech_ttimestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# 오디오 파일 로드
wav = read_audio('voice_file/recording_fixed (1).wav')

# VAD를 사용하여 말하는 구간 찾기
speech_timestamps = get_speech_ttimestamps(wav, model)

# 말하는 구간만 추출
speech_wav = collect_chunks(speech_timestamps, wav)

# 결과 저장
save_audio('voice_file/speech_only (1).wav', speech_wav, sampling_rate=16000)
