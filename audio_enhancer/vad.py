import torch
import torchaudio
from IPython.display import Audio
import sys
import time

def vad(input_file, output_filename):
    start_time_vad = time.time()

    # Silero VAD 모델 다운로드
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    (get_speech_ttimestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    # 오디오 파일 로드
    wav = read_audio(input_file)

    # VAD를 사용하여 말하는 구간 찾기
    speech_timestamps = get_speech_ttimestamps(wav, model)

    # 말하는 구간만 추출
    speech_wav = collect_chunks(speech_timestamps, wav)

    # VAD 처리된 결과 저장
    save_audio(output_filename, speech_wav, sampling_rate=16000)

    duration_vad = time.time() - start_time_vad
    print(f"VAD processing time: {duration_vad} seconds")


    return duration_vad

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the input audio file as a command line argument.")
        sys.exit(1)

    input_file = sys.argv[1]
    output_filename = input_file.replace(".wav", "_vad.wav")

    vad(input_file, output_filename)