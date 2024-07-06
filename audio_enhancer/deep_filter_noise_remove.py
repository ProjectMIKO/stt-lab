import torchaudio
from df import enhance, init_df
import sys
import time

def remove_noise(input_path, output_path):
    start_time = time.time()
    
    # 입력 오디오 파일 로드
    waveform, sample_rate = torchaudio.load(input_path)
    
    # DeepFilterNet 모델 초기화
    model, df_state, _ = init_df()

    # 소음 제거
    denoised_waveform = enhance(model, df_state, waveform)

    # 결과 저장
    torchaudio.save(output_path, denoised_waveform, sample_rate)

    duration = time.time() - start_time
    print(f"Noise reduction processing time: {duration} seconds")

    return duration

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the input audio file as a command line argument.")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = input_file.replace(".wav", "_df.wav")

    # 노이즈 제거 수행
    duration = remove_noise(input_file, output_file)

    print(f"Total noise reduction processing time: {duration} seconds")
    print(f"Denoised file saved to: {output_file}")
