import torchaudio
from torchaudio import functional as F
from df import enhance, init_df

def enhance_audio(input_path, output_path, low_center_freq, low_gain, mid_center_freq, mid_gain, high_center_freq, high_gain, Q=0.707):
    # DeepFilterNet 모델 초기화
    model, df_state, _ = init_df()

    # 입력 오디오 파일 로드
    waveform, sample_rate = torchaudio.load(input_path)

    # 소음 제거
    enhanced_waveform = enhance(model, df_state, waveform)

    # 저주파수 부스팅
    low_boosted_waveform = F.equalizer_biquad(enhanced_waveform, sample_rate, low_center_freq, low_gain, Q)

    # 중주파수 부스팅
    mid_boosted_waveform = F.equalizer_biquad(low_boosted_waveform, sample_rate, mid_center_freq, mid_gain, Q)

    # 고주파수 부스팅
    high_boosted_waveform = F.equalizer_biquad(mid_boosted_waveform, sample_rate, high_center_freq, high_gain, Q)

    # 결과 저장
    torchaudio.save(output_path, high_boosted_waveform, sample_rate)

# 사용 예시
if __name__ == "__main__":
    input_path = 'voice_file/recording_fixed (1).wav'
    output_path = 'voice_file/result/recording (1)_equalized.wav'
    low_center_freq = 200.0  # 저주파수 중심 주파수 (예: 200Hz)
    low_gain = 3.0  # 저주파수 부스팅 게인 (예: 3dB)
    mid_center_freq = 1000.0  # 중주파수 중심 주파수 (예: 1000Hz)
    mid_gain = 2.0  # 중주파수 부스팅 게인 (예: 2dB)
    high_center_freq = 5000.0  # 고주파수 중심 주파수 (예: 5000Hz)
    high_gain = 5.0  # 고주파수 부스팅 게인 (예: 5dB)
    Q = 0.707  # 품질 계수 (예: 0.707)
    enhance_audio(input_path, output_path, low_center_freq, low_gain, mid_center_freq, mid_gain, high_center_freq, high_gain, Q)
