import torchaudio
from torchaudio import functional as F
from df import enhance, init_df


def enhance_audio(input_path, output_path, Q=1.0):
    
    # 입력 오디오 파일 로드
    waveform, sample_rate = torchaudio.load(input_path)

    # 저주파수 부스팅
    low_center_freq = 150.0
    low_gain = 3.0
    enhanced_waveform = F.equalizer_biquad(waveform, sample_rate, low_center_freq, low_gain, Q)

    # 중주파수 부스팅
    mid_center_freq = 1000.0
    mid_gain = 3.0
    enhanced_waveform = F.equalizer_biquad(enhanced_waveform, sample_rate, mid_center_freq, mid_gain, Q)

    # 고주파수 부스팅
    high_center_freq = 4000.0
    high_gain = 4.0
    enhanced_waveform = F.equalizer_biquad(enhanced_waveform, sample_rate, high_center_freq, high_gain, Q)

    # 추가적인 중고주파수 부스팅 (특히 "주" 음절의 명료도 향상)
    additional_center_freq = 2500.0
    additional_gain = 3.0
    enhanced_waveform = F.equalizer_biquad(enhanced_waveform, sample_rate, additional_center_freq, additional_gain, Q)

    # 볼륨 부스팅
    boosted_waveform = enhanced_waveform * 1.2
    
    # DeepFilterNet 모델 초기화
    model, df_state, _ = init_df()

    # 소음 제거
    denoised_waveform = enhance(model, df_state, boosted_waveform)

    # 결과 저장
    torchaudio.save(output_path, denoised_waveform, sample_rate)


# 사용 예시
if __name__ == "__main__":
    input_path = 'voice_file/recording_fixed (1).wav'
    output_path = 'voice_file/result/recording 1111_equalized.wav'
    
    Q = 1.0  # 품질 계수 
    
    enhance_audio(input_path, output_path, Q)
