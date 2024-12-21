import librosa
import numpy as np
import matplotlib.pyplot as plt

# 1. 오디오 파일 로드
audio_path = r"C:\Users\jmys1\Desktop\훈련데이터\호흡주기_wav\7일 1_1(3초).wav"
y, sr = librosa.load(audio_path, sr=None)

# 40초까지만 자르기
max_duration_sec = 40  # 최대 40초
y = y[:int(max_duration_sec * sr)]  # 40초에 해당하는 샘플만 남기기
print(f"Data truncated to {max_duration_sec} seconds.")

# 노이즈 필터링
y_filtered = librosa.effects.preemphasis(y)
print("Noise remove success")

# 2. 데이터 길이 확인
audio_duration = len(y) / sr
print(f"Audio Duration: {audio_duration:.2f} seconds")

# 3. MFCC 계산
mfcc = librosa.feature.mfcc(y=y_filtered, sr=sr, n_mfcc=13)  # 13차원 MFCC 계산
mfcc_energy = mfcc[0]  # 첫 번째 MFCC 계수를 사용
print("MFCC Energy:", mfcc_energy)

# 4. 중앙값을 임계값으로 설정
#threshold = np.median(mfcc_energy)-50
threshold = np.median(mfcc_energy)
print("Threshold (Median):", threshold)

# 5. breath_segments 정의
breath_segments = mfcc_energy > threshold  # 날숨(True), 들숨(False)

# 6. 전환점 탐지
change_points = np.where(np.diff(breath_segments.astype(int)) != 0)[0]

# 빈 배열 처리
if len(change_points) == 0:
    print("No breath change points detected.")
    change_points = np.array([0])  # 기본값 설정

# 7. 최소 간격 적용 (1.0초)
min_interval = int(sr * 1.0 / 512)  # 최소 간격을 프레임 단위로 변환
filtered_change_points = [change_points[0]]

for cp in change_points[1:]:
    if cp - filtered_change_points[-1] > min_interval:
        filtered_change_points.append(cp)

change_points = np.array(filtered_change_points)

# 8. MFCC 시각화
frames = np.arange(len(mfcc_energy))  # 프레임 생성
time_axis = librosa.frames_to_time(frames, sr=sr, hop_length=512)

plt.figure(figsize=(10, 4))
plt.plot(time_axis, mfcc_energy, label="MFCC (Energy-like Feature)", color="blue")
plt.axhline(y=threshold, color="red", linestyle="--", label="Threshold")
plt.vlines(change_points * 512 / sr, ymin=np.min(mfcc_energy), ymax=np.max(mfcc_energy), colors='green', linestyle="--", label="Breath Change")
plt.title("MFCC Energy and Breath Cycle Changes")
plt.xlabel("Time (s)")
plt.ylabel("MFCC (1st Coefficient)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("mfcc_breath_cycle_changes4-2.png")  # 플롯을 이미지 파일로 저장
plt.close()  # 플롯 창 닫기
print("Plot saved successfully.")

# 9. 호흡 주기 계산
if len(change_points) >= 2:
    breath_durations = np.diff(change_points)  # 구간 길이 계산
    average_cycle = np.mean(breath_durations) / sr * 512  # 초 단위 변환
    print(f"평균 호흡 주기: {average_cycle:.2f} 초")
else:
    print("호흡 주기를 계산할 수 없습니다.")