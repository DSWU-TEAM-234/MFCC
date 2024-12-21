import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import re  # 정규표현식 사용

# 오디오 파일 경로
folder_path = r"C:\Users\jmys1\Desktop\규나_호흡데이터2\wav"

# 결과 저장
accuracy_results = []

# 폴더의 모든 .wav 파일 순회
for file_name in os.listdir(folder_path):
    if file_name.endswith(".wav"):  # .wav 파일만 선택
        audio_path = os.path.join(folder_path, file_name)

        # 파일 이름에서 수작업 기준값 추출
        match = re.search(r"\(([\d.]+)\)", file_name)  # () 안 숫자 추출
        if not match:
            print(f"기준값을 찾을 수 없습니다: {file_name}")
            continue
        manual_cycle = float(match.group(1))  # 기준값

        # 오디오 파일 로드
        y, sr = librosa.load(audio_path, sr=None)

        # 40초까지만 자르기
        max_duration_sec = 40  # 최대 40초
        y = y[:int(max_duration_sec * sr)]  # 40초에 해당하는 샘플만 남기기

        # 노이즈 필터링
        y_filtered = librosa.effects.preemphasis(y)

        # MFCC 계산
        mfcc = librosa.feature.mfcc(y=y_filtered, sr=sr, n_mfcc=13)  # 13차원 MFCC 계산
        mfcc_energy = mfcc[0]  # 첫 번째 MFCC 계수를 사용

        # 중앙값을 임계값으로 설정
        threshold = np.median(mfcc_energy)

        # breath_segments 정의
        breath_segments = mfcc_energy > threshold  # 날숨(True), 들숨(False)

        # 전환점 탐지
        change_points = np.where(np.diff(breath_segments.astype(int)) != 0)[0]

        # 빈 배열 처리
        if len(change_points) == 0:
            print(f"호흡 주기를 계산할 수 없습니다: {file_name}")
            continue

        # 최소 간격 적용 (1.0초)
        min_interval = int(sr * 1.0 / 512)  # 최소 간격을 프레임 단위로 변환
        filtered_change_points = [change_points[0]]

        for cp in change_points[1:]:
            if cp - filtered_change_points[-1] > min_interval:
                filtered_change_points.append(cp)

        change_points = np.array(filtered_change_points)

        # 호흡 주기 계산
        if len(change_points) >= 2:
            breath_durations = np.diff(change_points)  # 구간 길이 계산
            average_cycle = np.mean(breath_durations) / sr * 512  # 초 단위 변환
            calculated_cycle = average_cycle * 2  # 전체 호흡 주기 (들숨 + 날숨)

            # 정확도 계산
            accuracy = 100 * (1 - abs(calculated_cycle - manual_cycle) / manual_cycle)
            accuracy_results.append((file_name, manual_cycle, calculated_cycle, accuracy))

            print(f"파일: {file_name}")
            print(f"수작업 기준값: {manual_cycle} 초, 알고리즘 계산값: {calculated_cycle:.2f} 초")
            print(f"정확도: {accuracy:.2f}%\n")

        else:
            print(f"호흡 주기를 계산할 수 없습니다: {file_name}")

# 전체 평균 정확도 계산
if accuracy_results:
    mean_accuracy = np.mean([result[3] for result in accuracy_results])
    print(f"전체 평균 정확도: {mean_accuracy:.2f}%")

    # 정확도 결과 출력
    for result in accuracy_results:
        print(f"파일: {result[0]}, 기준값: {result[1]}, 계산값: {result[2]:.2f}, 정확도: {result[3]:.2f}%")
else:
    print("정확도를 계산할 수 없습니다.")
