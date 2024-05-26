import cv2
import numpy as np

# 물체 추적 함수 정의
def start_tracking():
    # 비디오 캡쳐 오브젝트 초기화
    cap = cv2.VideoCapture(0)

    # 프레임에 대한 크기 조정 인자 정의
    scaling_factor = 0.5

    # 추적할 프레임 수
    num_frames_to_track = 5

    # 건너뛸 프레임 수
    num_frames_jump = 2

    # 변수 초기화
    tracking_paths = []
    frame_index = 0

    # 추적 매개변수(윈도우 크기, 최대 수준, 종료 조건) 정의
    tracking_params = dict(winSize  = (11, 11), maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                10, 0.03))

    # 사용자가 ESC 키 누를 때까지 반복
    while True:
        # 카메라에서 현재 프레임 캡쳐
        _, frame = cap.read()

        # 프레임 크기 조정
        frame = cv2.resize(frame, None, fx=scaling_factor, 
                fy=scaling_factor, interpolation=cv2.INTER_AREA)

        # RGB 프레임을 흑백으로 변환
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 복사본 생성
        output_img = frame.copy()

        # 추적 경로의 길이가 0보다 큰지 확인
        if len(tracking_paths) > 0:
            # 이미지 가져오기
            prev_img, current_img = prev_gray, frame_gray

            # 특징점 구성
            feature_points_0 = np.float32([tp[-1] for tp in \
                    tracking_paths]).reshape(-1, 1, 2)

            # 광학 흐름 계산 (이전 프레임과 현재 프레임 사이의)
            feature_points_1, _, _ = cv2.calcOpticalFlowPyrLK(
                    prev_img, current_img, feature_points_0, 
                    None, **tracking_params)

            # 역(reverse) 광학 흐름 계산
            feature_points_0_rev, _, _ = cv2.calcOpticalFlowPyrLK(
                    current_img, prev_img, feature_points_1, 
                    None, **tracking_params)

            # 순 광학 흐름과 역 광학 흐름 사이의 차이 계산
            diff_feature_points = abs(feature_points_0 - \
                    feature_points_0_rev).reshape(-1, 2).max(-1)

            # 대표 특징점 추출
            good_points = diff_feature_points < 1

            # 새로운 추적 경로에 대한 변수 초기화
            new_tracking_paths = []

            # 추출한 대표 특징점에 대해 루프를 돌며 그 주위에 원을 그림
            for tp, (x, y), good_points_flag in zip(tracking_paths, 
                        feature_points_1.reshape(-1, 2), good_points):
                # 플래그가 차밍 아니면 건너뜀
                if not good_points_flag:
                    continue

                # x, y 좌표를 추가하고 추적할 프레임의 수 초과하지 않는지 확인
                # (x, y 지점까지의 거리가 임계점을 넘지 않는지 확인)
                # 넘으면 삭제
                tp.append((x, y))
                if len(tp) > num_frames_to_track:
                    del tp[0]

                new_tracking_paths.append(tp)

                # 특징점 주위에 원 그리기
                cv2.circle(output_img, (x, y), 3, (0, 255, 0), -1)

            # 추적 경로 업데이트
            tracking_paths = new_tracking_paths

            # 선 그리기
            cv2.polylines(output_img, [np.int32(tp) for tp in \
                    tracking_paths], False, (0, 150, 0))

        # 지정된 프레임 수만큼 건너뛰고 나서 if 조건문으로 들어감
        if not frame_index % num_frames_jump:
            # 마스크 생성한 후 원 그림
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            for x, y in [np.int32(tp[-1]) for tp in tracking_paths]:
                cv2.circle(mask, (x, y), 6, 0, -1)

            # 추적할 대표 특징점 계산
            # 함수에 여러 가지 매개변수(마스크, 최대 모서리, 품질 수준, 최소 거리, 블록 크기)
            # 를 지정해서 호출하는 방식으로 처리
            feature_points = cv2.goodFeaturesToTrack(frame_gray, 
                    mask = mask, maxCorners = 500, qualityLevel = 0.3, 
                    minDistance = 7, blockSize = 7) 

            # 특징점이 있다면 추적 경로에 추가
            if feature_points is not None:
                for x, y in np.float32(feature_points).reshape(-1, 2):
                    tracking_paths.append([(x, y)])

        # 변수 업데이트
        frame_index += 1
        prev_gray = frame_gray

        # 결과 출력
        cv2.imshow('Optical Flow', output_img)

        # 사용자가 ESC 키 눌렀으면 빠져나감
        c = cv2.waitKey(1)
        if c == 27:
            break

if __name__ == '__main__':
	# 추적기 시작
    start_tracking()

    # 모든 창(window) 닫기
    cv2.destroyAllWindows()

