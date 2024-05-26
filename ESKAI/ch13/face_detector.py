import cv2
import numpy as np

# Haar cascade 파일 불러옴
face_cascade = cv2.CascadeClassifier(
        'haar_cascade_files/haarcascade_frontalface_default.xml')

# 파일 체크
if face_cascade.empty():
	raise IOError('Unable to load the face cascade classifier xml file')

# 비디오 캡처 오브젝트 초기화
cap = cv2.VideoCapture(0)

# 크기 조정 인자 정의
scaling_factor = 0.5

# 사용자가 ESC 키 누를 때까지 반복
while True:
    # 카메라에서 현재 프레임 캡쳐
    _, frame = cap.read()

    # 프레임 크기 조정
    frame = cv2.resize(frame, None, 
            fx=scaling_factor, fy=scaling_factor, 
            interpolation=cv2.INTER_AREA)

    # grayscale로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 흑백 이미지에 대해 얼굴 검출기 실행
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 얼굴 주위에 직사각형 그리기
    for (x,y,w,h) in face_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

    # 결과 화면 출력
    cv2.imshow('Face Detector', frame)

    # ESC 눌렸는지 확인하고 빠져나옴
    c = cv2.waitKey(1)
    if c == 27:
        break

# 비디오 캡처 오브젝트 해제
cap.release()

# 창 모두 닫기
cv2.destroyAllWindows()
