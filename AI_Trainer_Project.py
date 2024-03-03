import cv2
import mediapipe
import time
import numpy as np
from Pose_Estimation_Module import PoseDetector

cap = cv2.VideoCapture("Hammer_Curl/Dumbbell_Hammer_Curl.mp4")
detector = PoseDetector()

p_time = 0
color = (255, 255, 255)

count_1, count_2 = -0.5, -0.5 # starts directly from 0.5, instead of 0, to start it from 0
direction_1, direction_2 = 0, 0

while True:
    success, img = cap.read()

    # img = cv2.imread("Hammer_Curl/hammer_curl.jpg")

    if not success:
        print("Video has ended.")
        break

    # cv2.imshow("Video", img)

    img = detector.find_pose(img, False)
    lm_list = detector.find_position(img, False)

    if len(lm_list) != 0:
        angle_1 = detector.find_angle(img, 12, 14, 16)
        angle_2 = detector.find_angle(img, 11, 13, 15)

        percentage_1 = np.interp(angle_1, (40, 175), (0, 100))
        percentage_2 = np.interp(angle_2, (60, 170), (0, 100))

        bar_1 = np.interp(angle_1, (40, 175), (650, 100))
        bar_2 = np.interp(angle_2, (60, 170), (650, 100))

        # print(angle_1, percentage_1, angle_2, percentage_2)

        if percentage_1 == 100:
            color = (200, 0, 200)
            if direction_1 == 0:
                count_1 += 0.5
                direction_1 = 1

        if percentage_1 == 0:
            color = (200, 200, 0)
            if direction_1 == 1:
                count_1 += 0.5
                direction_1 = 0

        if percentage_2 == 100:
            color = (200, 0, 200)
            if direction_2 == 0:
                count_2 += 0.5
                direction_2 = 1

        if percentage_2 == 0:
            color = (200, 200, 0)
            if direction_2 == 1:
                count_2 += 0.5
                direction_2 = 0

        # print(count_1, count_2)

        cv2.putText(img, f'Right: {str(int(count_1))}', (180, 100), cv2.FONT_HERSHEY_PLAIN,
                    5, (200, 150, 100), 5)

        cv2.putText(img, f'Left: {str(int(count_2))}', (700, 100), cv2.FONT_HERSHEY_PLAIN,
                    5, (100, 150, 200), 5)

        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar_1)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{str(int(percentage_1))}%', (1100, 60), cv2.FONT_HERSHEY_PLAIN,
                    4, color, 4)

        cv2.rectangle(img, (50, 100), (125, 650), color, 3)
        cv2.rectangle(img, (50, int(bar_1)), (125, 650), color, cv2.FILLED)
        cv2.putText(img, f'{str(int(percentage_1))}%', (30, 60), cv2.FONT_HERSHEY_PLAIN,
                    4, color, 4)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.imshow("Video", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
