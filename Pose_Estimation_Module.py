import math

import cv2
import mediapipe as mp
import time


class PoseDetector:
    def __init__(self, model=False, m_complexity=1, smooth=True, enable_seg=False, smooth_seg=True, det_con=0.5, track_con=0.5):
        self.model = model
        self.m_complexity = m_complexity
        self.smooth = smooth
        self.enable_seg = enable_seg
        self.smooth_seg = smooth_seg
        self.det_con = det_con
        self.track_con = track_con

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.model, self.m_complexity, self.smooth, self.enable_seg, self.smooth_seg, self.det_con, self.track_con)

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)

        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return img

    def find_position(self, img, draw=True):
        self.lm_list = []

        if self.results.pose_landmarks:
            for l_id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([l_id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (100, 100, 200), cv2.FILLED)

        return self.lm_list

    def find_angle(self, img, p1, p2, p3, draw=True):

        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        x3, y3 = self.lm_list[p3][1:]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

        if angle < 0:
            angle += 360

        # print(angle)

        mx, my = int((x1 + x3) / 2), int((y1 + y3) / 2)

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 0, 0), 3)
            cv2.circle(img, (x1, y1), 10, (150, 255, 100), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (150, 255, 100), 2)
            cv2.circle(img, (x2, y2), 10, (150, 255, 100), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (150, 255, 100), 2)
            cv2.circle(img, (x3, y3), 10, (150, 255, 100), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (150, 255, 100), 2)
            cv2.putText(img, str(int(angle)), (mx, my), cv2.FONT_HERSHEY_PLAIN,
                        2, (200, 50, 100), 3)

        return angle


def main():
    cap = cv2.VideoCapture('Videos/1.mp4')
    detector = PoseDetector()
    p_time = 0

    while True:
        success, img = cap.read()

        # Check if the video has finished if not you'll get an error
        if not success:
            print("Video has ended.")
            break

        img = detector.find_pose(img)
        lm_list = detector.find_position(img, draw=False)

        if len(lm_list) != 0:
            print(lm_list[14])
            cv2.circle(img, (lm_list[14][1], lm_list[14][2]), 5, (200, 100, 100), cv2.FILLED)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, f"FPS: {str(int(fps))}", (70, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3,
                    (200, 100, 150), 3)

        cv2.imshow("Video", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# https://www.youtube.com/watch?v=01sAkU_NvOY
