import cv2
import mediapipe as mp
from FaceDetectionModule import FaceDetection

import time
max_time = 3600 # 1 hour in seconds

def main():

    cap = cv2.VideoCapture(0)
    detector = FaceDetection()
    pTime = 0
    cTime = 0
    start_time = time.time()
    elapsed_time = 0
    while True:

        success, img = cap.read()
        list_bbox = []
        img, list_bbox = detector.detection(img)
        if list_bbox:
            elapsed_time = int(time.time()-start_time)
        else:
            start_time = time.time()
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        if elapsed_time > max_time:
            cv2.setWindowProperty("Image", cv2.WND_PROP_TOPMOST,1)
            cv2.putText(img, "Out of work hours", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)

        cv2.putText(img, str(int(fps)), (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)
        cv2.putText(img, f"Time in seconds: {elapsed_time}", (10,40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break



if __name__ == "__main__":
    main()