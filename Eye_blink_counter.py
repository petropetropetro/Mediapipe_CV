import cv2
import mediapipe as mp
import time
import numpy as np
import math
import cvzone
from FaceMeshModule import FaceMesh
from cvzone.PlotModule import LivePlot



def calc_distance(point, point1):
    id, x, y = point
    id1, x1, y1 = point1
    return math.sqrt((x1-x)**2 + (y1-y)**2)


def main():

    TOTAL_BLINKS =0
    ratio_list:list = []
    cap = cv2.VideoCapture(0)
    plotY = LivePlot(640,360,[20,40], invert=True)
    detector = FaceMesh(max_num_of_face=1)
    counter = 0
    LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ] # point around one eye
    RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
    while True:

        success, frame = cap.read()
        
        img, faces = detector.detection(frame)

        if faces:
            face = faces[0]
            # print(face[1][159])
            # for id in id_list:
                # cv2.circle(img, face[id],5,(255,0,255))
            left_up = face[1][159]
            left_down = face[1][23]
            left_right = face[1][130]
            left_left = face[1][243]
            length = calc_distance(left_up, left_down)
            width = calc_distance(left_right, left_left)
            
            ratio = int((length/width) * 100)
            ratio_list.append(ratio)


            if len(ratio_list) > 3:
                ratio_list.pop(0)
            ratio_avg = sum(ratio_list)/len(ratio_list)
            if ratio_avg < 35 and counter == 0:
                TOTAL_BLINKS +=1
                counter = 1
            if counter != 0:
                counter += 1
                if counter > 10:
                    counter = 0

            img_plot = plotY.update(ratio_avg)

            # cv2.imshow("Image plot", img_plot)
            cv2.resize(img, (640, 360))
            img_stack = cvzone.stackImages([img, img_plot],1,1)
        else:
            cv2.resize(img, (640, 360))
            img_stack = cvzone.stackImages([img, img],1,1)
                

        cv2.putText(img_stack, str(int(TOTAL_BLINKS)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img_stack)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break



if __name__ == "__main__":
    main()