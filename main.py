import sys
import cv2
import os
from sys import platform
import csv
import time
from control.time_thread import myThread
from recognition.face_model import Facemodel
from recognition.hand_gesture import HandModel
from recognition.raise_recognition import ArmModel
import numpy as np
import threading
# Import Openpose (Windows/Ubuntu/OSX)

dir_path = os.path.dirname(os.path.realpath(__file__))
# Windows Import
if platform == "win32":
    # Change these variables to point to the correct folder (Release/x64 etc.)
    sys.path.append(dir_path + '/openpose');
    os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/openpose;' + dir_path + '/3rdparty;'
    import pyopenpose as op

# Flags
#parser = argparse.ArgumentParser()
# parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
#args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "./models/"
params["net_resolution"] = "80x80"
params["hand"] = True
#params["face"] = True
#params["hand_detector"] = 2
#params["body_disable"] = True
#params["disable_blending"] = True
"""
# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1]) - 1:
        next_item = args[1][i + 1]
    else:
        next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-', '')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-', '')
        if key not in params: params[key] = next_item
"""
# Construct it from system arguments
# Fixed the handRectangles to only 1 person and 1 big rectangle, don't have to keep changing rectangle

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

class Openpose():
    def __init__(self):
        super().__init__()
        self.gesture_model = HandModel()
        self.face_model = Facemodel()
        self.arm_recognition = ArmModel()
        self.head_points = [0, 15, 16, 17, 18]
        self.result = {0: 'ok', 1: 'one'}
        self.name = {0: 'WONG Yuk Kit', 6: 'Unknown'}
        self.hand_state = {0: 'Up', 1: 'Down'}

    def gesture_recognition(self, frame, keypoints, person_message):
        hands = keypoints[1]
        hand_gesture = "unknown"
        if hands[0].size == 1:
            return frame, keypoints, person_message
        else:
            person_num = hands[0].shape[0]
            for i in range(person_num):
                for hand in hands:
                    rect = self.gesture_model.hand_bbox(hand[i])
                    gesture,probability = self.gesture_model(keypoints,i)
                    if rect:
                        x, y, w, h = rect
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255))
                        cv2.putText(frame, self.result[gesture], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
                person_message = person_message.tolist()
                person_message[i].append(gesture)
                person_message = np.array(person_message)
        return frame, keypoints, person_message


    def update_frame(self, frame):
        original_frame = frame
        start_time = time.time()
        frame, keypoints = self.get_skeleton(frame)
        if keypoints[0].size == 1:
            return frame, keypoints, None
        frame, keypoints, person_message = self.face_recognition(keypoints, original_frame, frame)
        #frame = self.gesture_recognition(frame,keypoints)
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        return frame, keypoints, person_message

    def face_recognition(self, keypoint, original_frame, frame):
        body = keypoint[0]
        person_message = []
        if body[0].size == 1:
            return frame,keypoint,None
        else:
            person_num = body.shape[0]
            for i in range(person_num):
                head = np.zeros([5,3],dtype='float32')
                for k in range(5):
                    head[k] = body[i][self.head_points[k]]
                user_name, face_rect = self.face_model(head, original_frame)
                person_message.append([user_name])
                if face_rect:
                    x, y, w, h = face_rect
                    cv2.rectangle(frame, (x - (int)(x / 4), y - (int)(y / 1.5)), (x + w + (int)(x / 4), y + h + (int)(y / 1.5)), (255, 255, 255))
                    cv2.putText(frame, self.name[user_name], (x - (int)(x / 4), y - (int)(y / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        return frame, keypoint, person_message

    def get_skeleton(self, frame):
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])
        frame = datum.cvOutputData
        body = datum.poseKeypoints
        hand = datum.handKeypoints
        face = datum.faceKeypoints
        return frame, (body, hand, face)

    def raise_hand_recognition(self, keypoints, person_message):
        body = keypoints[0]
        #result = False
        if body.size == 1:
            return None
        else:
            person_num = body.shape[0]
            for i in range(person_num):
                result  = self.arm_recognition(body[i])
                person_message[i].append(result)
        return person_message


def control(all_message):
    number = all_message.shape[0]
    name = []
    gesture = []
    result = {0: 'ok', 1: 'one'}
    #name_show = {0: 'WONG Yuk Kit', 6: 'Unknown'}
    for i in range(number):
        counts = np.bincount(all_message[i][:,0])
        name.append(np.argmax(counts))
        counts = np.bincount(all_message[i][:, 2])
        gesture.append(np.argmax(counts))
    if 0 in name:
        j = name.index(0)
        gesture = gesture[j]
        print("WONG Yuk Kit gesture: ", result[gesture])
    else:
        print("Unknown gesture: ", result[gesture[0]])

# Process Image
if __name__ == "__main__":
    url ='rtsp://admin:AdminAdmin@192.168.0.101:554/live/av0'
    #cam = cv2.VideoCapture(url)    #IP camera
    cam = cv2.VideoCapture(0)   # modify here for camera number#
    pose = Openpose()
    #frame = cv2.imread("test.jpg")
    f = open('test.csv', 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    data_record = np.empty([1,40], dtype='int64')
    all_message = np.zeros([1,1,3], dtype='int64')
    thread1 = myThread(1, "timer", 3)
    while (cv2.waitKey(1) != 27):
        # Get camera frame
        ret, frame = cam.read()
        frame, datapose, person_message = pose.update_frame(frame)
        person_message = pose.raise_hand_recognition(datapose, person_message)
        person_number = 0
        if datapose[0].size != 1:
            person_number = datapose[0].shape[0]
        if person_message:
            for i in range(person_number):
                person_message = np.array(person_message,dtype='int64')
                if person_message[i][1] == 0:
                    all_message.resize((person_number, all_message.shape[1], 3))  # 人數，數據數量，數據種類
                    frame, keypoints, person_message = pose.gesture_recognition(frame, datapose, person_message)
                    person_message.resize((person_number, 1, 3))
                    all_message = np.append(all_message,person_message, axis=1)
                    person_message.resize(person_number, 3)
                    #print(all_message)
                    if thread1.thread_status() == False:
                        control(all_message)
                        all_message = np.zeros([1, 1, 3], dtype='int64')
                    if thread1.is_alive() != True:
                        thread1 = myThread(1, "timer", 3)
                        thread1.start()
                        all_message = np.zeros([1, 1, 3], dtype='int64')
                elif (person_message[:, 1] == 1).all():
                    #print('clear')
                    #thread1.stop()
                    all_message = np.zeros([1, 1, 3], dtype='int64')
        """
        if cv2.waitKey(1) & 0xFF == ord('s'):
        #for i in range(3):
            x = vector[1].flatten()
            data_record = np.append(data_record, x[np.newaxis, :], axis=0)
            i+=1
            print("Success! "+ str(i))
        """
        #test = pose.raise_hand_recognition(datapose)
        #print(test)
        cv2.imshow("Openpose", frame)
    # Always clean up

    #csv_writer.writerows(person_message)
    cv2.destroyAllWindows()
