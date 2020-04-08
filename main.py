import sys
import cv2
import os
from sys import platform
import csv
import time
#from recognition.gesture_model import GestureModel
from recognition.face_model import Facemodel
from recognition.hand_gesture import HandModel
import numpy as np
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
params["net_resolution"] = "96x96"
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
        self.head_points = [0, 15, 16, 17, 18]

    def gesture_recognition(self, frame, keypoints):
        hands = keypoints[1]
        if hands[0].size == 1:
            return frame
        else:
            person_num = hands[0].shape[0]
            for i in range(person_num):
                for hand in hands:
                    rect = self.gesture_model.hand_bbox(hand[i])
                    gesture = self.gesture_model(keypoints,i)
                    result = str(gesture)
                    if rect:
                        x, y, w, h = rect
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255))
                        cv2.putText(frame, result, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        return frame


    def update_frame(self, frame):
        original_frame = frame
        start_time = time.time()
        frame, keypoints = self.get_skeleton(frame)
        if keypoints[0].size == 1:
            return frame, keypoints
        frame, keypoints = self.face_recognition(keypoints, original_frame, frame)
        frame = self.gesture_recognition(frame,keypoints)
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        return frame, keypoints

    def face_recognition(self, keypoint, original_frame, frame):
        body = keypoint[0]
        if body[0].size == 1:
            return frame,keypoint
        else:
            person_num = body.shape[0]
            for i in range(person_num):
                head = np.zeros([5,3],dtype='float32')
                for k in range(5):
                    head[k] = body[i][self.head_points[k]]
                user_name, face_rect = self.face_model(head, original_frame)
                if face_rect:
                    x, y, w, h = face_rect
                    cv2.rectangle(frame, (x - (int)(x / 4), y - (int)(y / 1.5)), (x + w + (int)(x / 4), y + h + (int)(y / 1.5)), (255, 255, 255))
                    cv2.putText(frame, user_name, (x - (int)(x / 4), y - (int)(y / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        return frame, keypoint

    def get_skeleton(self, frame):
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])
        frame = datum.cvOutputData
        body = datum.poseKeypoints
        hand = datum.handKeypoints
        face = datum.faceKeypoints
        return frame, (body, hand, face)

# Process Image
if __name__ == "__main__":
    cam = cv2.VideoCapture(0)  # modify here for camera number#
    pose = Openpose()
    frame = cv2.imread("test.jpg")
    f = open('test.csv', 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    data_record = np.empty([1,40],dtype='float32')
    while (cv2.waitKey(1) != 27):
        # Get camera frame
        ret, frame = cam.read()
        frame, datapose = pose.update_frame(frame)
        """
        if cv2.waitKey(1) & 0xFF == ord('s'):
        #for i in range(3):
            x = vector[1].flatten()
            data_record = np.append(data_record, x[np.newaxis, :], axis=0)
            i+=1
            print("Success! "+ str(i))
        """
        cv2.imshow("Openpose", frame)
    # Always clean up
    #csv_writer.writerows(data_record)
    cv2.destroyAllWindows()
