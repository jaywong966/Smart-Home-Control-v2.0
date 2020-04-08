import face_recognition
import cv2
import numpy as np

class Facemodel():
    # Load a sample picture and learn how to recognize it.
    def __init__(self):
        obama_image = face_recognition.load_image_file("obama.jpg")
        obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

        # Load a second sample picture and learn how to recognize it.
        biden_image = face_recognition.load_image_file("biden.jpg")
        biden_face_encoding = face_recognition.face_encodings(biden_image)[0]


        # Create arrays of known face encodings and their names
        self.known_face_encodings = [
            obama_face_encoding,
            biden_face_encoding,
        ]
        self.known_face_names = [
            "Barack Obama",
            "Joe Biden",
        ]
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

    def __call__(self, head, frame):
        frame, face_rect = self.face_bbox(head, frame)
        # Resize frame of video to 1/4 size for faster face recognition processing
        #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        frame = frame[:, :, ::-1]
        # Only process every other frame of video to save time
        if self.process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            self.face_locations = face_recognition.face_locations(frame)
            self.face_encodings = face_recognition.face_encodings(frame, self.face_locations)
            name = "Unknown"
            for face_encoding in self.face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                #face_names.append(name)
        return name, face_rect


    def face_bbox(self, head, frame):
        if np.sum(head[:, 2]) > 5 * 0.5:
            face_rect = cv2.boundingRect(np.array(head[:, :2]))
            x, y, w, h = face_rect
            frame = frame[y - (int)(y / 1.5): y + h + (int)(y / 1.5), x - (int)(x / 4): x + w + (int)(x / 4)]
            #image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            return frame, face_rect
        else:
            face_rect = ()
            return frame, face_rect

