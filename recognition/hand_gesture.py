import csv
import numpy as np
import cv2
import time
from sklearn import svm


class HandModel():
    def __init__(self):
        self.AllData = np.zeros([1, 40])
        self.AllLabel = []
        self.point_pairs = [[0, 1], [1, 2], [2, 3], [3, 4],
                            [0, 5], [5, 6], [6, 7], [7, 8],
                            [0, 9], [9, 10], [10, 11], [11, 12],
                            [0, 13], [13, 14], [14, 15], [15, 16],
                            [0, 17], [17, 18], [18, 19], [19, 20]]
        self.idx_to_gesture = ["ok","one"]
        self.SVM = self.TrainSVMModel()

    def __call__(self, hand, i):
        hand = self.data_process(hand, i)
        result = self.SVM.predict(hand)
        probability = self.SVM.predict_proba(hand)
        return result[0], probability

    def TrainSVMModel(self):
        # Prepare positive data
        with open('new.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            vector = np.zeros([1, 40])
            for row in csv_reader:
                row = np.array(row)
                row = row[np.newaxis, :]
                vector = np.append(vector, row, axis=0)
                self.AllLabel.append(0)
            vector = np.delete(vector, 0, axis=0)
            vector = np.array(vector, dtype='float32')
            self.AllData = np.append(self.AllData, vector, axis=0)
            self.AllData = np.delete(self.AllData, 0, axis=0)
        # Prepare negative data
        with open('one.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            vector = np.zeros([1, 40])
            for row in csv_reader:
                row = np.array(row)
                row = row[np.newaxis, :]
                vector = np.append(vector, row, axis=0)
                self.AllLabel.append(1)
            vector = np.delete(vector, 0, axis=0)
            vector = np.array(vector, dtype='float32')
            self.AllData = np.append(self.AllData, vector, axis=0)
        SVM = svm.SVC(kernel='poly', gamma=10, probability = True)
        SVM.fit(self.AllData, self.AllLabel)
        return SVM

    def data_process(self, hand, i):
        get_hand = []
        if hand[0].size == 1:
            return np.zeros((2,20,2), dtype='float32')
        get_hand.append(hand[1][0][i])
        get_hand.append(hand[1][1][i])
        get_hand = self.point_to_vector(get_hand)
        x = get_hand[1].flatten()
        get_hand = x[np.newaxis, :]
        return get_hand

    def unit_vector(self,hand):
        for i in range(0,2):
            for j in range(0,20):
                norm = pow(pow(hand[i][j][0], 2) + pow(hand[i][j][1], 2), 0.5)
                if norm!=0:
                    norm = 1/norm
                hand[i][j][0] = norm * hand[i][j][0]
                hand[i][j][1] = norm * hand[i][j][1]
        return hand

    def point_to_vector(self, hand):
        vector = np.ones((2,20,2), dtype='float32')
        for j in range(0, 2):
            for i in range(0,20):
                vector[j][i][0] = hand[j][self.point_pairs[i][0]][0] - hand[j][self.point_pairs[i][1]][0]
                vector[j][i][1] = hand[j][self.point_pairs[i][0]][1] - hand[j][self.point_pairs[i][1]][1]
        vector = self.unit_vector(vector)
        return vector

    @staticmethod
    def hand_bbox(hand):
        if np.sum(hand[:, 2]) > 21 * 0.5:
            rect = cv2.boundingRect(np.array(hand[:, :2]))
            return rect
        else:
            return None