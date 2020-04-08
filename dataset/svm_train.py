import csv
import numpy as np
from sklearn import svm

class SVM_Train():
    def TrainSVMModel(self):
        # Prepare all the data
        AllData = np.zeros([1,40])
        AllLabel = []
        NumOfPostiveSamples = 0
        NumOfNegativeSamples = 0
        # Prepare positive data
        with open('new.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            vector = np.zeros([1, 40])
            for row in csv_reader:
                row = np.array(row)
                row = row[np.newaxis, :]
                vector = np.append(vector, row, axis=0)
                AllLabel.append(0)  # Stand case with label 0
                line_count += 1
            print(f'Add {line_count} records as Negative samples. (Stand)')
            vector = np.delete(vector, 0, axis=0)
            vector = np.array(vector, dtype='float32')
            AllData = np.append(AllData,vector,axis=0)
            AllData = np.delete(AllData, 0, axis=0)
            NumOfPostiveSamples = line_count
        # Prepare negative data
        with open('one.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            vector = np.zeros([1, 40])
            for row in csv_reader:
                row = np.array(row)
                row = row[np.newaxis, :]
                vector = np.append(vector, row, axis=0)
                AllLabel.append(1)  # Stand case with label 0
                line_count += 1
            print(f'Add {line_count} records as Negative samples. (Stand)')
            vector = np.delete(vector, 0, axis=0)
            vector = np.array(vector, dtype='float32')
            AllData = np.append(AllData, vector, axis=0)
            NumOfNegativeSamples = line_count
        print(f'NumOfPostiveSamples = {NumOfPostiveSamples} \nNumOfNegativeSamples = {NumOfNegativeSamples}')
        # ***************************
        # Divide the data into two groups: TrainingData and TestingData
        trainingDataPorportion = 0.7

        AllDataSize = len(AllData)
        print(f'AllDataSize = {AllDataSize}')
        if len(AllLabel) != AllDataSize:
            print(f'AllLabel length != AllDataSize #######################')
            #raise SystemExit(f'Problem')

        # Make a list of randomized index
        np.random.seed(0)
        randomizedIndex = np.random.permutation(AllDataSize)  # Generate a list of random number. All the numbers are from 0 to AllDataSize - 1
        # print(randomizedIndex)

        trainingDataSize = (int)(AllDataSize * trainingDataPorportion)

        TrainingData = []
        TestingData = []
        TrainingDataLabel = []
        TestingDataLabel = []

        # Randomly sample the AllData set to get the set of training data
        TrainingDataPosSamplesSize = 0
        TrainingDataNegSamplesSize = 0
        i = 0
        while i < trainingDataSize:
            TrainingData.append(AllData[randomizedIndex[i]])
            TrainingDataLabel.append(AllLabel[randomizedIndex[i]])
            if AllLabel[randomizedIndex[i]] == 1:
                TrainingDataPosSamplesSize += 1
            else:
                TrainingDataNegSamplesSize += 1
            i += 1

        TrainingDataSize = len(TrainingData)
        if TrainingDataSize != len(TrainingDataLabel):
            print(f'len(TrainingData) != len(TrainingDataLabel) #######################')
            raise SystemExit(f'Problem')

        print(f'Training data size = {TrainingDataSize}')
        print(f'Training data positive samples size = {TrainingDataPosSamplesSize}')
        print(f'Training data negative samples size = {TrainingDataNegSamplesSize}')

        # Randomly sample the AllData set to get the set of testing data (No overlapping between TrainingData and TestingData)
        TestingDataPosSamplesSize = 0
        TestingDataNegSamplesSize = 0
        i = trainingDataSize
        while i < AllDataSize:
            TestingData.append(AllData[randomizedIndex[i]])
            TestingDataLabel.append(AllLabel[randomizedIndex[i]])
            if AllLabel[randomizedIndex[i]] == 1:
                TestingDataPosSamplesSize += 1
            else:
                TestingDataNegSamplesSize += 1
            i += 1

        TestingDataSize = len(TestingData)
        if TestingDataSize != len(TestingDataLabel):
            print(f'len(TestingData) != len(TestingDataLabel) #######################')
            raise SystemExit(f'len(TestingData) != len(TestingDataLabel)')

        print(f'Testing data size = {TestingDataSize}')
        print(f'Testing data positive samples size = {TestingDataPosSamplesSize}')
        print(f'Testing data negative samples size = {TestingDataNegSamplesSize}')

        # ***************************************************************************
        # Train the SVM model
        # fit the model - training
        SVM = svm.SVC(kernel='poly', gamma=10)
            # sys.exit
        #print(TrainingData)
        #print(TrainingDataLabel)
        SVM.fit(TrainingData, TrainingDataLabel)
        # ***************************************************************************
        # Test the accuracy of the trained model using testing data
        Result = SVM.predict(TestingData)  # Do prediction for each element in TestingData
        TruePositiveCount = 0
        FalsePositiveCount = 0
        TrueNegativeCount = 0
        FalseNegativeCount = 0

        i = 0
        while i < TestingDataSize:
            if Result[i] == TestingDataLabel[i]:  # True case
                if Result[i] == 1:  # Positive case
                    TruePositiveCount += 1
                else:
                    TrueNegativeCount += 1  # Negative case
            else:  # False case
                if Result[i] == 1:  # Positive case
                    FalsePositiveCount += 1
                else:
                    FalseNegativeCount += 1  # Negative case
            i += 1

        print(f'*********** Testing data set ***************')
        print(f'True Positive Count = {TruePositiveCount}')
        print(f'True Negative Count = {TrueNegativeCount}')
        print(f'False Positive Count = {FalsePositiveCount}')
        print(f'False Negative Count = {FalseNegativeCount}')

        Precision = TruePositiveCount / (TruePositiveCount + FalsePositiveCount)
        Recall = TruePositiveCount / (TruePositiveCount + FalseNegativeCount)

        print(f'Number of testing data = {TestingDataSize}')
        print('Precision = {:0.3f}'.format(Precision))
        print('Recall = {:0.3f}'.format(Recall))
        # ***************************************************************************
        # Test the accuracy of the trained model using training data
        Result = SVM.predict(TrainingData)  # Do prediction for each element in TrainingData
        TruePositiveCount = 0
        FalsePositiveCount = 0
        TrueNegativeCount = 0
        FalseNegativeCount = 0

        i = 0
        while i < TrainingDataSize:
            if Result[i] == TrainingDataLabel[i]:  # True case
                if Result[i] == 1:  # Positive case
                    TruePositiveCount += 1
                else:
                    TrueNegativeCount += 1  # Negative case
            else:  # False case
                if Result[i] == 1:  # Positive case
                    FalsePositiveCount += 1
                else:
                    FalseNegativeCount += 1  # Negative case
            i += 1

        print(f'*********** Training data set ***************')
        print(f'True Positive Count = {TruePositiveCount}')
        print(f'True Negative Count = {TrueNegativeCount}')
        print(f'False Positive Count = {FalsePositiveCount}')
        print(f'False Negative Count = {FalseNegativeCount}')

        Precision = TruePositiveCount / (TruePositiveCount + FalsePositiveCount)
        Recall = TruePositiveCount / (TruePositiveCount + FalseNegativeCount)

        print(f'Number of training data = {TrainingDataSize}')
        print('Precision = {:0.3f}'.format(Precision))
        print('Recall = {:0.3f}'.format(Recall))
        return SVM


# ***************************************************
"""
if __name__ == '__main__':
    SVM_Train = SVM_Train()
    SVM = SVM_Train.TrainSVMModel()  # kernel 'linear' 'poly' 'rbf'
    results = []
    with open('test.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        vector = np.zeros([1, 40])
        for row in csv_reader:
            row = np.array(row)
            row = row[np.newaxis, :]
            result = SVM.predict(row)
            results.append(result)
        print(results)
        """
