import csv
import numpy as np
with open('new.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    lable = []
    vector = np.zeros([1,40])
    for row in csv_reader:
        row = np.array(row)
        row = row[np.newaxis, :]
        vector = np.append(vector,row,axis=0)
        lable.append(0)  # Stand case with label 0
        line_count += 1
        print(vector)
    print(f'Add {line_count} records as Negative samples. (Stand)')
    vector = np.delete(vector, 0, axis=0)
    vector = np.array(vector,dtype='float32')
    print(vector)
    NumOfNegativeSamples = line_count
