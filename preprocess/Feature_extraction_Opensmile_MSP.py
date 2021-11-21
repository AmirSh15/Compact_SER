import numpy as np
import os
import sys
import csv
from scipy.stats import kurtosis, skew

def csv_reader(add):
    with open(outfilename, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        headers = next(reader)
        data = np.array(list(reader)).astype(float)
        
    return data[:,2:] 


#### Load dat
emotions_used = { 'A':0, 'H':1, 'N':2, 'S':3 }
emotions_used_comp = {'Neutral;':2, 'Anger;':0, 'Sadness;':3, 'Happiness;':1}
data_path = "path_to_the_MSP_directory"
sessions = ['session1', 'session2', 'session3', 'session4', 'session5', 'session6']
framerate = 44100

Label = []
Data = []
fix_len = 120

# openSMILE
exe_opensmile = "path_to_the_opensmile_exe_fileee"
path_config   = "path_to_the_opensmile_config_file"


for ses in sessions:
    emt_label_path = data_path + ses
    for file in os.listdir(emt_label_path):
        for file2 in os.listdir(emt_label_path+'/'+file):
            if file2=='S':
                sorted_list = []
                file_list = os.listdir(emt_label_path + '/' + file + '/' + file2)
                file_list.sort()
                while(len(file_list)!=0):
                    sorted_list.append(file_list[0])
                    cand = file_list[0].split('-')[-1]
                    for fil in file_list[1:]:
                        if cand in fil:
                            sorted_list.append(fil)
                            file_list.remove(fil)
                            break
                    file_list.remove(file_list[0])
                Graph = []
                Graph_label = []
                for file3 in sorted_list:
                    wav_path = emt_label_path + '/' + file + '/' + file2 + '/' + file3
                    ### Reading Emotion labels
                    label = file[-1]

                    if (label in emotions_used):
                        infilename = wav_path
                        outfilename = "MSP.csv"
                        opensmile_call = exe_opensmile + " -C " + path_config + " -I " + infilename + " -O " + outfilename + " -l 0"
                        os.system(opensmile_call)

                        MFCC = csv_reader(outfilename)
                        mean = np.mean(MFCC, axis=0)
                        max = np.max(MFCC, axis=0)
                        std = np.std(MFCC, axis=0)
                        sk = skew(MFCC, axis=0)
                        kurt = kurtosis(MFCC, axis=0)

                        Graph.append(np.concatenate([mean, max, std, sk, kurt]))
                        label = emotions_used[label]
                        Graph_label.append(label)
                Data.extend(Graph)
                Label.extend(Graph_label)




# Save Graph data
np.save('data_MSP_open.npy', np.array(Data))
np.save('label_MSP_open.npy', np.array(Label))