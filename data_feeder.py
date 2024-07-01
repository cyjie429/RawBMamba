import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import feature_extract
import os

def load_label(label_file):
    labels = {}
    wav_lists = []
    encode = {'spoof': 0, 'bonafide': 1}
    with open(label_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_id = line[1]
                wav_lists.append(wav_id)
                tmp_label = encode[line[4]]
                labels[wav_id] = tmp_label

    return labels, wav_lists


class ASVDataSet(Dataset):

    def __init__(self, data, label, wav_ids=None, transform=True, mode="train", lengths=None, feature_type="fft"):
        super(ASVDataSet, self).__init__()
        self.data = data
        self.label = label
        self.wav_ids = wav_ids
        self.transform = transform
        self.lengths = lengths
        self.mode = mode
        self.feature_type=feature_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        each_data, each_label = self.data[idx], self.label[idx]
        each_data=feature_extract.extract(each_data, self.feature_type)
        if self.transform:
            each_data=torch.Tensor(each_data)
        return each_data, each_label


# this will load data by wav
def load_data(dataset, label_file, mode="train", feature_type="fft"):
    if mode!="eval":
        data, label=load_train_data(dataset, label_file, feature_type="fft")
        return data,label
    else:
        data, folder_list, flag = load_eval_data(dataset, label_file, feature_type="fft")
        # Path to the ASVspoof2019LA evaluation set WAV files
        folder = "/data2/xxx/data/ASVspoof2019LA/eval/wav/"
        return data, folder_list, flag, folder


def load_train_data(dataset, label_file, feature_type="fft"):
    labels, wav_lists = load_label(label_file)
    final_data = []
    final_label = []

    for wav_id in tqdm(wav_lists, desc="load {} data".format(dataset)):
        label = labels[wav_id]
        
        if "T" in wav_id:
            #  Path to the ASVspoof2019LA train set WAV files
            wav_path = "/data2/xxx/data/ASVspoof2019LA/train/wav/{}.wav".format(wav_id)
        if "D" in wav_id:
            #  Path to the ASVspoof2019LA development set WAV files
            wav_path = "/data2/xxx/data/ASVspoof2019LA/dev/wav/{}.wav".format(wav_id)
        
        if os.path.exists(wav_path):
            final_data.append(wav_path)
            final_label.append(label)
        else:
            print("can not open {}".format(wav_path))
        
    return final_data, final_label


def load_eval_data(dataset, scp_file, feature_type="fft"):
    wav_lists = []
    folder_list={}
    flag = {}
    with open(scp_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_id = line[1]
                wav_lists.append(wav_id)
                folder_list[wav_id]=line[-1]
                if line[-2] == '-':
                    flag[wav_id] = 'A00'
                else:
                    flag[wav_id] = line[-2]
    return wav_lists, folder_list, flag


def main():
    labels, wav_lists=load_label("ASVspoof2019.LA.cm.train.trl.txt")
    wav_list=wav_lists[4000:4005]
    for wav_id in wav_list:
        print(wav_id)
        print(labels[wav_id])

if __name__ == '__main__':
    main()
