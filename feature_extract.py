import librosa
import random
import math
import numpy as np
from scipy import signal
#import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
#from Copy import true_wav_path
# parameters
sample_rate = 8000
win_length = 1728
hop_length = 130

def process_Rawboost_feature(feature, sr, algo=3,nBands = 5,minF = 20,maxF = 8000,minBW = 100,maxBW = 1000,minCoeff = 10,maxCoeff = 100,minG = 0,maxG = 0,minBiasLinNonLin = 5,maxBiasLinNonLin = 20,N_f = 5,P = 10,g_sd = 2,SNRmin = 10,SNRmax = 40):
    # Data process by Convolutive noise (1st algo)
    if algo == 1:
        feature = LnL_convolutive_noise(feature, N_f, nBands, minF, maxF, minBW, maxBW,
                                        minCoeff, maxCoeff, minG, maxG, minBiasLinNonLin,
                                        maxBiasLinNonLin, sr)

    # Data process by Impulsive noise (2nd algo)
    elif algo == 2:

        feature = ISD_additive_noise(feature, P, g_sd)


    # Data process by coloured additive noise (3rd algo)
    elif algo == 3:

        feature = SSI_additive_noise(feature, SNRmin, SNRmax, nBands, minF, maxF, minBW,
                                     maxBW, minCoeff, maxCoeff, minG, maxG, sr)

    # Data process by all 3 algo. together in series (1+2+3)
    elif algo == 4:

        feature = LnL_convolutive_noise(feature, N_f, nBands, minF, maxF, minBW, maxBW,
                                        minCoeff, maxCoeff, minG, maxG, minBiasLinNonLin,
                                        maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, P, g_sd)
        feature = SSI_additive_noise(feature, SNRmin, SNRmax, nBands, minF, maxF, minBW,
                                     maxBW, minCoeff, maxCoeff, minG, maxG, sr)

        # Data process by 1st two algo. together in series (1+2)
    elif algo == 5:

        feature = LnL_convolutive_noise(feature, N_f, nBands, minF, maxF, minBW, maxBW,
                                        minCoeff, maxCoeff, minG, maxG, minBiasLinNonLin,
                                        maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, P, g_sd)

        # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo == 6:

        feature = LnL_convolutive_noise(feature, N_f, nBands, minF, maxF, minBW, maxBW,
                                        minCoeff, maxCoeff, minG, maxG, minBiasLinNonLin,
                                        maxBiasLinNonLin, sr)
        feature = SSI_additive_noise(feature, SNRmin, SNRmax, nBands, minF, maxF, minBW,
                                     maxBW, minCoeff, maxCoeff, minG, maxG, sr)

        # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo == 7:

        feature = ISD_additive_noise(feature, P, g_sd)
        feature = SSI_additive_noise(feature, SNRmin, SNRmax, nBands, minF, maxF, minBW,
                                     maxBW, minCoeff, maxCoeff, minG, maxG, sr)

        # Data process by 1st two algo. together in Parallel (1||2)
    elif algo == 8:

        feature1 = LnL_convolutive_noise(feature, N_f, nBands, minF, maxF, minBW, maxBW,
                                         minCoeff, maxCoeff, minG, maxG, minBiasLinNonLin,
                                         maxBiasLinNonLin, sr)
        feature2 = ISD_additive_noise(feature, P, g_sd)

        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)  # normalized resultant waveform

    # original data without Rawboost processing
    else:

        feature = feature

    return feature

def extract(wav_path,  feature_type):
    if feature_type == "cqt":
        return extract_cqt(wav_path)
    if feature_type == "fft":
        return extract_fft(wav_path)
    if feature_type == "f0":
        return extract_f0(wav_path)

def extract_cqt(wav_path):
    y, fs = librosa.load(wav_path, sr=sample_rate)
    cqt = librosa.cqt(y, fs, hop_length=hop_length, fmin=fs/(2^10), bins_per_octave=96)
    total=cqt.shape[1]
    if total<400:
        for j in range(0,400//total):
            if j==0:
                cqt=np.hstack((cqt,np.fliplr(cqt)))
            else:
                cqt=np.hstack((cqt,cqt))
    feature=cqt[:,0:400]
    #feature=feature[432:865,:]
    return np.reshape(np.array(feature),(-1,84,400))



def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def extract_fft(wav_path):
    p_preemphasis = 0.97
    num_freq = 1728

    def preemphasis(x):
        return signal.lfilter([1, -p_preemphasis], [1], x)

    def _stft(y):
        return librosa.stft(y=y, n_fft=num_freq, hop_length=hop_length, win_length=win_length, window=signal.windows.blackman)

    y, fs = librosa.load(wav_path, sr=None)
    if(len(y)>80000):
        y=y[0:80000]
    y = pad(y)
    #print(y.shape)
    # y = process_Rawboost_feature(y, fs, algo=7,nBands = 5,minF = 20,maxF = 8000,minBW = 100,maxBW = 1000,minCoeff = 10,maxCoeff = 100,minG = 0,maxG = 0,minBiasLinNonLin = 5,maxBiasLinNonLin = 20,N_f = 5,P = 10,g_sd = 2,SNRmin = 10,SNRmax = 40 )
    #D = _stft(preemphasis(y))
    ##print(D.shape)
    #S = np.log(abs(D)+np.exp(-80))
    ##feature=S
    ##height=S.shape[0]
    #total=S.shape[1]
    #if total<600:
    #    for j in range(0,600//total):
    #        if j==0:
    #            S=np.hstack((S,np.fliplr(S)))
    #        else:
    #            S=np.hstack((S,S))
    #            if S.shape[1]>=600:
    #                break
    #feature=S[0:45,0:600]
    #plt.imshow(feature)
    #plt.show()
    #print(feature.shape)
    # feature=feature[432:,:]
    #feature = feature[0:45, :]
    #feature = torch.Tensor(feature)
    # print(feature.shape)
    #f1 = Mask(feature)
    #f2 = crop(feature, 4, 1)
    #feature = torch.cat((feature, f1), 0)
    #feature = torch.cat((feature, f2), 0)
    #feature = Mask(feature, label)
    #mask = Mask(feature)
    #mask = torch.Tensor(np.reshape(np.array(feature), (-1, 45, 600)))
    #feature = np.reshape(np.array(feature), (-1, 45, 600))

    #Crop = torch.Tensor(np.reshape(np.array(feature),(-1,45,600)))
    #Crop = crop(Crop, 3, 1)
    #return feature#,mask ,Crop
    return torch.Tensor(np.reshape(np.array(y), (-1, 64600)))




def extract_f0(wav_path):
    sound, _ = librosa.load(wav_path, sr=16000)
    #print(f'sound.shape = {sound.shape}')  # sound.shape = (80000,)

    sr = 16000
    # 输入sound 需要为 double类型 librosa load 的waveform 是 float32
    #print(f'sound.dtype = {sound.dtype}')  # sound.dtype = float32
    sound = sound.astype(np.double)

    # 第一种
    _f0, t = pw.dio(sound, sr)  # raw pitch extractor
    # print("_f0", _f0)
    # print("_f0:", _f0.shape)
    # print("t", t)
    # print("t:", t.shape)
    f0 = pw.stonemask(sound, _f0, t, sr)  # pitch refinement
    # print("f0", f0)
    # print("f0:", f0.shape)
    # print(t)
    # print(t.shape)
    # S = np.stack((f0, t), axis=0)
    # print("S", S)
    # print(S.shape)
    f0 = np.reshape(np.array(f0), (1, -1))
    # print(f0.shape)

    S = f0

    total = S.shape[1]
    if total < 600:
        for j in range(0, 600 // total):
            if j == 0:
                S = np.hstack((S, np.fliplr(S)))
            else:
                S = np.hstack((S, S))
                if S.shape[1] >= 600:
                    break
    feature = S[:, 0:600]
    # print(feature)
    return np.reshape(np.array(feature), (-1, 1, 600))
    # return feature

def extract_ap(wav_path):
    sound, _ = librosa.load(wav_path, sr=16000)
    #print(f'sound.shape = {sound.shape}')  # sound.shape = (80000,)

    sr = 16000
    # 输入sound 需要为 double类型 librosa load 的waveform 是 float32
    #print(f'sound.dtype = {sound.dtype}')  # sound.dtype = float32
    sound = sound.astype(np.double)

    # 第一种
    _f0, t = pw.dio(sound, sr)  # raw pitch extractor
    f0 = pw.stonemask(sound, _f0, t, sr)  # pitch refinement
    # 第二种
    #f0, timeaxis = pw.harvest(sound, sr)

    # print(f'f0.shape = {f0.shape}')  # f0.shape = (1001,)
    # python
    # f0_length = GetSamplesForHarvest(fs, x_length, option.frame_period)
    # 提取非周期特征AP
    fft_size = 1728
    ap = pw.d4c(sound, f0, t, sr, fft_size=fft_size)
    print(ap.shape)
    ap=ap.T
    total=ap.shape[1]
    if total<600:
        for j in range(0,600//total):
            if j==0:
                ap=np.hstack((ap,np.fliplr(ap)))
            else:
                ap=np.hstack((ap,ap))
                if ap.shape[1]>=600:
                    break
    feature = ap[:, 0:600]
    # feature = feature.T
    return np.reshape(np.array(feature), (-1, 865, 600))



def main():
    #c=extract_cqt('LA_T_1000137.flac')
    #print(c.shape)
    r=extract_fft('/datahuman_dataset/ASVspoof/ASVspoof2019_LA_train/LA_T_1000137.wav')
    print(r.shape)
    return r

if __name__ == '__main__':
    main()

# def to_one_hot(self, x):
#         # e = de_norm_mean_std(e, hp.e_mean, hp.e_std)
#         # For pytorch > = 1.6.0
#
#     quantize = torch.bucketize(x, self.energy_bins).to(device=x.device)  # .cuda()
#     return F.one_hot(quantize.long(), 256).float()
