from email.mime import base
import torch
import torch.nn as nn
import numpy as np
import librosa
import joblib
import warnings
import base64

warnings.filterwarnings('ignore')


def loadAudioFiles(file_name, sample_rate=48000):

    audio_signal, sample_rate = librosa.load(
        file_name, duration=3, offset=0.5, sr=sample_rate)

    signal = np.zeros(int(sample_rate*3,))
    signal[:len(audio_signal)] = audio_signal

    test_data = []

    test_data.append(signal)
    return test_data


def calculateMelSpectrogram(audio, sample_rate):
    mel_spec = librosa.feature.melspectrogram(y=audio,
                                              sr=sample_rate,
                                              n_fft=1024,
                                              win_length=512,
                                              window='hamming',
                                              hop_length=256,
                                              n_mels=128,
                                              fmax=sample_rate/2
                                              )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


class ParallelModel(nn.Module):
    def __init__(self, num_emotions):
        super().__init__()
        ###-------------------코드 작성해야하는 부분---------------------
        #가이드라인1. 위에 model architecture table을 보고 각 스테이지를 구현.
        # **tip : 각 stage별로 nn.Sequential 함수로 묶어서 사용하면 모델 변경에 용이함.
        
        # 1. 1stage (Conv + BatchNorm + ReLU + Maxpooling + Dropout)
        self.c1 = torch.nn.Conv2d(1,16,kernel_size=3,stride=1,padding=1)
        self.b1 = torch.nn.BatchNorm2d(16)
        self.r1 = torch.nn.ReLU()
        self.s1 = torch.nn.MaxPool2d(kernel_size=2,stride=2)
        self.d1 = torch.nn.Dropout(0)
        
        
        # 2. 2stage (Conv + BatchNorm + ReLU + Maxpooling + Dropout)
        self.c2 = torch.nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1)
        self.b2 = torch.nn.BatchNorm2d(32)
        self.r2 = torch.nn.ReLU()
        self.s2 = torch.nn.MaxPool2d(kernel_size=4,stride=4)
        self.d2 = torch.nn.Dropout(0.05)
        
        
        

        # 3. 3stage (Conv + BatchNorm + ReLU + Maxpooling + Dropout)
        self.c3 = torch.nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.b3 = torch.nn.BatchNorm2d(64)
        self.r3 = torch.nn.ReLU()
        self.s3 = torch.nn.MaxPool2d(kernel_size=4,stride=4)
        self.d3 = torch.nn.Dropout(0.1)
        
        
        

        # 4. 4stage (Conv + BatchNorm + ReLU + Maxpooling + Dropout)

        self.c4 = torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1)
        self.b4 = torch.nn.BatchNorm2d(128)
        self.r4 = torch.nn.ReLU()
        self.s4 = torch.nn.MaxPool2d(kernel_size=4,stride=4)
        self.d4 = torch.nn.Dropout(0.2)

        # Linear softmax layer
        
        self.fc1 = torch.nn.Linear(512, 128)
        self.fc2 = torch.nn.Linear(128, 32)
        self.fc3 = torch.nn.Linear(32, num_emotions)
        
    
    def forward(self,x):
        out = self.r1(self.b1(self.c1(x)))
        out = self.d1(self.s1(out))

        out = self.r2(self.b2(self.c2(out)))
        out = self.d2(self.s2(out))
        
        out = self.r3(self.b3(self.c3(out)))
        out = self.d3(self.s3(out))
        
        out = self.r4(self.b4(self.c4(out)))
        out = self.d4(self.s4(out))
        
        out = out.view(out.size(0),-1) # batch x 1 x 1 x 120  
        
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        output = out
        
        return output


def getEmotion(filename):

    scaler = joblib.load("scaler.joblib")
    le = joblib.load("labelEncoder.joblib")
    model = torch.load('model.pt', map_location='cpu')

    test_data = loadAudioFiles(filename)

    test_data = calculateMelSpectrogram(test_data[0], sample_rate=48000)

    h, w = test_data.shape

    test_data = np.reshape(test_data, newshape=(1, -1))
    test_data = scaler.transform(test_data)
    test_data = np.reshape(test_data, newshape=(1, 1, h, w))

    X = torch.Tensor(test_data)
    hypothesis = model(X)
    predict = torch.argmax(hypothesis, dim=1)
    predict = predict.numpy()
    predict = le.inverse_transform(predict)

    return predict[0]


def predict(wav_base64=None):

    file = 'sample.wav'
    if wav_base64 is not None:

        tmp = open('tmp.wav', 'wb')
        bfile = base64.b64decode(wav_base64)
        tmp.write(bfile)
        tmp.close()

        file = open('tmp.wav', 'rb')
        emotion = getEmotion(file)

    return emotion


if __name__ == '__main__':
    predict()
