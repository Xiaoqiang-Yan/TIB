import scipy.io
import torch
import numpy as np
import random


class Dateset_mat():
    def __init__(self, data_path):

        # self.audio = np.load(data_path + r"/audio.npy")
        # self.video = np.load(data_path + r"/video.npy")
        # self.txt = np.load(data_path + r"/text.npy")
        self.audio = np.load(data_path + r"/audio.npy")
        self.video = np.load(data_path + r"/video.npy")
        self.txt = np.load(data_path + r"/text.npy")
        # self.txt = scipy.io.loadmat(data_path + r"/txt.mat")

        try:
            self.label = np.load(data_path + r"/labels.npy")
            # self.label = scipy.io.loadmat(data_path + r"/label_7.mat")
        except:
            self.label = scipy.io.loadmat(data_path + r"/L.mat")

    def getdata(self):
        self.data = []
        self.data.append(self.txt)
        self.data.append(self.video)
        self.data.append(self.audio)
        self.data.append(self.label)
        # self.data.append(self.img["img"])
        # self.data.append(self.txt["txt"])
        # self.data.append(self.label["L"])
        fix_seed(358)
        return self.data

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


