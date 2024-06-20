import argparse
import torch
import numpy as np
import utils1
from mutual_information import mutual_information
from dataset import Dateset_mat
from tqdm import trange
from model import Model, UD_constraint
import torch.distributions.normal as normal
import torch.nn.functional as F
import warnings
from train import run

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", default=r'data', type=str)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=1000)
config = parser.parse_args()

Dataset = Dateset_mat(config.dataset_root)
dataset = Dataset.getdata()
# label = np.array(dataset[3])
label = np.array(dataset[3])
label = np.squeeze(label)
# . = max(label) +1
cluster_num = 7
feature_dim = 60

print("clustering number: ", cluster_num)
text = torch.tensor(dataset[0], dtype=torch.float32).to(device)
video = torch.tensor(dataset[1], dtype=torch.float32).to(device)
audio = torch.tensor(dataset[2], dtype=torch.float32).to(device) #load three modlity

prior_loc = torch.zeros(text.size(0), feature_dim)
prior_scale = torch.ones(text.size(0), feature_dim)
prior = normal.Normal(prior_loc, prior_scale)

def main():
    run(text, video, audio,  cluster_num, label, feature_dim)

if __name__ == '__main__':
    main()

