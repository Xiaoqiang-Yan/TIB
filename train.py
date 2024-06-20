import argparse

import matplotlib.pyplot as plt
import torch
import numpy as np
import utils1
from mutual_information import mutual_information
from dataset import Dateset_mat
from tqdm import trange
from model1 import Model, UD_constraint
import torch.distributions.normal as normal
import torch.nn.functional as F
import warnings
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#
# parser = argparse.ArgumentParser()
# parser.add_argument("--dataset_root", default=r'data', type=str)
# parser.add_argument("--lr", type=float, default=0.0001)
# parser.add_argument("--num_epochs", type=int, default=1000)
# config = parser.parse_args()


criterion = torch.nn.CrossEntropyLoss().to(device)


def t_sne(feature, cluster_num, x_out, epoch):
    tsne = TSNE(n_components=2, init='pca', verbose=1, random_state=402)
    feature = feature.detach().cpu().numpy()
    tsne_results = tsne.fit_transform(feature)

    plt.figure()
    for i in range(cluster_num):
        indices = np.where(x_out == i)
        indices = indices[0]
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=i)
        font = {'size': 10}
    plt.legend(ncol=7, loc=(0.005, 1.01), prop=font)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlabel('特征', size=12)
    plt.savefig(f'{epoch}.png', dpi=100, bbox_inches='tight')
    plt.show()


def run(modal1, modal2, modal3, cluster_num, label, feature_dim):
    max_ACC = 0
    max_nmi = 0
    model = Model(cluster_num)
    model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=0.05)

    prior_loc = torch.zeros(modal1.size(0), feature_dim)
    prior_scale = torch.ones(modal2.size(0), feature_dim)
    prior = normal.Normal(prior_loc, prior_scale)  # sample martrix

    for epoch in trange(500):
        model.train()
        model.zero_grad()
        modal1_C, modal2_C, modal11_C, modal3_C = model(modal1, modal2, modal3)
        loss1_a = mutual_information(modal1_C, modal2_C).to(device)
        loss1_b = mutual_information(modal11_C, modal3_C).to(device)
        loss1 = loss1_a + loss1_b  # L1 consistent informaiton
        z_modal1, z_modal3 = model.encoder_A(modal1), model.encoder_B(modal3)
        z1, z3, prior_sample = z_modal1.rsample().cpu(), z_modal3.rsample().cpu(), prior.sample()
        z1, z3 = F.log_softmax(z1, dim=-1), F.log_softmax(z3, dim=-1)
        prior_sample = F.softmax(prior_sample, dim=-1)
        skl1 = torch.nn.functional.kl_div(z1, prior_sample).to(device)
        # skl3 = torch.nn.functional.kl_div(z3, prior_sample).to(device)
        loss2 = skl1  # L2:eliminate the superfluous information

        # loss3_a = mutual_information(modal1_C, modal3_C).to(device)
        loss3_b = mutual_information(modal1_C, modal11_C).to(device)
        loss3 = loss3_b  # L3:complement information

        loss_g = loss1 + loss3 + loss2
        loss_g.backward(retain_graph=True)
        optimiser.step()

        if epoch % 10 == 0:
            model.eval()
            x_out, _, _, _ = model(modal1, modal2, modal3)

            pre_label = np.array(x_out.cpu().detach().numpy())
            pre_label = np.argmax(pre_label, axis=1)

            # t_sne(z1, cluster_num, label, epoch)

            acc = utils1.metrics.acc(pre_label, label)
            nmi = utils1.metrics.nmi(pre_label, label)

            # loss = loss_g.cpu().detach().float()
            # arry_loss.append(loss)
            # arry_acc.append(max_ACC)
            print(
                " epoch %d loss1 %.3f loss2 %.3f loss3 %.3f acc %.3f nmi %.3f" % (epoch, loss1, loss2, loss3, acc, nmi))
