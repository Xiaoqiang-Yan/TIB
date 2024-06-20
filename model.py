import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
import numpy as np


class Encoder(nn.Module):
    def __init__(self, in_channel):
        super(Encoder, self).__init__()
        self.in_channel = in_channel
        self.net = nn.Sequential(
            nn.Linear(self.in_channel, self.in_channel),
            nn.BatchNorm1d(self.in_channel),
            nn.ReLU(),
            nn.Linear(self.in_channel, self.in_channel),
            nn.BatchNorm1d(self.in_channel),
            nn.ReLU()
        )

        self.double_line = nn.Linear(self.in_channel, self.in_channel*2)

    def forward(self, *input):
        x = self.net(*input)
        params = self.double_line(x)
        mu, sigma = params[:, :int(self.in_channel)], params[:, int(self.in_channel):]
        sigma = softplus(sigma) + 1e-7
        return Independent(Normal(loc=mu, scale=sigma), 1)


class Model(Encoder):
    def __init__(self, out_dim, in_channel=60):
        super(Model, self).__init__(in_channel)
        self.in_channel = in_channel
        self.output = int(out_dim)
        self.cluster = nn.Sequential(
            nn.Linear(self.in_channel, self.output)
        )
        self.encoder = Encoder(in_channel)
        _initialize_weights(self)

    def forward(self, input):
        x = self.encoder.net(input)
        x = self.cluster(x)

        x_out = torch.softmax(x, dim=1)
        return x_out, self.encoder




def _initialize_weights(self):
    print("initialize")
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            assert (m.track_running_stats == self.batchnorm_track)
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def UD_constraint(model, data):
    _, classer, _ = model(data)
    CL = classer.detach().cpu().numpy()
    N, K = CL.shape
    CL = CL.T
    r = np.ones((K, 1)) / K
    c = np.ones((N, 1)) / N
    CL **= 10
    inv_K = 1. / K
    inv_N = 1. / N
    err = 1e3
    _counter = 0
    while err > 1e-2 and _counter < 75:
        r = inv_K / (CL @ c)
        c_new = inv_N / (r.T @ CL).T
        if _counter % 10 == 0:
            err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
        _counter += 1
    CL *= np.squeeze(c)
    CL = CL.T
    CL *= np.squeeze(r)
    CL = CL.T
    argmaxes = np.nanargmax(CL, 0)
    newL = torch.LongTensor(argmaxes)
    return newL

