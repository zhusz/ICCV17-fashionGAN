import numpy as np
import sys
from random import randint
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.io import loadmat
from scipy.io import savemat
mat = loadmat('./cache/script2.mat')
codeJ = mat['codeJ']

dim_voc = 539
bsz = 1
dim_h = 100
dim_cate_new = 19
dim_color = 17
dim_gender = 2
dim_sleeve = 4
num_layers = 2
class define_network(nn.Module):
    def __init__(self):
        super(define_network, self).__init__()
        self.rnn = nn.RNN(dim_voc, dim_h, num_layers)
        self.net_cate_new = nn.Linear(dim_h, dim_cate_new)
        self.net_color = nn.Linear(dim_h, dim_color)
        self.net_gender = nn.Linear(dim_h, dim_gender)
        self.net_sleeve = nn.Linear(dim_h, dim_sleeve)

    def forward(self, x):
        h0 = Variable(torch.zeros(num_layers, bsz, dim_h).cuda())
        _, hn = self.rnn(x, h0)
        hn2 = hn[-1]
        y_cate_new = self.net_cate_new(hn2)
        y_color = self.net_color(hn2)
        y_gender = self.net_gender(hn2)
        y_sleeve = self.net_sleeve(hn2)
        return hn2, y_cate_new, y_color, y_gender, y_sleeve

model = define_network()
model.cuda()
model.load_state_dict(torch.load('rnn_latest.pth'))
model.eval()

test_hn2 = np.zeros((len(codeJ), dim_h))
for sample_id in range(len(codeJ)):
    c = codeJ[sample_id][0]
    l = len(c)
    cuda_c_onehot = torch.zeros(l, bsz, dim_voc).cuda()
    for i in range(l):
        cuda_c_onehot[i][0][int(c[i][0]-1)] = 1
    cuda_c_onehot_v = Variable(cuda_c_onehot)
    hn2, _, _, _, _ = model(cuda_c_onehot_v)
    test_hn2[sample_id] = hn2.data[0].cpu().numpy()

result = {"hn2": test_hn2}
savemat("./cache/test_lang_initial.mat", result)

