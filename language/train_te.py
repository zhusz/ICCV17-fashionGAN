import sys
from random import randint
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from scipy.io import loadmat
mat = loadmat('../data_release/benchmark/language_original.mat')
for k, v in mat.iteritems():
    exec(k +  " = mat['" + k + "']")

dim_voc = 539
bsz = 1
m = 78979
m_train = 70000
dim_h = 100
dim_cate_new = 19
dim_color = 17
dim_gender = 2
dim_sleeve = 4
num_layers = 2

mat = loadmat('../data_release/benchmark/ind.mat')
train_ind = torch.IntTensor(m_train)
for i in range(m_train):
    train_ind[i] = int(mat['train_ind'][0][i] - 1)

data_cate_new = torch.IntTensor(m, 1)
data_color = torch.IntTensor(m, 1)
data_gender = torch.IntTensor(m, 1)
data_sleeve = torch.IntTensor(m, 1)
for i in range(m):
    data_cate_new[i][0] = int(cate_new[i][0] - 1)
    data_color[i][0] = int(color_[i][0] - 1)
    data_gender[i][0] = int(gender_[i][0])
    data_sleeve[i][0] = int(sleeve_[i][0] - 1)

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
criterion = nn.CrossEntropyLoss().cuda()
cuda_label_cate_new = Variable(torch.LongTensor(bsz).zero_().cuda())
cuda_label_color = Variable(torch.LongTensor(bsz).zero_().cuda())
cuda_label_gender = Variable(torch.LongTensor(bsz).zero_().cuda())
cuda_label_sleeve = Variable(torch.LongTensor(bsz).zero_().cuda())

optimizer = optim.SGD(model.parameters(), lr=0.01)
model.train()
for iter in range(1000000 * 10):

    if iter == 50000:
        optimizer = optim.SGD(model.parameters(), lr=0.001)
    assert bsz == 1
    t = randint(0, m_train-1)
    sample_id = train_ind[t]
    c = codeJ[sample_id][0]
    l = len(c)
    cuda_c_onehot = torch.zeros(l, bsz, dim_voc).cuda()
    for i in range(l):
        cuda_c_onehot[i][0][int(c[i][0]-1)] = 1
    cuda_c_onehot = Variable(cuda_c_onehot)

    cuda_label_cate_new.data[0] = data_cate_new[sample_id][0]
    cuda_label_color.data[0] = data_color[sample_id][0]
    cuda_label_gender.data[0] = data_gender[sample_id][0]
    cuda_label_sleeve.data[0] = data_sleeve[sample_id][0]

    optimizer.zero_grad()
    hn2, y_cate_new, y_color, y_gender, y_sleeve = model(cuda_c_onehot)
    loss_cate_new = criterion(y_cate_new, cuda_label_cate_new)
    loss_color = criterion(y_color, cuda_label_color)
    loss_gender = criterion(y_gender, cuda_label_gender)
    loss_sleeve = criterion(y_sleeve, cuda_label_sleeve)
    loss = loss_cate_new + loss_color + loss_gender + loss_sleeve
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print('Training Iter %d: Loss = %.5f, cate_new (%.5f), color (%.5f), gender(%.5f), sleeve(%.5f)' % (iter, loss.data[0], loss_cate_new.data[0], loss_color.data[0], loss_gender.data[0], loss_sleeve.data[0]))

    if iter % 100000 == 1:
        torch.save(model.state_dict(), 'rnn_latest.pth')

