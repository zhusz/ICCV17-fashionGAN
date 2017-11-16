require 'nngraph'
require 'cunn'
require 'cudnn'
require 'image'
local matio = require 'matio'

local testConf = {}
testConf.n_map_all = 7
testConf.n_z = 80
testConf.nt_input = 100
testConf.n_condition = 4
testConf.disp_win_id = 50
testConf.win_size = 128
testConf.lr_win_size = 8
testConf.nc = 3
testConf.n_condition_2 = 3

local lr = matio.load('./cache/script2.mat','lr')
lr = lr:permute(4,3,1,2)
lr = lr:contiguous()
local m = lr:size(1)
local text = matio.load('./cache/test_lang_initial.mat', 'hn2')
text = text:contiguous()
text = text:view(m, testConf.nt_input, 1, 1)
local ih_mean_temp = matio.load('./ih_mean.mat', 'ih_mean')
local ih_mean = ih_mean_temp:permute(3,1,2):contiguous():view(1,testConf.nc,testConf.win_size,testConf.win_size)
local test_set_b_ = matio.load('./cache/script2.mat', 'b_')
test_set_b_ = test_set_b_:permute(4,3,1,2)
test_set_b_:contiguous()

local a = torch.load('./sr1.t7')
local G1 = a.G
a = nil
local b = torch.load('./ih1_skip.t7')
local G2 = b.G
b = nil

lr = lr:cuda()
text = text:cuda()
local z = torch.Tensor(m, testConf.n_z, 1,1)
z = z:cuda()
z:normal(0,1)
local out1 = G1:forward{z, text, lr}:float()
local kernel = image.gaussian(5, 3):float():contiguous()
for i = 1,m do
    for j = 1, testConf.n_map_all do
        out1[{{i},{j},{},{}}] = image.convolve(out1[{{i},{j},{},{}}]:view(testConf.win_size, testConf.win_size), kernel, 'same'):contiguous():view(1,1,testConf.win_size, testConf.win_size)
    end
end
_, out1pmax = torch.max(out1, 2)
out1pmax[out1pmax:eq(testConf.n_map_all)] = 0

local cb = {torch.Tensor{3,2,1,1,2,3,2}, torch.Tensor{2,3,3,2,1,1,2}, torch.Tensor{1,1,2,3,3,2,2}}
for i = 1,testConf.n_condition_2 do cb[i] = cb[i] * 0.25 end
local H = torch.Tensor{0.0030,0.0133,0.0219,0.0133,0.0030,0.0133,0.0596,0.0983,0.0596,0.0133,0.0219,0.0983,0.1621,0.0983,0.0219,0.0133,0.0596,0.0983,0.0596,0.0133,0.0030,0.0133,0.0219,0.0133,0.0030}:view(5,5):float()

local batch_condition = torch.Tensor(m, testConf.n_condition_2, testConf.win_size, testConf.win_size)
for i = 1,m do
    local t = out1pmax[{{i},{1},{},{}}]
    for j = 1,testConf.n_condition_2 do
        local u = torch.Tensor(1,1,testConf.win_size, testConf.win_size)
        for k = 1,testConf.n_map_all do
            u[t:eq(k%testConf.n_map_all)] = cb[j][k]
        end
        local v = image.convolve(u:squeeze():float(), H:float(), 'same'):contiguous()
        batch_condition[{{i},{j},{},{}}] = v:view(1,1,testConf.win_size, testConf.win_size)
    end
end
batch_condition = batch_condition - 0.5

z:normal(0,1)
batch_condition = batch_condition:cuda()
local out2 = G2:forward{z, text, batch_condition}:float()

local namesFile = io.open('./cache/script1.txt')
local test_set_ih = torch.zeros(m, testConf.nc, testConf.win_size, testConf.win_size):float()
local nameList = {}
local idx = 0
for line in namesFile:lines() do
    idx = idx + 1
    test_set_ih[{{idx},{},{},{}}] = image.scale(image.load('./cache/' .. line .. '.png', 3, 'float'), testConf.win_size, testConf.win_size)
    nameList[idx] = line
end
namesFile:close()
local out2_final = out2:clone()
for i = 1,m do
    for j = 1, testConf.nc do
        local t = out2[{{i},{j},{},{}}] + ih_mean[{{},{j},{},{}}]
        local s = test_set_ih[{{i},{j},{},{}}]
        local ori_b_ = test_set_b_[{{i},{},{},{}}]
        local now_b_ = out1pmax[{{i},{},{},{}}]
        t[ori_b_:eq(1)] = s[ori_b_:eq(1)]
        t[ori_b_:eq(2)] = s[ori_b_:eq(2)]
        if ori_b_:eq(5):sum() > 0 and now_b_:eq(5):sum() > 0 then
            local sc5 = s[ori_b_:eq(5)]:median()
            t[now_b_:eq(5)] = (t[now_b_:eq(5)] + sc5[1]) / 2
        end
        if ori_b_:eq(6):sum() > 0 and now_b_:eq(6):sum() > 0 then
            local sc6 = s[ori_b_:eq(6)]:median()
            t[now_b_:eq(6)] = (t[now_b_:eq(6)] + sc6[1]) / 2
        end
        out2_final[{{i},{j},{},{}}] = t
    end
end
for i = 1, m do
    image.save('./output/' .. nameList[i] .. '.png', out2_final[{{i},{},{},{}}]:view(testConf.nc, testConf.win_size, testConf.win_size))
end

