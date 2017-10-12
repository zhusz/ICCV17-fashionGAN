require 'nngraph'
require 'cunn'
require 'cudnn'
require 'hdf5'
require 'image'
require 'optim'
require 'paths'
local vis = dofile('../codes_lua/vis.lua')
local matio = require 'matio'
local test_ind = matio.load('../data_release/benchmark/ind.mat','test_ind'):view(-1)
local test_set_pair_ind = matio.load('../data_release/benchmark/ind.mat','test_set_pair_ind'):view(-1)

local dispSurrogate = dofile('../codes_lua/dispSurrogate.lua')
local disp = require 'display'
local getNet = dofile('../codes_lua/getNet.lua')
torch.setdefaulttensortype('torch.FloatTensor')

-- Randomly shuffle
local randperm = torch.randperm(test_ind:size(1))
for i = 1, test_ind:size(1) do
    test_ind[i] = test_ind[randperm[i]]
    test_set_pair_ind[i] = test_set_pair_ind[randperm[i]]
end

-- If you want to get the whole test set results on the benchmark, please comment all the lines in the following block, as well as the lines below the `Visualize' label.
-------------------------- Visualization at your choice ----------------------------
local num_to_show = 64

-- Unconment the following two lines to condition only on the same original person.
-- local selected_original_person_id = test_ind[torch.randperm(test_ind:size(1)):narrow(1,1,1)[1]]
-- for i = 1, num_to_show do test_ind[i] = selected_original_person_id; end

-- Uncomment the following two lines to condition only on the same language specification.
-- local selected_text_provider_id = test_set_pair_ind[torch.randperm(test_set_pair_ind:size(1)):narrow(1,1,1)[1]]
-- for i = 1, num_to_show do test_set_pair_ind[i] = selected_text_provider_id; end

test_ind = test_ind[{{1,num_to_show}}];  test_set_pair_ind = test_set_pair_ind[{{1,num_to_show}}]
------------------------------------------------------------------------------------

-- Loading test phase inputs
local lr = matio.load('../data_release/test_phase_inputs/sr1_8.mat','d')
lr = lr:permute(4,3,1,2)
lr = lr:contiguous()
local text = matio.load('../data_release/test_phase_inputs/encode_hn2_rnn_100_2_full.mat', 'hn2')
text = text:contiguous()
local h5file = hdf5.open('../data_release/test_phase_inputs/G2.h5','r')
local ih = h5file:read('/ih'):all() -- the original image is the required input
ih = ih:permute(1,2,4,3)
local b_ = h5file:read('/b_'):all() -- we assume a state-of-the-art segmentation network can get good human parsing results
b_ = b_:permute(1,2,4,3)
local ih_mean = h5file:read('/ih_mean'):all()
ih_mean = ih_mean:view(1,3,128,128)
ih_mean = ih_mean:permute(1,2,4,3)
h5file:close()

-- Stage 1 network construction
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
local a = torch.load('./off_the_shelf_model/sr1.t7')
G1 = a.G
a = nil

-- Pickout only the test data
local lr_tmp = lr:clone()
local text_tmp = text:clone()
lr = torch.Tensor(test_ind:size(1), testConf.n_condition, testConf.lr_win_size, testConf.lr_win_size)
text = torch.Tensor(test_ind:size(1), testConf.nt_input, 1,1)
local test_set_ih = torch.Tensor(test_ind:size(1), ih:size(2), testConf.win_size, testConf.win_size)
local test_set_b_ = torch.Tensor(test_ind:size(1), 1, testConf.win_size, testConf.win_size)
local text_provider = torch.Tensor(test_ind:size(1), ih:size(2), testConf.win_size, testConf.win_size)
for i = 1, test_ind:size(1) do
    lr[{{i},{},{},{}}] = lr_tmp[{{test_ind[i]},{},{},{}}]
    text[{{i},{},{},{}}] = text_tmp[{{test_set_pair_ind[i]},{}}]:view(1, testConf.nt_input, 1, 1)
    test_set_ih[{{i},{},{},{}}] = ih[{{test_ind[i]},{},{},{}}]
    test_set_b_[{{i},{},{},{}}] = b_[{{test_ind[i]},{},{},{}}]
    text_provider[{{i},{},{},{}}] = ih[{{test_set_pair_ind[i]},{},{},{}}]
end
lr_tmp = nil
text_tmp = nil
ih = nil
b_ = nil

-- Containers for Stage 1
local condition1 = torch.Tensor(1, testConf.n_condition, testConf.lr_win_size, testConf.lr_win_size)
condition1 = condition1:cuda()
local noise1 = torch.Tensor(1, testConf.n_z, 1, 1)
noise1 = noise1:cuda()
local encode1 = torch.Tensor(1, testConf.nt_input, 1, 1)
encode1 = encode1:cuda()
local out1 = torch.Tensor(test_set_ih:size(1), testConf.n_map_all, testConf.win_size, testConf.win_size)

-- Running Stage 1
for i = 1, test_set_ih:size(1) do
    noise1:normal(0,1)
    condition1:copy(lr[{{i},{},{},{}}])
    encode1:copy(text[{{i},{},{},{}}])
    out1[{{i},{},{},{}}] = G1:forward{noise1, encode1, condition1}:float()
    if i % 100 == 0 then print('A' .. tostring(i)) end
end
local kernel = image.gaussian(5,3):float():contiguous()
out1_p = out1:clone()
for i = 1, out1_p:size(1) do
    for j = 1, out1_p:size(2) do
        out1_p[{{i},{j},{},{}}] = image.convolve(out1[{{i},{j},{},{}}]:view(testConf.win_size, testConf.win_size), kernel, 'same'):contiguous():view(1,1,testConf.win_size, testConf.win_size)
    end
end

-- Stage 2 stuffs

local cb = {torch.Tensor{3,2,1,1,2,3,2}, torch.Tensor{2,3,3,2,1,1,2}, torch.Tensor{1,1,2,3,3,2,2}}
for i = 1,testConf.n_condition_2 do cb[i] = cb[i] * 0.25 end
local H = torch.Tensor{0.0030,0.0133,0.0219,0.0133,0.0030,0.0133,0.0596,0.0983,0.0596,0.0133,0.0219,0.0983,0.1621,0.0983,0.0219,0.0133,0.0596,0.0983,0.0596,0.0133,0.0030,0.0133,0.0219,0.0133,0.0030}:view(5,5)

local out1pmax
_, out1pmax = torch.max(out1_p:float(), 2)
out1pmax[out1pmax:eq(testConf.n_map_all)] = 0
local batch_condition = torch.Tensor(test_set_ih:size(1), testConf.n_condition_2, testConf.win_size, testConf.win_size)
for i = 1,test_set_ih:size(1) do
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

-- Comment / Uncomment the following two lines to switch between the skip version and the non-skip version.
local b = torch.load('./off_the_shelf_model/ih1_skip.t7')
-- local b = torch.load('./off_the_shelf_model/ih1.t7')
local G2 = b.G
local condition2 = torch.Tensor(1, testConf.n_condition_2, testConf.win_size, testConf.win_size)
condition2 = condition2:cuda()
local noise2 = torch.Tensor(1, testConf.n_z, 1, 1)
noise2 = noise2:cuda()
local encode2 = torch.Tensor(1, testConf.nt_input, 1, 1)
encode2 = encode2:cuda()
local out2 = torch.Tensor(test_set_ih:size(1), testConf.nc, testConf.win_size, testConf.win_size)
local out2_final = out2:clone()
for i = 1, test_set_ih:size(1) do
    condition2:copy(batch_condition[{{i},{},{},{}}])
    noise2:normal(0,1)
    encode2:copy(text[{{i},{},{},{}}])
    out2[{{i},{},{},{}}] = G2:forward{noise2, encode2, condition2}:float() + ih_mean
    test_set_ih[{{i},{},{},{}}] = test_set_ih[{{i},{},{},{}}] + ih_mean
    text_provider[{{i},{},{},{}}] = text_provider[{{i},{},{},{}}] + ih_mean
    -- Post processing for hair + face + and balance the skin color using the median of the original region.
    for j = 1, testConf.nc do
        local t = out2[{{i},{j},{},{}}]
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
    if i % 100 == 0 then print('B' .. tostring(i)) end
end

-- Visualize
engJ = matio.load('../data_release/benchmark/language_original.mat','engJ')
print(string.format('Text input for the %d samples:', num_to_show))
local ss = {}
for i = 1, num_to_show do
    local s = tostring(i) .. ': '
    for j = 1, engJ[test_set_pair_ind[i]][1]:size(1) do
        s = s .. string.format('%c', engJ[test_set_pair_ind[i]][1][j])
    end
    print(s)
    ss[i] = s
end
dispSurrogate(test_set_ih[{{1,num_to_show},{},{},{}}], 159, 'original')
dispSurrogate(out1_p[{{1,num_to_show},{},{},{}}], 163, 'stage_1_output', 'm2c')
dispSurrogate(out2_final[{{1,num_to_show},{},{},{}}], 167, 'out2_post_p2pzd')

