require 'nngraph'
require 'cunn'
require 'cudnn'
require 'hdf5'
require 'image'
require 'optim'
require 'paths'
local matio = require 'matio'
train_ind = matio.load('../data_release/benchmark/ind.mat','train_ind'):view(-1)

local dispSurrogate = dofile('../codes_lua/dispSurrogate.lua')
local disp = require 'display'
local getNet = dofile('../codes_lua/getNet.lua')
torch.setdefaulttensortype('torch.FloatTensor')

local theme = 'ih1'
assert(theme == 'ih1')

local config = dofile('./config_ih1.lua')
local lambda = 100 -- zhuzhu
local G,D = dofile('./net_graph_ih1.lua')

local h5file = hdf5.open('../data_release/supervision_signals/G2.h5', 'r')
local ih = h5file:read('/ih'):all()
ih = ih:permute(1,2,4,3)
local ih_mean = h5file:read('/ih_mean'):all()
ih_mean = ih_mean:view(1,3,128,128)
ih_mean = ih_mean:permute(1,2,4,3)
local b_ = h5file:read('/b_'):all()
b_ = b_:permute(1,2,4,3)
local n_file = ih:size(1)

config.lr = 0.0002
config.beta1 = 0.5

local criterion = nn.BCECriterion()
local criterionAE = nn.AbsCriterion()
local optimStateG = {
    learningRate = config.lr,
    beta1 = config.beta1,
}
local optimStateD = {
    learningRate = config.lr,
    beta1 = config.beta1,
}

local nz = config.nz
local input = torch.Tensor(config.batchSize, config.n_c, config.win_size, config.win_size)
local condition = torch.Tensor(config.batchSize, config.n_condition, config.win_size, config.win_size) -- zhuzhu

print(nz)
local label = torch.Tensor(config.batchSize, 1, 14, 14)
local errD, errG, errL1
cutorch.setDevice(1)
input = input:cuda();  label = label:cuda();  condition = condition:cuda()
local input_record, condition_record

if pcall(require, 'cudnn') then
    require 'cudnn'
    cudnn.benchmark = true
    cudnn.convert(G, cudnn)
    cudnn.convert(D, cudnn)
end
D:cuda();           G:cuda();           criterion:cuda();      criterionAE:cuda()

local parametersD, gradParametersD = D:getParameters()
local parametersG, gradParametersG = G:getParameters()

----------- zhuzhu ------------
local cb = {torch.Tensor{3,2,1,1,2,3,2}, torch.Tensor{2,3,3,2,1,1,2}, torch.Tensor{1,1,2,3,3,2,2}}
for i = 1,config.n_condition do cb[i] = cb[i] * 0.25 end
local H = torch.Tensor{0.0030,0.0133,0.0219,0.0133,0.0030,0.0133,0.0596,0.0983,0.0596,0.0133,0.0219,0.0983,0.1621,0.0983,0.0219,0.0133,0.0596,0.0983,0.0596,0.0133,0.0030,0.0133,0.0219,0.0133,0.0030}:view(5,5)
-------------------------------

local train_size = train_ind:size(1)
local simple_sample = function()
    local ind = torch.randperm(train_size):narrow(1,1,config.batchSize)
    for i = 1,config.batchSize do ind[i] = train_ind[ind[i]] end

    for i = 1,config.batchSize do
        input[{{i},{},{},{}}] = ih[{{ind[i]},{},{},{}}]

        local t = b_[{{ind[i]},{1},{},{}}]
        for j = 1,config.n_condition do
            local u = torch.Tensor(1,1,config.win_size,config.win_size):zero()
            for k = 1,config.n_map_all do
                u[t:eq(k%config.n_map_all)] = cb[j][k]
            end
            -- do blurring toward u
            local v = image.convolve(u:squeeze():float(), H:float(), 'same'):contiguous()
            condition[{{i},{j},{},{}}] = v:view(1,1,config.win_size,config.win_size)
        end
    end
    condition = condition - 0.5; -- zero mean

    do return end
end

local real_label = 1
local fake_label = 0
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()

local errD_real, errD_fake
local errG_bce, errG_fm
local fDx = function(x)
    D:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    G:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end) -- zhuzhu
   gradParametersD:zero()

   -- train with real
   data_tm:reset(); data_tm:resume()
   simple_sample()
   local fake = G:forward(condition) -- zhuzhuf2A
   input_record = input:clone()
   condition_record = condition:clone()
   data_tm:stop()
   label:fill(real_label)

   local output = D:forward{input, condition} --
   errD_real = criterion:forward(output, label)
   local de_do = criterion:backward(output, label)
   D:backward({input, condition}, de_do) --

   input:copy(fake)
   label:fill(fake_label)

   output = D:forward{input, condition} --
   errD_fake = criterion:forward(output, label)
   de_do = criterion:backward(output, label)
   D:backward({input, condition}, de_do) --

   errD = (errD_real + errD_fake) / 2 -- zhuzhu

   return errD, gradParametersD
end

local fGx = function(x)
    D:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    G:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end) -- zhuzhu
   gradParametersG:zero()

   local output = D.output -- zhuzhuf2A
   label:fill(real_label)
   errG = criterion:forward(output, label)
   local de_do = criterion:backward(output, label)
   local de_dg = D:updateGradInput({input, condition}, de_do) --
   errL1 = criterionAE:forward(input, input_record:cuda())
   local df_do_AE = criterionAE:backward(input, input_record:cuda())
   G:backward({condition}, de_dg[1] + df_do_AE:mul(lambda))

   return errG, gradParametersG
end

local condition_vis = torch.Tensor(config.batchSize, config.n_condition, config.win_size, config.win_size)
condition_vis = condition_vis:cuda()

for iter = 1, 1e9 do
    optim.adam(fDx, parametersD, optimStateD)
    optim.adam(fGx, parametersG, optimStateG)
    if iter % 20 == 0 then
        print(('Iter %d: ErrD %.5f (ErrD_real %.5f, ErrD_fake %.5f), ErrG %.5f.'):format(iter,errD,errD_real,errD_fake,errG))
    end

    if iter == 1 then
        for i = 1, config.batchSize do condition_vis[{{i},{},{},{}}] = condition[{{1},{},{},{}}] end
    end
    local base = 400
    local fake = G:forward(condition)

    if iter % 20 == 0 then
        dispSurrogate(fake:type('torch.FloatTensor') + ih_mean:repeatTensor(config.batchSize, 1,1,1), 3+base, 'fake')
        dispSurrogate(input_record:type('torch.FloatTensor') + ih_mean:repeatTensor(config.batchSize, 1,1,1), 4+base, 'real')
        dispSurrogate(condition_record:type('torch.FloatTensor'), 5+base, 'condition')
    end

    if iter == 1 or iter % 10000 == 0 then
        local net = {}
        net.G = G
        net.D = D
        torch.save('./' .. theme .. '/' .. theme .. '_'.. tostring(iter) .. '.t7', net)
    end
end
