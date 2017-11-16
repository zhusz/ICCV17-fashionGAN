require 'cunn'
require 'cudnn'
require 'image'
local matio = require 'matio'

local namesFile = io.open('./cache/script1.txt')
local idx = 1
local nameList = {}
for line in namesFile:lines() do
    nameList[idx] = line
    idx = idx + 1
end
namesFile:close()
local m = idx - 1
local img_root = './cache/'
local model = torch.load(os.getenv('SEG_MODEL'))
local win_size = 224
local input = torch.zeros(m, 3, win_size, win_size):float()

for i = 1, m do
    local im = image.load(img_root .. nameList[i] .. '.png', 3, 'float')
    input[{{i},{},{},{}}] = im
end

input = input:cuda()

local output = model:forward(input)

local prob = cudnn.SpatialSoftMax():cuda():forward(output):float()

matio.save('./cache/test_seg.mat', {prob=prob})

