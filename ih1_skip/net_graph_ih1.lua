require 'nngraph'
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02) -- zhuzhu
      m.bias:fill(0) -- zhuzhu
      -- m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local nc = 3
local ncondition = 3 -- zhuzhu
local nz = 80 -- opt.nz
local nt_input = 100
local nt = 20
local ndf = 64 -- opt.ndf
local ngf = 64 -- opt.ngf
local inplace = true

local bn4 = nn.SpatialBatchNormalization
local conv = nn.SpatialConvolution
local deconv = nn.SpatialFullConvolution
local relu = nn.ReLU
local lerelu = nn.LeakyReLU

local input_condition = nn.Identity()()
local e1 = input_condition - conv(ncondition, ngf, 4, 4, 2, 2, 1, 1)
local e2 = e1 - lerelu(0.2, inplace) - conv(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - bn4(ngf * 2)
local e3 = e2 - lerelu(0.2, inplace) - conv(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - bn4(ngf * 4)
local e4 = e3 - lerelu(0.2, inplace) - conv(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - bn4(ngf * 8)
local e5 = e4 - lerelu(0.2, inplace) - conv(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - bn4(ngf * 8)
local e6 = e5 - lerelu(0.2, inplace) - conv(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - bn4(ngf * 8)
local e7 = e6 - lerelu(0.2, inplace) - conv(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1)

local input_encode = nn.Identity()()
local h1 = input_encode - conv(nt_input, nt, 1, 1) - lerelu(0.2, inplace)

local input_z = nn.Identity()()
local d1_ = {input_z,h1,e7} - nn.JoinTable(2) - relu(inplace) - deconv(ngf*8+nz+nt, ngf*8, 4, 4, 2, 2, 1, 1) - bn4(ngf * 8) - nn.Dropout(0.5)
local d1 = {d1_, e6} - nn.JoinTable(2)
local d2_ = d1 - relu(inplace) - deconv(ngf*8*2, ngf*8, 4, 4, 2, 2, 1, 1) - bn4(ngf*8) - nn.Dropout(0.5)
local d2 = {d2_, e5} - nn.JoinTable(2)
local d3_ = d2 - relu(inplace) - deconv(ngf*8*2, ngf*8, 4, 4, 2, 2, 1, 1) - bn4(ngf*8) - nn.Dropout(0.5)
local d3 = {d3_, e4} - nn.JoinTable(2)
local d4_ = d3 - relu(inplace) - deconv(ngf*8*2, ngf*4, 4, 4, 2, 2, 1, 1) - bn4(ngf*4)
local d4 = {d4_, e3} - nn.JoinTable(2)
local d5_ = d4 - relu(inplace) - deconv(ngf*4*2, ngf*2, 4, 4, 2, 2, 1, 1) - bn4(ngf*2)
local d5 = {d5_, e2} - nn.JoinTable(2)
local d6_ = d5 - relu(inplace) - deconv(ngf*2*2, ngf, 4, 4, 2, 2, 1, 1) - bn4(ngf)
local d6 = {d6_, e1} - nn.JoinTable(2)
local d7 = d6 - relu(inplace) - deconv(ngf * 2, nc, 4, 4, 2, 2, 1, 1)
local o1 = d7 - nn.Tanh()
local netG = nn.gModule({input_z, input_encode, input_condition}, {o1})
netG:apply(weights_init)

local output_data = nn.Identity()()
local output_condition = nn.Identity()()
local output_merge = {output_data, output_condition} - nn.JoinTable(2)
local x0 = output_merge - conv(ncondition+nc, ndf, 4, 4, 2, 2, 1, 1) - lerelu(0.2, inplace)
local x1 = x0 - conv(ndf*1,ndf*2,4,4,2,2,1,1) - bn4(ndf*2) - lerelu(0.2, inplace)
local x2 = x1 - conv(ndf*2,ndf*4,4,4,2,2,1,1) - bn4(ndf*4) - lerelu(0.2, inplace)
local output_encode = nn.Identity()()
local b1 = output_encode - conv(nt_input, nt, 1, 1) - bn4(nt) - lerelu(0.2, inplace) - nn.Replicate(16,3) - nn.Replicate(16,4)
local x3 = {x2, b1} - nn.JoinTable(2) - conv(ndf*4+nt,ndf*8,4,4,1,1,1,1) - bn4(ndf*8) - lerelu(0.2, inplace)
local x4 = x3 - conv(ndf*8,1,4,4,1,1,1,1) - nn.Sigmoid()
local netD = nn.gModule({output_data, output_encode, output_condition}, {x4})
netD:apply(weights_init)

return netG, netD
