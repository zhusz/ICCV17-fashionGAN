local h5rw = {}

require 'hdf5'
require 'paths'
-- require the global variable G_codes_lua to localize the other toolbox lua files
local checkVar = dofile(G_codes_lua .. 'checkVar.lua')

h5rw.write = function(a, file_name)
    local h5file = hdf5.open(file_name, 'w')
    local k = checkVar.getKeys(a)
    for i = 1,#k do
        h5file:write('/' .. k[i], a[k[i]]:type('torch.FloatTensor'))
    end
    h5file:close()
end

h5rw.read = function(file_name, k)
    local h5file = hdf5.open(file_name, 'r')
    local a = {}
    for i = 1,#k do
        a[k[i]] = h5file:read(k[i]):all()
    end
    return a
end

return h5rw
