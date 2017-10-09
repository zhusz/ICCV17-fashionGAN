-- Assume that nngraph / cunn / cudnn have been required
-- So does hdf5

local getNet = {}

-- getNet.getNet_weightedLayers = function(net)
--     local a = {}
--     local count = 0
--     for indexNode, node in ipairs(net.forwardnodes) do
--         if node.data.module then
--             local t = node.data.module
--             for k,v in pairs(t) do
--                 if k == 'weight' then
--                     count = count + 1
--                     a[indexNode] = {}
--                     a[indexNode].count = count
--                     a[indexNode].name = tostring(t)
--                     a[indexNode].weight = v:clone()
--                     a[indexNode].bias = t.bias:clone()
--                     a[indexNode].output = t.output:clone()
--                     a[indexNode].gradWeight = t.gradWeight:clone()
--                 end
--             end
--         end
--     end
--     return a
-- end

getNet.print_layer_content = function(net, layer_id)
    local i = 0
    for indexNode, node in ipairs(net.forwardnodes) do
        i = i + 1
        if i == layer_id then
            if node.data.module then
                local t = node.data.module
                for k,v in pairs(t) do
                    if k == 'weight' or k == 'output' or k == 'gradWeight' or k == 'bias' then
                        print(k)
                    else
                        print(k .. ": " .. tostring(v))
                    end
                end
            else
                print("This is an empty later. Please double check the index of the layer!")
            end
            return
        end
    end
end

getNet.getNet_full = function(net)
    local a = {}
    local count = 0
    local len_a = 0
    for indexNode, node in ipairs(net.forwardnodes) do
        if node.data.module then
            local t = node.data.module
            count = count + 1
            a[indexNode] = {}
            a[indexNode].count = count
            a[indexNode].name = tostring(t)
            for k,v in pairs(t) do
                if k == 'weight' then
                    a[indexNode].weight = v:clone()
                end
                if k == 'output' then
                    a[indexNode].output = v:clone()
                end
                if k == 'gradWeight' then
                    a[indexNode].gradWeight = v:clone()
                end
                if k == 'bias' then
                    a[indexNode].bias = v:clone()
                end
            end
            if a[indexNode].output:nDimension() == 0 then
                a[indexNode].output = NULL
            end
            a[indexNode].name = a[indexNode].name or "00"
            a[indexNode].weight = a[indexNode].weight or torch.Tensor{0,0}
            a[indexNode].output = a[indexNode].output or torch.Tensor{0,0}
            a[indexNode].gradWeight = a[indexNode].gradWeight or torch.Tensor{0,0}
            a[indexNode].bias = a[indexNode].bias or torch.Tensor{0,0}
        end
        len_a = indexNode
    end
    return a, len_a
end

getNet.serializeNet_foldermat = function(net, folder_name)
    local matio = require 'matio'
    local a, len_a = getNet.getNet_full(net) -- upvalue
    folder_name = folder_name or "Net";
    os.execute("rm -r " .. folder_name)
    os.execute("mkdir " .. folder_name)
    for i = 1,len_a do
        if a[i] then
            print(i)
            print(a[i])
            matio.save('./' .. folder_name .. '/name_' .. tostring(i) .. '.mat', {name = a[i].name})
            matio.save('./' .. folder_name .. '/weight_' .. tostring(i) .. '.mat', {weight = a[i].weight:type('torch.FloatTensor')})
            matio.save('./' .. folder_name .. '/output_' .. tostring(i) .. '.mat', {output = a[i].output:type('torch.FloatTensor')})
            matio.save('./' .. folder_name .. '/gradWeight_' .. tostring(i) .. '.mat', {gradWeight = a[i].gradWeight:type('torch.FloatTensor')})
            matio.save('./' .. folder_name .. '/bias_' .. tostring(i) .. '.mat', {bias = a[i].bias:type('torch.FloatTensor')})
        end
    end
    matio.save('./' .. folder_name .. '/info_root.mat', {count = len_a})
end

return getNet
