local disp = require 'display'

local channelMap2compactMap = function(channelMap, flag, n)
    assert(channelMap:dim() == 4)
    assert(channelMap:size(1) == 1)
    n = n or channelMap:size(2)
    assert(n >= channelMap:size(2))

    local compactMap = torch.zeros(1,1,channelMap:size(3),channelMap:size(4))
    for k = 1,channelMap:size(2) do
        local t = channelMap[{{},{k},{},{}}]
        if flag == 0 then
            compactMap[t:gt(0)] = k % n
        else
            compactMap = torch.cmax(compactMap, t * flag)
        end
    end
    return compactMap
end

local batch_channelMap2compactMap = function(channelMap, flag)
    assert(channelMap:dim() == 4)
    local compactMaps = torch.zeros(channelMap:size(1),1,channelMap:size(3),channelMap:size(4))
    for i = 1,channelMap:size(1) do
        compactMaps[{{i},{},{},{}}] = channelMap2compactMap(channelMap[{{i},{},{},{}}], flag)
    end
    return compactMaps
end

local maxingChannelMap2compactMap = function(channelMap, n) -- assuems that flag is always true
    assert(channelMap:dim() == 4)
    assert(channelMap:size(1) == 1)
    n = n or channelMap:size(2)
    assert(n >= channelMap:size(2))

    local compactMap = torch.zeros(1,1,channelMap:size(3),channelMap:size(4))
    local max_indices
    _, max_indices = torch.max(channelMap,2)
    max_indices = max_indices % n
    return max_indices
end

local batch_maxingChannelMap2compactMap = function(channelMap, n)
    assert(channelMap:dim() == 4)
    local compactMaps = torch.zeros(channelMap:size(1),1,channelMap:size(3),channelMap:size(4))
    for i = 1,channelMap:size(1) do
        compactMaps[{{i},{},{},{}}] = maxingChannelMap2compactMap(channelMap[{{i},{},{},{}}], flag)
    end
    return compactMaps
end

local dispSurrogate = function(t, wid, txt, method)
    assert(t:dim() == 4)
    txt = txt or ''
    if not(method) then
        disp.image(t, {win=wid, title = tostring(wid) .. ': ' .. txt})
    elseif method == 'c2cl' then --channel2compactLabels
        disp.image(batch_channelMap2compactMap(t, 0), {win=wid, title = tostring(wid) .. ': ' .. txt})
    elseif method == 'c2cc' then --channel2compactCascades
        disp.image(batch_channelMap2compactMap(t, 1), {win=wid, title = tostring(wid) .. ': ' .. txt})
    elseif method == 'm2c' then -- maxingChannel2compact
        disp.image(batch_maxingChannelMap2compactMap(t),{win=wid,title = tostring(wid) .. ': ' .. txt})
    else
        error('Not implemented')
    end
end

return dispSurrogate
