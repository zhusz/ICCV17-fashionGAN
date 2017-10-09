local vis = {}

-- In case of wired error, always clone and not inplaced!!!

vis.showPoint = function(toPlot, pose, mask, color, circleSize)
    assert(toPlot:dim() == 3)
    local width = toPlot:size(3)
    local hight = toPlot:size(2)
    circleSize = torch.round(circleSize)
    assert(toPlot:type() == color:type())
    assert(width >= 2 * circleSize + 1)
    assert(hight >= 2 * circleSize + 1)

    local len = 2*circleSize+1
    local logic = torch.zeros(len,len):type('torch.ByteTensor')
    for i = 1,len do
        local s = torch.abs(1+circleSize-i)
        s = torch.round(torch.sqrt(circleSize^2 - s^2))
        logic[{{i},{1+circleSize-s,1+circleSize+s}}] = 1
    end
    local logic_plain = torch.zeros(hight,width):type('torch.ByteTensor')
    pose = pose:view(-1)
    assert(pose:size(1) % 2 == 0)
    local n_pts = pose:size(1) / 2

    if mask then
        mask = mask:view(-1)
        assert(mask:size(1) == n_pts)
    end
    for i = 1,n_pts do
        if not(mask) or (mask and mask[i] == 1) then
            local x = pose[i]
            local y = pose[i+n_pts]
            x = torch.round(x);
            y = torch.round(y);
            if x <= circleSize then x = 1 + circleSize end
            if x > width - circleSize then x = width - circleSize end
            if y <= circleSize then y = 1 + circleSize end
            if y > hight - circleSize then y = hight - circleSize end
            logic_plain[{{y-circleSize,y+circleSize},{x-circleSize,x+circleSize}}] = logic
        end
    end

    color = color:view(-1)
    assert(color:size(1) == toPlot:size(1))
    local drawn = toPlot:clone()
    if toPlot:size(1) == 3 then
        error('Have not implemented')
        -- Just do it 3 times and then merge
    elseif toPlot:size(1) == 1 then
        local t = t:reshape(hight,width);
        t[logic_plain:eq(1)] = color[1]
        drawn = t:reshape(1,hight,width)
    else
        error('toPlot dimension 3 must be 1 or 3')
    end

    return drawn
end

vis.line = function(img,x1,y1,x2,y2,squareRadius,color)
    assert(img:dim() == 3)
    assert(img:type() == color:type())
    color = color:view(-1)
    assert(img:size(1) == color:size(1))
    local hight = img:size(2)
    local width = img:size(3)
    assert(squareRadius >= 0)

    local logic_map = torch.zeros(hight,width):type('torch.ByteTensor')
    local interp_x
    local itnerp_y
    local xmin = torch.min(torch.Tensor{x1,x2})
    local xmax = torch.max(torch.Tensor{x1,x2})
    local ymin = torch.min(torch.Tensor{y1,y2})
    local ymax = torch.max(torch.Tensor{y1,y2})
    local dx = torch.abs(x1-x2)
    local dy = torch.abs(y1-y2)
    if dx > dy then
        interp_x = torch.range(xmin, xmax, 1)
        interp_y = (interp_x - xmin) * dy / dx + ymin
        if (x1 - x2) * (y1 - y2) < 0 then
            local buffer = interp_y:clone()
            for j = 1,buffer:size(1) do interp_y[j] = buffer[buffer:size(1)+1-j] end
        end
    else
        interp_y = torch.range(ymin, ymax, 1)
        interp_x = (interp_y - ymin) * dx / dy + xmin
        if (x1 - x2) * (y1 - y2) < 0 then
            local buffer = interp_x:clone()
            for j = 1,buffer:size(1) do interp_x[j] = buffer[buffer:size(1)+1-j] end
        end
    end

    interp_x = interp_x:round():view(-1)
    interp_y = interp_y:round():view(-1)
    assert(interp_x:size(1) == interp_y:size(1))
    for i = 1,interp_x:size(1) do
        local x = interp_x[i]
        local y = interp_y[i]
        if x >= 1 + squareRadius and x <= width - squareRadius and y >= 1 + squareRadius and y <= hight - squareRadius then
            logic_map[{{y-squareRadius,y+squareRadius},{x-squareRadius,x+squareRadius}}] = 1
        end
    end

    local drawn = img:clone()
    if img:size(1) == 1 then
        drawn[logic_map:eq(1)] = color[1]
    elseif img:size(1) == 3 then
        for i = 1,3 do
            local t = drawn[{{i},{},{}}]
            t[logic_map:eq(1)] = color[i]
            drawn[{{i},{},{}}] = t
        end
    else
        error('img:size(1) can only be 1 or 3')
    end

    return drawn
end

return vis
