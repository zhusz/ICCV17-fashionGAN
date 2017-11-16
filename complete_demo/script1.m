clear;
fn = dir('./input/*.png');
assert(length(fn) > 0, 'There is no input images (png and jpg only).');
f = fopen('./cache/script1.txt','w');
for i = 1:length(fn)
    n = fn(i).name;
    yml_fn = ['./cache/' n(1:end-4) '_pose.yml'];
    x = parseYML(yml_fn);
    xdxd = zeros(size(x,1), 4);
    for j = 1:size(x,1)
        y = squeeze(x(j,:,:));
        idx = find(y(:,3) > 0.1);
        y = y(idx, :);
        if length(idx) > 0
            xdxd(j,1) = min(y(:,1));
            xdxd(j,2) = max(y(:,1));
            xdxd(j,3) = min(y(:,2));
            xdxd(j,4) = max(y(:,2));
        end;
    end;

    % We are now only selecting the largest person in the image.
    % You can specify your own via modifying the following block of codes.
    [~, id] = max((xdxd(:,4)-xdxd(:,3)) .* (xdxd(:,2)-xdxd(:,1)));

    xmin = xdxd(id,1);
    xmax = xdxd(id,2);
    ymin = xdxd(id,3);
    ymax = xdxd(id,4);

    win_size = 224;
    cx = 0.5;
    cy = 0.5;
    ylen = 0.75/2;
    xlen = ylen / (ymax - ymin) * (xmax - xmin);
    target_xmin = win_size * (cx - xlen);
    target_xmax = win_size * (cx + xlen);
    target_ymin = win_size * (cy - ylen);
    target_ymax = win_size * (cy + ylen);

    t = cp2tform([xmin,ymin;xmax,ymin;xmin,ymax;xmax,ymax],...
        [target_xmin,target_ymin; target_xmax, target_ymin; ...
        target_xmin, target_ymax; target_xmax, target_ymax],...
        'nonreflective similarity');

    im = imread(['./input/' fn(i).name]);
    im = imtransform(im2single(im), t, 'XData', [1 win_size], ...
        'YData', [1 win_size], 'XYScale', 1);
    imwrite(im, ['./cache/' fn(i).name]);
    fprintf(f, '%s\n', fn(i).name(1:end-4));
end;
fclose(f);

