clear;
addpath('./mex');
load map map;
f = fopen('./cache/script1.txt','r');
load ./cache/test_seg.mat prob;
prob = permute(prob, [3,4,2,1]);
win_size = 224;
L = 18;
i = 0;
label_assign = [1,1,2,3,3,   3,3,3,3,3,   2,3,3,3,3,   3,3,4];
lrc = cell(10000,1);
codeJ = cell(10000,1);
seg_final = cell(10000,1);
while ~feof(f)
    fn = fgetl(f);
    i = i + 1;
    img = imread(['./cache/' fn '.png']);
    bb = DCRF(img, prob(:,:,:,i), L, [3 3 5 30 30 10 10 10 9 5]);
    bb = bb + 1;
    seg_final{i} = bb;

    % bb is the final segmentation result. You can use this as your human
    % parsing result :-)
    b_temp = bb;
    for j = 1:L
        b_temp(bb == j) = label_assign(j);
    end;
    lrc{i} = zeros([8,8,4]);
    for j = 1:4
        lrc{i}(:,:,j) = imresize(single(b_temp == j), [8,8]);
    end;

    % Lanugae preprocess as follows
    fid = fopen(['./input/' fn '.txt'], 'r');
    s = fgetl(fid);
    fclose(fid);
    s = strtrim(s);
    while(s(end) == '.'), s = s(1:end-1); end;
    ss = strsplit(s, ' ');
    for j = 1:length(ss)
        codeJ{i} = [codeJ{i} double(map(ss{j}))]; % The error from this line means that you have contained a word that our initial language embedding model does not know.
    end;
    codeJ{i} = codeJ{i}(:);
end;
fclose(f);
lrc = lrc(1:i);
codeJ = codeJ(1:i);
lr = zeros(8,8,4,i);
for j = 1:i
    lr(:,:,:,j) = lrc{j};
end;
% lr = permute(lr, [4,3,1,2]);
seg_final = seg_final(1:i);
fine_size = 128;
b_ = zeros(fine_size, fine_size, 1, i);
label_assign = [1,1,2,3,4, 4,4,4,5,5, 2,5,5,6,6, 4,3,0];
for j = 1:i
    t = seg_final{j};
    v = t;
    for k = 1:L, v(t == k) = label_assign(k); end;
    b_(:,:,:,j) = imresize(v, [fine_size, fine_size], 'nearest');
end;
save('./cache/script2.mat', 'lr', 'codeJ', 'b_', 'seg_final');

