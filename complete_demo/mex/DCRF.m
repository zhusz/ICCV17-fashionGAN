function new_map = DCRF(image, prob, M, params)

assert(length(params) == 10);
assert(size(image,1) == size(prob,1));
assert(size(image,2) == size(prob,2));
if size(image,3) == 1, image = cat(3,image,image,image);end;
assert(size(image,3) == 3);
assert(size(prob,3) == M);
image = im2uint8(image);
prob = double(prob);
prob(prob < 1e-8) = 1e-8;

% ------------ Uncomment this part of code to enable background confusion setting -------------- %
% for j = 1:size(prob,4)
%     [~,tmp] = max(prob(:,:,:,j),[],3);
%     for k = 1:size(prob,3)
%         t = prob(:,:,k,j);
%         t(tmp) = 1 / M;
%         prob(:,:,k,j) = t;
%     end;
% end;
% ---------------------------------------------------------------------------------------------- % 

new_map = DCRF_inner(permute(image,[3 1 2]), permute(prob,[3 1 2]), M, size(image,1), size(image,2), params);
new_map = reshape(new_map, size(image,1), size(image,2));
% new_map = permute(new_map, [2 3 1]);
new_map = uint8(new_map);

end

