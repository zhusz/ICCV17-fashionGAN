function x = parseYML(yml_file_name)

fid = fopen(yml_file_name, 'r');
fgetl(fid);
fgetl(fid);
s = fscanf(fid, '   sizes: [ %f, %f, %f ]');
s = s(:)';
assert(length(s) == 3);
fgetl(fid);
fscanf(fid, '   data: [ ');
data = fscanf(fid, '%f, ', prod(s));
x = reshape(data, s([3 2 1]));
x = permute(x, [3 2 1]);

end

