wget https://www.dropbox.com/s/9wiyiqrc6mm51eq/human_parsing_model.tar.gz
tar -xzf human_parsing_model.tar.gz
cd mex
matlab -nodesktop -nojvm -r "my_compile; exit"
cd ..


