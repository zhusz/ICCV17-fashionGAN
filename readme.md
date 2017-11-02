# ICCV17-fashionGAN
------
Shizhan Zhu

Released on Oct 11, 2017

## Updates

We will soon release a complete demo that you can use your own image and language to serve as the input. Your own original image is not limited to be 128x128 but our output is 128x128. Your input sentence is assumed not to contains words that our model does not know. You don't need to do human parsing on your own as the new demo will do it (Yes!!!). **Available next Monday.**

To facilitate future researches, we provide the indexing of our selected subset from the DeepFashion Dataset (attribute prediction task). It contains a .mat file which contains a 78979-dim indexing vector pointing to the index among the full set (the values are between 1 and 289222). We also provide the nameList of the selected subset. Download the indexing [here](https://www.dropbox.com/s/2koeocszpnusm4y/subset_index.tar.gz).

## Description

This is the implementation of Shizhan Zhu et al.'s ICCV-17 work [Be Your Own Prada: Fashion Synthesis with Structural Coherence](https://arxiv.org/abs/1710.07346). It is open source under BSD-3 license (see the `LICENSE` file). Codes can be used freely only for academic purpose. If you want to apply it to industrial products, please send an email to Shizhan Zhu at `zhshzhutah2@gmail.com` first.

## Acknoledgement

The motivation of this work, as well as the training data used, are from the [DeepFashion dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html). Please cite the following papers if you use the codes or data of this work:

```
@inproceedings{liuLQWTcvpr16DeepFashion,
 author = {Ziwei Liu and Ping Luo and Shi Qiu and Xiaogang Wang and Xiaoou Tang},
 title = {DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations},
 booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 month = June,
 year = {2016} 
 }
@inproceedings{zhu2017be,
  title={Be Your Own Prada: Fashion Synthesis with Structural Coherence},
  author={Zhu, Shizhan and Fidler, Sanja and Urtasun, Raquel and Lin, Dahua and Chen, Change Loy},
  booktitle={Proceedings of the IEEE Conference on International Conference on Computer Vision},
  year={2017}
}
```

## Qualitative Results

Matirx Visualization: The samples shown in the same row are generated from the same original person while the samples shown in the same collumn are generated from the same text description.

![](https://raw.githubusercontent.com/zhusz/ICCV17-fashionGAN/master/matrix.png)

Walking the latent space: For each row, the first and the last images are the two samples that we will make the interpolation. We gradually change the input from the left image. In the first row, we only interpolate the input to the first stage and hence the generated results only change in shapes. In the second row, we only interpolate the input to the second stage and hence the results only change in textures. The last row interpolate the input for both the first and second stages and hence the generated interpolated results transfer smoothly from the left to the right.

![](https://raw.githubusercontent.com/zhusz/ICCV17-fashionGAN/master/interp.png)

## Dependency
The implementation is based on [Torch](https://github.com/torch/torch7). [CuDNN](https://github.com/soumith/cudnn.torch) is required.

## Getting data
1. Step 1: Run the following command to obtain part of the training data and the off-the-shelf pre-trained model. Folders for models are also created here.
```shell
sh download.sh
```
This part of the data contains all the new annotations (languages and segmentation maps) on the subset of the [DeepFashion dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html), as well as the benchmarking info (the train-test split and the image-language pairs of the test set). Compared to the full data, it does not contain the ``G2.h5`` (which you need to obtain according to Step 2 below). 

2. Step 2: You can obtain ``G2.h5`` in the same way as obtaining the [DeepFashion dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html). Please refer to [this page](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/FashionSynthesis.html) for detailed instructions (e.g. sign up an agreement). After obtaining the ``G2.h5``, you need to put it into the directory of ``./data_release/supervision_signals/`` before you can use the codes.

For any questions regarding obtaining the data (e.g. cannot obtain through the Dropbox via the link) please send an email to ``zhshzhutah2@gmail.com``.

## Testing
All the testing codes are in the `demo_release` folder. The GAN of our second stage has three options in our implementation.
1. Run `demo_full.lua` with [this](https://github.com/zhusz/ICCV17-fashionGAN/blob/master/demo_release/demo_full.lua#L141) line uncommented. The network structure is our original submitted version.
2. Run `demo_full.lua` as it is. It adds the skip connection technique proposed in [Hour-glass](https://github.com/anewell/pose-hg-demo) and [pix2pix](https://github.com/phillipi/pix2pix).
3. Run `demo_p2p.lua`. The network structure completely follows [pix2pix](https://github.com/phillipi/pix2pix). The texture would be nice but cannot be controlled.

You can modify [this](https://github.com/zhusz/ICCV17-fashionGAN/blob/master/demo_release/demo_full.lua#L26) block to switch different types of visualization.

## Training
1. To train the first-stage-gan, enter the `sr1` folder and run the `train.lua` file.
2. To train the second-stage-gan, enter the relevant folder to run the `train.lua` file. Folder `ih1` refers to our original submission. Filder `ih1_skip` refers to the second-stage-network coupled with skip connection. Folder `ih1_p2p` uses pix2pix as our second stage.

## Language encoding
Please refer to the `language` folder for training and testing the initial language encoding model.

## Feedback
Suggestions and opinions of this work (both positive and negative) are greatly welcome. Please contact the author by sending email to `zhshzhutah2@gmail.com`.

## License
BSD-3, see `LICENSE` file for details.
