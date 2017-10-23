# ICCV17-fashionGAN
------
Shizhan Zhu

Released on Oct 11, 2017

## Description

This is the implementation of Shizhan Zhu et al.'s ICCV-17 work [Be Your Own Prada: Fashion Synthesis with Structural Coherence](https://arxiv.org/abs/1710.07346). It is open source under BSD-3 license (see the `LICENSE` file). Codes can be used freely only for academic purpose. If you want to apply it to industrial products, please send an email to Shizhan Zhu at `zhshzhutah2@gmail.com` first.

## Qualitative Results

Matirx Visualization: The samples shown in the same row are generated from the same original person while the samples shown in the same collumn are generated from the same text description.

![](https://raw.githubusercontent.com/zhusz/ICCV17-fashionGAN/master/matrix.png)

Walking the latent space: For each row, the first and the last images are the two samples that we will make the interpolation. We gradually change the input from the left image. In the first row, we only interpolate the input to the first stage and hence the generated results only change in shapes. In the second row, we only interpolate the input to the second stage and hence the results only change in textures. The last row interpolate the input for both the first and second stages and hence the generated interpolated results transfer smoothly from the left to the right.

![](https://raw.githubusercontent.com/zhusz/ICCV17-fashionGAN/master/interp.png)

## Dependency
The implementation is based on [Torch](https://github.com/torch/torch7). CuDNN is required.

## Getting data
Run the following command to obtain the training data and the off-the-shelf pre-trained model. It might take some time. Folders for models are also created here.
```shell
sh download.sh
```

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
Please refer to the `language` folder for training and testing the initial langugae encoding model.

## Feedback
Suggestions and opinions of this work (both positive and negative) are greatly welcome. Please contact the author by sending email to `zhshzhutah2@gmail.com`.

## License
BSD-3, see `LICENSE` file for details.
