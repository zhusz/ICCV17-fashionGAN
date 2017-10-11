# ICCV17-fashionGAN
------
Shizhan Zhu

Released on Oct 11, 2017

## Description

This is the implementation of Shizhan Zhu et al.'s ICCV-17 work [Be Your Own Prada: Fashion Synthesis with Structural Coherence](http://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Be_Your_Own_ICCV_2017_paper.html). It is open source under BSD-3 license (see the `LICENSE` file). Codes can be used freely only for academic purpose. If you want to apply it to industrial products, please send an email to Shizhan Zhu at `zhshzhutah2@gmail.com` first.

## Qualitative Results

![alt tag](https://www.dropbox.com/s/8ws8esku27wbfel/matrix.png)

## Dependency
The implementation is based on [Torch](https://github.com/torch/torch7). CuDNN is required if you use the released off-the-shelf pre-trained model.

## Getting data
Run the following command to obtain the training data and the off-the-shelf pre-trained model. It might take some time.
```shell
sh download.sh
```

## Testing
All the testing codes are in the `demo_release` folder. The GAN of our second stage has three options in our implementation.
1. Run `demo_full.lua` with [this](https://github.com/zhusz/ICCV17-fashionGAN/blob/master/demo_release/demo_full.lua#L143) line uncommented. The network structure is our original submitted version.
2. Run `demo_full.lua` as it is. It adds the skip connection technique proposed in [Hour-glass](https://github.com/anewell/pose-hg-demo) and [pix2pix](https://github.com/phillipi/pix2pix).
3. Run `demo_p2p.lua`. The network structure completely follows [pix2pix](https://github.com/phillipi/pix2pix). The texture would be nice but cannot be controlled.

## Training
1. To train the first-stage-gan, enter the `sr1` folder and run the `train.lua` file.
2. To train the second-stage-gan, enter the relevant folder to run the `train.lua` file. Folder `ih1` refers to our original submission. Filder `ih1_skip` refers to the second-stage-network coupled with skip connection. Folder `ih1_p2p` uses pix2pix as our second stage.

## Feedback
Suggestions and opinions of this work (both positive and negative) are greatly welcome. Please contact the author by sending email to `zhshzhutah2@gmail.com`.

## License
BSD-3, see `LICENSE` file for details.
