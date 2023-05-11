# Depth Estimation

[ reference ]
> **Digging into Self-Supervised Monocular Depth Prediction**

> [ICCV 2019 (arXiv pdf)](https://arxiv.org/abs/1806.01260)

<p align="center">
  <img src="assets/teaser.gif" alt="example input output gif" width="600" />
</p>

```
@article{monodepth2,
  title     = {Digging into Self-Supervised Monocular Depth Prediction},
  author    = {Cl{\'{e}}ment Godard and
               Oisin {Mac Aodha} and
               Michael Firman and
               Gabriel J. Brostow},
  booktitle = {The International Conference on Computer Vision (ICCV)},
  month = {October},
year = {2019}
}
```

**Splits**

The train/test/validation splits are defined in the `splits/` folder.
By default, the code will train a depth model using [Zhou's subset](https://github.com/tinghuiz/SfMLearner) of the standard Eigen split of KITTI, which is designed for monocular training.
You can also train a model using the new [benchmark split](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) or the [odometry split](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) by setting the `--split` flag.


**Custom dataset**

You can train on a custom monocular or stereo dataset by writing a new dataloader class which inherits from `MonoDataset` – see the `KITTIDataset` class in `datasets/kitti_dataset.py` for an example.


## ⏳ Training

By default models and tensorboard event files are saved to `~/tmp/<model_name>`.
This can be changed with the `--log_dir` flag.


**Monocular training:**
```shell
python train.py --model_name mono_model
```

### GPUs

The code can only be run on a single GPU.
You can specify which GPU to use with the `CUDA_VISIBLE_DEVICES` environment variable:
```shell
CUDA_VISIBLE_DEVICES=2 python train.py --model_name mono_model
```

All our experiments were performed on a single NVIDIA Titan Xp.

| Training modality | Approximate GPU memory  | Approximate training time   |
|-------------------|-------------------------|-----------------------------|
| Mono              | 9GB                     | 12 hours                    |
| Stereo            | 6GB                     | 8 hours                     |
| Mono + Stereo     | 11GB                    | 15 hours                    |


## 👩‍⚖️ License
Copyright © Niantic, Inc. 2019. Patent Pending.
All rights reserved.
Please see the [license file](LICENSE) for terms.
