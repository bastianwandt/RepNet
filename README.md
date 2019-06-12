# RepNet

This is the original implementation of the CVPR 2019 Paper "RepNet:  Weakly Supervised Training of an Adversarial Reprojection Network for 3D Human Pose Estimation" by Bastian Wandt and Bodo Rosenhahn.

## Getting Started

Required packages:
* numpy
* scipy
* h5py
* keras
* tensorflow

For visualization you also need:
* matplotlib

## Testing

Clone the repository and run eval_h36m.py for an example evaluation of a sequence of detections from the Human3.6M database.

## Training

Unfortunately, due to licensing it is not possible to provide the training and evaluation data for Human3.6M. However, you can easily create your own training data if you have access to the dataset. For the 3d data the data structure is a simple Fx48 Matlab matrix, where each row describes a single frame. The row vector contains the x,y,z coordinates of the joints in the format (x_1, x_2, ..., x_16, y_1, y_2, ..., y_16, z_1, z_2, ..., z_16). Note that the measurements are in millimeters and all 3d poses are aligned to a template as described in the paper. The 2d data has the same structure except the missing z-coordinates.

After creating your training data run train_repnet.py.

## Citation
Please cite the paper in your publications if it helps your research:

    @inproceedings { Wandt2019RepNet,
      author = {Bastian Wandt and Bodo Rosenhahn},
      title = {RepNet: Weakly Supervised Training of an Adversarial Reprojection Network for 3D Human Pose Estimation},
      booktitle = {Computer Vision and Pattern Recognition (CVPR)},
      year = {2019},
      month = jun
    }

Links to the paper:

- [RepNet: Weakly Supervised Training of an Adversarial Reprojection Network for 3D Human Pose Estimation](https://arxiv.org/abs/1902.09868)

## License

This project is licensed under the MIT License.

