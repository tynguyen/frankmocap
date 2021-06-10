# FrankMocap: A Strong and Easy-to-use Single View 3D Hand+Body Pose Estimator

FrankMocap pursues an easy-to-use single view 3D motion capture system developed by Facebook AI Research (FAIR). FrankMocap provides state-of-the-art 3D pose estimation outputs for body, hand, and body+hands in a single system. The core objective of FrankMocap is to democratize the 3D human pose estimation technology, enabling anyone (researchers, engineers, developers, artists, and others) can easily obtain 3D motion capture outputs from videos and images.

<b>Btw, why the name FrankMocap? </b> Our pipeline to integrate body and hand modules reminds us of [Frankenstein's monster](https://en.wikipedia.org/wiki/Frankenstein)!


## Webcam Demo
Make sure to install additional packages
```
pip install -r docs/requirements.txt
```
Note that I've added some more packages:
- [x] mesh-to-depth


There are two ways to run the demos
* Install this package
* Copy demo files from docs/ to the main repository

## Showing Mesh on the Depth Map Demo
```
python -d demo_show_mesh_depth.py --input_path ~/bags/kinect_record/record_extracted/color --out_dir ~/bags/kinect_record/frankmocap_bodytracking --renderer_type pytorch3d --start_frame 600 --dmap_path ~/bags/kinect_record/record_extracted/depth
```

Note from [this link](https://github.com/facebookresearch/frankmocap/issues/55)
```
The output of pred_img_joints is the 3D coordinates of body joints, where the format is [x, y, z]. x and y is rescaled and translated to image space and z is rescaled accordingly. P.S. The original vertices / joints of SMPL/SMPL-X lies in [-1, 1].
```

Note from [this link](https://github.com/facebookresearch/frankmocap/issues/59)
```
Hi! I am using your awesome repository to extract 3D hand skeleton coordinates. So far, the estimated joints pred_joints_img are pretty accurate with respect to the original image. I can see how the X/Y coordinate values refer to the image pixels, however, the Z (depth) is relative to the hand, not the camera or the world.
```

Note from [this link](https://github.com/facebookresearch/frankmocap/issues/21)
```
pred_vertices_img is the SMPL-X vertices re-scaled to the image space. If you want coordinates in the world space, then you need to know the camera extrinsic parameters, which is not covered by our project.
```

Note from [this link](https://github.com/facebookresearch/frankmocap/issues/25)
```
As far as I know, the depth of the raw 3D coordinates is relative to the base of the index finger, not the camera or the world.
```


### Run
```
python demo.demo_bodymocap --input_path webcam --out_dir ./mocap_output
```


### News:
  - [2020/10/09] We have improved openGL rendering speed. It's about 40% faster. (e.g., body module: 6fps -> 11fps)

## Key Features
- Body Motion Capture:
<p>
    <img src="https://github.com/jhugestar/jhugestar.github.io/blob/master/img/eft_bodymocap.gif" height="200">
</p>

- Hand Motion Capture
<p>
    <img src="https://github.com/jhugestar/jhugestar.github.io/blob/master/img/frankmocap_hand.gif" height="200">
</p>

- Egocentric Hand Motion Capture
<p>
    <img src="https://github.com/jhugestar/jhugestar.github.io/blob/master/img/frankmotion_egohand.gif" height="150">
</p>

- Whole body Motion Capture (body + hands)
<p>
    <img src="https://github.com/jhugestar/jhugestar.github.io/blob/master/img/frankmocap_wholebody.gif" height="200">
</p>
<p>
    <img src="https://penincillin.github.io/frank_mocap/video_02.gif" height="200">
</p>


## Installation
- See [INSTALL.md](docs/INSTALL.md)

## A Quick Start
- Run body motion capture
  ```
  # using a machine with a monitor to show output on screen
  python -m demo.demo_bodymocap --input_path ./sample_data/han_short.mp4 --out_dir ./mocap_output

  # screenless mode (e.g., a remote server)
  xvfb-run -a python -m demo.demo_bodymocap --input_path ./sample_data/han_short.mp4 --out_dir ./mocap_output
  ```

- Run hand motion capture
  ```
  # using a machine with a monitor to show outputs on screen
  python -m demo.demo_handmocap --input_path ./sample_data/han_hand_short.mp4 --out_dir ./mocap_output

  # screenless mode  (e.g., a remote server)
  xvfb-run -a python -m demo.demo_handmocap --input_path ./sample_data/han_hand_short.mp4 --out_dir ./mocap_output
  ```

- Run whole body motion capture
  ```
  # using a machine with a monitor to show outputs on screen
  python -m demo.demo_frankmocap --input_path ./sample_data/han_short.mp4 --out_dir ./mocap_output

  # screenless mode  (e.g., a remote server)
  xvfb-run -a python -m demo.demo_frankmocap --input_path ./sample_data/han_short.mp4 --out_dir ./mocap_output
  ```
- Note:
  - Above commands use openGL by default. If it does not work, you may try alternative renderers (pytorch3d or openDR).
  - See the readme of each module for details


## Joint Order
- See [joint_order](docs/joint_order.md)


## Body Motion Capture Module
- See [run_bodymocap](docs/run_bodymocap.md)

## Hand Motion Capture Module
- See [run_handmocap](docs/run_handmocap.md)

## Whole Body Motion Capture Module (Body + Hand)
- See [run_totalmocap](docs/run_totalmocap.md)

## License
- [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode).
See the [LICENSE](LICENSE) file.

## References
- FrankMocap is based on the following research outputs:
```
@article{rong2020frankmocap,
  title={FrankMocap: Fast Monocular 3D Hand and Body Motion Capture by Regression and Integration},
  author={Rong, Yu and Shiratori, Takaaki and Joo, Hanbyul},
  journal={arXiv preprint arXiv:2008.08324},
  year={2020}
}

@article{joo2020eft,
  title={Exemplar Fine-Tuning for 3D Human Pose Fitting Towards In-the-Wild 3D Human Pose Estimation},
  author={Joo, Hanbyul and Neverova, Natalia and Vedaldi, Andrea},
  journal={arXiv preprint arXiv:2004.03686},
  year={2020}
}
```

- FrankMocap leverages many amazing open-sources shared in research community.
    - [SMPL](https://smpl.is.tue.mpg.de/), [SMPLX](https://smpl-x.is.tue.mpg.de/)
    - [Detectron2](https://github.com/facebookresearch/detectron2)
    - [Pytorch3D](https://pytorch3d.org/) (for rendering)
    - [OpenDR](https://github.com/mattloper/opendr/wiki) (for rendering)
    - [SPIN](https://github.com/nkolot/SPIN) (for body module)
    - [100DOH](https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/) (for hand detection)
    - [lightweight-human-pose-estimation](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) (for body detection)
