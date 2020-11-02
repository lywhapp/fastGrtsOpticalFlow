# Introduction

fast optical flow estimation for videos based on 3d-gradient. ANN is modified based on the source code from [PatchMatch](https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/index.php).

# How to Use

```
$ mkdir build
$ cd build
$ cmake ..
$ make
$ ./fastGrtsOpticalFlow VideoFile
```

# Reference

```
@article{2017Fast,
  title={Fast Optical Flow Estimation Without Parallel Architectures},
  author={ Zhu, En  and  Li, Yuanwei  and  Shi, Yanling },
  journal={ieee transactions on circuits & systems for video technology},
  volume={27},
  number={11},
  pages={2322-2332},
  year={2017},
}
```