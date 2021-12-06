# Enhancing Twin Delayed Deep Deterministic Policy Gradient with Cross-Entropy Method
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

Hieu Trung Nguyen, Khang Tran and Ngoc Hoang Luong
<!-- In NICS 2021. -->
## Setup
- Clone this repo: 
```
$ git clone https://github.com/junhill-2000/eTD3.git
$ cd eTD3
```
- The following packages are needed:
```
tianshou==0.4.4
gym==0.19.0
mujoco_py==2.0.2.13 (need to install mujoco200 from http://www.mujoco.org/)
```

## Usage

### train agent with eTD3
```
python3 mujoco_td3_mod_v3.py --task [environment_name] --seed [seed] --logdir [log_directory]
```

### Visualize (comming soon)

## Acknowledgement
Our source code is inspired by:
- [CEM-RL by Apourchot et al.](https://github.com/apourchot/CEM-RL)
- [Tianshou library](https://github.com/thu-ml/tianshou/tree/master/examples/mujoco)