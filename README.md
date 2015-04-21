#CUDA implementation of dehazing algorithm using dark channel prior

##Feature
- OpenCV to read images and processing them on GPU
- Shared memory optimization
- Multi-platform support (Windows, Linux, Mac)

##Usage

Make sure you have openCV, CUDA toolkit installed and a NVIDIA graphic card

```sh
git clone https://github.com/arsenalliu123/dehazing-GPU.git
cd dehazing-GPU
make clean && make
Debug/dehazing -h
```

##Todo
+ Add transmission image refinition

**Developed by Yichen Liu and Yin Lin**