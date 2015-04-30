#CUDA implementation of dehazing algorithm using dark channel prior

Dehazing algorithm implementation on CUDA.

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

**Developed by Yichen Liu and Yin Lin**
