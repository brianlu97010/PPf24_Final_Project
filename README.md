# A CUDA-Based Parallelization of the JPEG_Encoder 
This project is a modified version of [jpeg_encoder](https://github.com/thejinchao/jpeg_encoder), an open-source JPEG encoder/decoder.


## Modifications

In this modified version, we have focused on:
```
- List your modifications, 
e.g., optimization of DCT and color modules using CUDA for parallel processing
 ``` 
These changes are designed to enhance performance and parallel processing capabilities. 

## Installation
```bash
git clone https://github.com/brianlu97010/PPf24_Final_Project
```
```bash
cd PPf24_Final_Project
```
```bash
make clean && make
```

## Usage
### To encode an image:
```bash
./jpeg {inputFile.bmp} {outputFile.jpg}
```

### To run the test
```bash
./run_test.sh
```

## Performance Results
Testing Hardware : 
待補

### Main Results
| Test Image               | (Avg.) Original Processing Time | (Avg.) CUDA Version Processing Time | SpeedUp |
|--------------------------|---------------------------------|-------------------------------------|---------|
| img/sample_5184×3456.bmp | 4.526 s                         | 0.688 s                             | 6.579x  |

### CUDA Operation
| Test Image               | (Avg.) Memory Allocation & Transfer Time | (Avg.) Kernel Execution Time |
|--------------------------|------------------------------------------|------------------------------|
| img/sample_5184×3456.bmp | 0.01697 s                                | 0.0124 s                     | 

## Contributors
- 呂彥鋒  brianlu.cs13@nycu.edu.tw
  
## References
- [1] [thejinchao/jpeg_encoder: a simple jpeg codec. (github.com) ](https://github.com/thejinchao/jpeg_encoder/tree/master) 
- [2]【平行運算】CUDA教學(一) 概念介紹 - 都會阿嬤 (weikaiwei.com)
- [3] 開發人員部落格系列：初學者「CUDA 總複習」教學 - NVIDIA 台灣官方部落格
- [4] https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- [5] https://developer.nvidia.com/blog/even-easier-introduction-cuda/
- [6]Ghorpade, Jayshree, et al. "GPGPU processing in CUDA architecture." arXiv preprint arXiv:1202.4347 (2012).
- [7] G. K. Wallace, "The JPEG still picture compression standard," in IEEE Transactions on Consumer Electronics, vol. 38, no. 1, pp. xviii-xxxiv, Feb. 1992, doi: 10.1109/30.125072.
- [8]https://zh.wikipedia.org/zh-tw/JPEG
