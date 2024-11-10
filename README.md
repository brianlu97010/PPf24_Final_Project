# A CUDA-Based Parallelization of the ffjpeg Library
This project is a modified version of [ffjpeg](https://github.com/rockcarry/ffjpeg), an open-source JPEG encoder/decoder.

## License

The original ffjpeg project is licensed under the GNU General Public License v3.0 (GPL-3.0). As per the terms of this license, any modifications to the original source code are also provided under the same license.

## Modifications

In this modified version, we have focused on:
```
- List your modifications, 
e.g., optimization of DCT and color modules using CUDA for parallel processing
 ``` 
These changes are designed to enhance performance and parallel processing capabilities. 

## Installation
```bash
git clone https://github.com/yourusername/modified-ffjpeg
cd PPf24_Final_Project
make
```

## Usage
To encode an image:
```bash
./ffjpeg input.bmp output.jpg
```

## Performance Results
| Test Image | Original Processing Time | Modified Processing Time |
|------------|--------------------------|--------------------------|
| image1.jpg | 2.5s                     | 1.2s                     |


## Contributors
- 呂彥鋒  brianlu.cs13@nycu.edu.tw
  
## References
[1] rockcarry/ffjpeg: a simple jpeg codec. (github.com)
[2]【平行運算】CUDA教學(一) 概念介紹 - 都會阿嬤 (weikaiwei.com)
[3] 開發人員部落格系列：初學者「CUDA 總複習」教學 - NVIDIA 台灣官方部落格
[4] https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
[5] https://developer.nvidia.com/blog/even-easier-introduction-cuda/
[6] https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
[7]Ghorpade, Jayshree, et al. "GPGPU processing in CUDA architecture." arXiv preprint arXiv:1202.4347 (2012).
[8] G. K. Wallace, "The JPEG still picture compression standard," in IEEE Transactions on Consumer Electronics, vol. 38, no. 1, pp. xviii-xxxiv, Feb. 1992, doi: 10.1109/30.125072.
[9]https://zh.wikipedia.org/zh-tw/JPEG