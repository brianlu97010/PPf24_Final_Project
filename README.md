# A CUDA-Based Parallelization of the JPEG_Encoder 
This project is a modified version of [jpeg_encoder](https://github.com/thejinchao/jpeg_encoder), an open-source JPEG encoder/decoder.


## Modifications*
1. 用 `cudaMemcpyToSymbol` 將 ZigZag table, Quality Table (Y_table and CbCr_table) copy 到 device 的 constant memory
2. 用 `cudaMalloc` allocate device 的 memory space
3. 用 `cudaMemcpy` 把 `m_rgbBuffer` (整個 image 的 pixel value) copy 到 `d_rgb`
4. 因為 DCT 是一次算一整個 8x8 的區域 ，因此 kernel 的一個 block 也用 8x8 個 threads，一個 thread 處理一個區域的 pixel，總共需要幾個 blocks ? 這邊假設 width 跟 height 都是 8 的倍數，因此不用處理額外區域，共需 `( m_width / 8, m_height / 8 )` 個 blocks，共 `m_width * m_height` 個 threads。 
   -  `dim3 numBlocks((m_width + 7)/8, (m_height + 7)/8);`
   -  `dim3 threadsPerBlock(8, 8);`
5.  

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
To encode an image:
```bash
./jpeg_encoder {inputFile} {outputFile}
```

## Performance Results
| Test Image | Original Processing Time | Modified Processing Time |
|------------|--------------------------|--------------------------|
| image1.jpg | 2.5s                     | 1.2s                     |


## Contributors
- 呂彥鋒  brianlu.cs13@nycu.edu.tw
  
## References
- [1]  
- [2]【平行運算】CUDA教學(一) 概念介紹 - 都會阿嬤 (weikaiwei.com)
- [3] 開發人員部落格系列：初學者「CUDA 總複習」教學 - NVIDIA 台灣官方部落格
- [4] https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- [5] https://developer.nvidia.com/blog/even-easier-introduction-cuda/
- [6]Ghorpade, Jayshree, et al. "GPGPU processing in CUDA architecture." arXiv preprint arXiv:1202.4347 (2012).
- [7] G. K. Wallace, "The JPEG still picture compression standard," in IEEE Transactions on Consumer Electronics, vol. 38, no. 1, pp. xviii-xxxiv, Feb. 1992, doi: 10.1109/30.125072.
- [8]https://zh.wikipedia.org/zh-tw/JPEG