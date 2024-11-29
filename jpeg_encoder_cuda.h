#ifndef __JPEG_ENCODER_HEADER__
#define __JPEG_ENCODER_HEADER__

#include <cuda_runtime.h>

class JpegEncoder
{
public:
    /** 清理数据 */
    void clean(void);

    /** 从BMP文件中读取文件，仅支持24bit，长度是8的倍数的文件 */
    bool readFromBMP(const char* fileName);

    /** 压缩到jpg文件中，quality_scale表示质量，取值范围(0,100), 数字越大压缩比例越高*/
    bool encodeToJPG(const char* fileName, int quality_scale);

private:
    int             m_width;
    int             m_height;
    unsigned char*  m_rgbBuffer;
    
    unsigned char   m_YTable[64];
    unsigned char   m_CbCrTable[64];

    struct BitString
    {
        int length;    
        int value;
    };

    BitString m_Y_DC_Huffman_Table[12];
    BitString m_Y_AC_Huffman_Table[256];

    BitString m_CbCr_DC_Huffman_Table[12];
    BitString m_CbCr_AC_Huffman_Table[256];

    // GPU 相關成員
    unsigned char* d_rgbBuffer;    // 設備端 RGB 數據
    short* d_YQuantOutput;         // 設備端 Y 量化輸出
    short* d_CbQuantOutput;        // 設備端 Cb 量化輸出
    short* d_CrQuantOutput;        // 設備端 Cr 量化輸出

private:
    // CPU 端函數 - 維持不變
    void _initHuffmanTables(void);
    void _initQualityTables(int quality);
    void _computeHuffmanTable(const char* nr_codes, const unsigned char* std_table, BitString* huffman_table);
    BitString _getBitCode(int value);

    void _doHuffmanEncoding(const short* DU, short& prevDC, const BitString* HTDC, const BitString* HTAC, 
        BitString* outputBitString, int& bitStringCounts);

    void _write_jpeg_header(FILE* fp);
    void _write_byte_(unsigned char value, FILE* fp);
    void _write_word_(unsigned short value, FILE* fp);
    void _write_bitstring_(const BitString* bs, int counts, int& newByte, int& newBytePos, FILE* fp);
    void _write_(const void* p, int byteSize, FILE* fp);

    // CUDA 記憶體管理
    bool _allocateDeviceMemory();
    void _freeDeviceMemory();
    bool _copyDataToDevice();
    bool _copyDataFromDevice();

    // CUDA kernel 啟動函數
    bool _processImageOnGPU();

public:
    JpegEncoder();
    ~JpegEncoder();
};

// Device functions declaration
namespace jpeg_cuda {
    __device__ void colorTransform(
        const unsigned char* rgbBuffer,
        int stride,
        char* yData,
        char* cbData,
        char* crData,
        int pos);

    __device__ void forwardDCT(
        const char* channel_data,
        short* fdc_data,
        const unsigned char* quant_table,
        int block_pos);

    __global__ void jpegCompressKernel(
        const unsigned char* rgb_buffer,
        char* y_channel,
        char* cb_channel,
        char* cr_channel,
        short* y_dct,
        short* cb_dct,
        short* cr_dct,
        int width,
        int height);
}

#endif