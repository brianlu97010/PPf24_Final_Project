#include <stdio.h>
#include <math.h>
#include <memory.h>
#include <cuda_runtime.h>
#include "jpeg_encoder_cuda.h"

#define BLOCK_SIZE 8
#define THREADS_PER_BLOCK 64  // 8x8 threads for each block

// Device Constants
namespace {
    __constant__ char ZigZag_d[64];
    __constant__ unsigned char YTable_d[64];
    __constant__ unsigned char CbCrTable_d[64];
    const float PI = 3.1415926f;
}

// Useful table for jpeg
namespace {
//-------------------------------------------------------------------------------
const unsigned char Luminance_Quantization_Table[64] = 
{
	16,  11,  10,  16,  24,  40,  51,  61,
	12,  12,  14,  19,  26,  58,  60,  55,
	14,  13,  16,  24,  40,  57,  69,  56,
	14,  17,  22,  29,  51,  87,  80,  62,
	18,  22,  37,  56,  68, 109, 103,  77,
	24,  35,  55,  64,  81, 104, 113,  92,
	49,  64,  78,  87, 103, 121, 120, 101,
	72,  92,  95,  98, 112, 100, 103,  99
};

//-------------------------------------------------------------------------------
const unsigned char Chrominance_Quantization_Table[64] = 
{
	17,  18,  24,  47,  99,  99,  99,  99,
	18,  21,  26,  66,  99,  99,  99,  99,
	24,  26,  56,  99,  99,  99,  99,  99,
	47,  66,  99,  99,  99,  99,  99,  99,
	99,  99,  99,  99,  99,  99,  99,  99,
	99,  99,  99,  99,  99,  99,  99,  99,
	99,  99,  99,  99,  99,  99,  99,  99,
	99,  99,  99,  99,  99,  99,  99,  99
};

//-------------------------------------------------------------------------------
const char ZigZag[64] =
{ 
	 0, 1, 5, 6,14,15,27,28,
	 2, 4, 7,13,16,26,29,42,
	 3, 8,12,17,25,30,41,43,
	 9,11,18,24,31,40,44,53,
	10,19,23,32,39,45,52,54,
	20,22,33,38,46,51,55,60,
	21,34,37,47,50,56,59,61,
	35,36,48,49,57,58,62,63 
};     

//-------------------------------------------------------------------------------
const char Standard_DC_Luminance_NRCodes[] = { 0, 0, 7, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
const unsigned char Standard_DC_Luminance_Values[] = { 4, 5, 3, 2, 6, 1, 0, 7, 8, 9, 10, 11 };

//-------------------------------------------------------------------------------
const char Standard_DC_Chrominance_NRCodes[] = { 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
const unsigned char Standard_DC_Chrominance_Values[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

//-------------------------------------------------------------------------------
const char Standard_AC_Luminance_NRCodes[] = { 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d };
const unsigned char Standard_AC_Luminance_Values[] = 
{
	0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
	0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
	0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
	0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
	0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
	0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
	0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
	0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
	0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
	0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
	0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
	0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
	0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
	0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
	0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
	0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
	0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
	0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
	0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
	0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
	0xf9, 0xfa
};

//-------------------------------------------------------------------------------
const char Standard_AC_Chrominance_NRCodes[] = { 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77 };
const unsigned char Standard_AC_Chrominance_Values[] =
{
	0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
	0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
	0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
	0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
	0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
	0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
	0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
	0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
	0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
	0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
	0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
	0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
	0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
	0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
	0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
	0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
	0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
	0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
	0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
	0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
	0xf9, 0xfa
};

}

//-------------------------------------------------------------------------------
JpegEncoder::JpegEncoder()
	: m_width(0)
	, m_height(0)
	, m_rgbBuffer(0)
{
	//初始化静态表格
	_initHuffmanTables();
}

//-------------------------------------------------------------------------------
JpegEncoder::~JpegEncoder()
{
	clean();
}

//-------------------------------------------------------------------------------
void JpegEncoder::clean(void)
{
	if(m_rgbBuffer) delete[] m_rgbBuffer;
	m_rgbBuffer=0;

	m_width=0;
	m_height=0;
}

//-------------------------------------------------------------------------------
bool JpegEncoder::readFromBMP(const char* fileName)
{
	//清理旧数据
	clean();

	//BMP 文件格式
#pragma pack(push, 2)
	typedef struct {
			unsigned short	bfType;
			unsigned int	bfSize;
			unsigned short	bfReserved1;
			unsigned short	bfReserved2;
			unsigned int	bfOffBits;
	} BITMAPFILEHEADER;

	typedef struct {
			unsigned int	biSize;
			int				biWidth;
			int				biHeight;
			unsigned short	biPlanes;
			unsigned short	biBitCount;
			unsigned int	biCompression;
			unsigned int	biSizeImage;
			int				biXPelsPerMeter;
			int				biYPelsPerMeter;
			unsigned int	biClrUsed;
			unsigned int	biClrImportant;
	} BITMAPINFOHEADER;
#pragma pack(pop)

	//打开文件
	FILE* fp = fopen(fileName, "rb");
	if(fp==0) return false;

	bool successed=false;
	do
	{
		BITMAPFILEHEADER fileHeader;
		BITMAPINFOHEADER infoHeader;

		if(1 != fread(&fileHeader, sizeof(fileHeader), 1, fp)) break;
		if(fileHeader.bfType!=0x4D42) break;

		if(1 != fread(&infoHeader, sizeof(infoHeader), 1, fp)) break;
		if(infoHeader.biBitCount!=24 || infoHeader.biCompression!=0) break;
		int width = infoHeader.biWidth;
		int height = infoHeader.biHeight < 0 ? (-infoHeader.biHeight) : infoHeader.biHeight;
		if((width&7)!=0 || (height&7)!=0) break;	//必须是8的倍数

		int bmpSize = width*height*3;

		unsigned char* buffer = new unsigned char[bmpSize];
		if(buffer==0) break;

		fseek(fp, fileHeader.bfOffBits, SEEK_SET);

		if(infoHeader.biHeight>0)
		{
			for(int i=0; i<height; i++)
			{
				if(width != fread(buffer+(height-1-i)*width*3, 3, width, fp)) 
				{
					delete[] buffer; buffer=0;
					break;
				}
			}
		}
		else
		{
			if(width*height != fread(buffer, 3, width*height, fp))
			{
				delete[] buffer; buffer=0;
				break;
			}
		}

		m_rgbBuffer = buffer;
		m_width = width;
		m_height = height;
		successed=true;
	}while(false);

	fclose(fp);fp=0;
	
	return successed;
}

//-------------------------------------------------------------------------------
void JpegEncoder::_initHuffmanTables(void)
{
	memset(&m_Y_DC_Huffman_Table, 0, sizeof(m_Y_DC_Huffman_Table));
	_computeHuffmanTable(Standard_DC_Luminance_NRCodes, Standard_DC_Luminance_Values, m_Y_DC_Huffman_Table);

	memset(&m_Y_AC_Huffman_Table, 0, sizeof(m_Y_AC_Huffman_Table));
	_computeHuffmanTable(Standard_AC_Luminance_NRCodes, Standard_AC_Luminance_Values, m_Y_AC_Huffman_Table);

	memset(&m_CbCr_DC_Huffman_Table, 0, sizeof(m_CbCr_DC_Huffman_Table));
	_computeHuffmanTable(Standard_DC_Chrominance_NRCodes, Standard_DC_Chrominance_Values, m_CbCr_DC_Huffman_Table);

	memset(&m_CbCr_AC_Huffman_Table, 0, sizeof(m_CbCr_AC_Huffman_Table));
	_computeHuffmanTable(Standard_AC_Chrominance_NRCodes, Standard_AC_Chrominance_Values, m_CbCr_AC_Huffman_Table);
}
//-------------------------------------------------------------------------------
JpegEncoder::BitString JpegEncoder::_getBitCode(int value)
{
	BitString ret;
	int v = (value>0) ? value : -value;
	
	//bit 的长度
	int length = 0;
	for(length=0; v; v>>=1) length++;

	ret.value = value>0 ? value : ((1<<length)+value-1);
	ret.length = length;

	return ret;
};

//-------------------------------------------------------------------------------
void JpegEncoder::_initQualityTables(int quality_scale)
{
	if(quality_scale<=0) quality_scale=1;
	if(quality_scale>=100) quality_scale=99;

	for(int i=0; i<64; i++)
	{
		int temp = ((int)(Luminance_Quantization_Table[i] * quality_scale + 50) / 100);
		if (temp<=0) temp = 1;
		if (temp>0xFF) temp = 0xFF;
		m_YTable[ZigZag[i]] = (unsigned char)temp;

		temp = ((int)(Chrominance_Quantization_Table[i] * quality_scale + 50) / 100);
		if (temp<=0) 	temp = 1;
		if (temp>0xFF) temp = 0xFF;
		m_CbCrTable[ZigZag[i]] = (unsigned char)temp;
	}
}

//-------------------------------------------------------------------------------
void JpegEncoder::_computeHuffmanTable(const char* nr_codes, const unsigned char* std_table, BitString* huffman_table)
{
	unsigned char pos_in_table = 0;
	unsigned short code_value = 0;

	for(int k = 1; k <= 16; k++)
	{
		for(int j = 1; j <= nr_codes[k-1]; j++)
		{
			huffman_table[std_table[pos_in_table]].value = code_value;
			huffman_table[std_table[pos_in_table]].length = k;
			pos_in_table++;
			code_value++;
		}
		code_value <<= 1;
	}  
}

//-------------------------------------------------------------------------------
void JpegEncoder::_write_byte_(unsigned char value, FILE* fp)
{
	_write_(&value, 1, fp);
}

//-------------------------------------------------------------------------------
void JpegEncoder::_write_word_(unsigned short value, FILE* fp)
{
	unsigned short _value = ((value>>8)&0xFF) | ((value&0xFF)<<8);
	_write_(&_value, 2, fp);
}

//-------------------------------------------------------------------------------
void JpegEncoder::_write_(const void* p, int byteSize, FILE* fp)
{
	fwrite(p, 1, byteSize, fp);
}

//-------------------------------------------------------------------------------
void JpegEncoder::_doHuffmanEncoding(const short* DU, short& prevDC, const BitString* HTDC, const BitString* HTAC, 
	BitString* outputBitString, int& bitStringCounts)
{
	BitString EOB = HTAC[0x00];
	BitString SIXTEEN_ZEROS = HTAC[0xF0];

	int index=0;

	// encode DC
	int dcDiff = (int)(DU[0] - prevDC);
	prevDC = DU[0];

	if (dcDiff == 0) 
		outputBitString[index++] = HTDC[0];
	else
	{
		BitString bs = _getBitCode(dcDiff);

		outputBitString[index++] = HTDC[bs.length];
		outputBitString[index++] = bs;
	}

	// encode ACs
	int endPos=63; //end0pos = first element in reverse order != 0
	while((endPos > 0) && (DU[endPos] == 0)) endPos--;

	for(int i=1; i<=endPos; )
	{
		int startPos = i;
		while((DU[i] == 0) && (i <= endPos)) i++;

		int zeroCounts = i - startPos;
		if (zeroCounts >= 16)
		{
			for (int j=1; j<=zeroCounts/16; j++)
				outputBitString[index++] = SIXTEEN_ZEROS;
			zeroCounts = zeroCounts%16;
		}

		BitString bs = _getBitCode(DU[i]);

		outputBitString[index++] = HTAC[(zeroCounts << 4) | bs.length];
		outputBitString[index++] = bs;
		i++;
	}

	if (endPos != 63)
		outputBitString[index++] = EOB;

	bitStringCounts = index;
}

//-------------------------------------------------------------------------------
void JpegEncoder::_write_bitstring_(const BitString* bs, int counts, int& newByte, int& newBytePos, FILE* fp)
{
	unsigned short mask[] = {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768};
	
	for(int i=0; i<counts; i++)
	{
		int value = bs[i].value;
		int posval = bs[i].length - 1;

		while (posval >= 0)
		{
			if ((value & mask[posval]) != 0)
			{
				newByte = newByte  | mask[newBytePos];
			}
			posval--;
			newBytePos--;
			if (newBytePos < 0)
			{
				// Write to stream
				_write_byte_((unsigned char)(newByte), fp);
				if (newByte == 0xFF)
				{
					// Handle special case
					_write_byte_((unsigned char)(0x00), fp);
				}

				// Reinitialize
				newBytePos = 7;
				newByte = 0;
			}
		}
	}
}



//-------------------------------------------------------------------------------
void JpegEncoder::_write_jpeg_header(FILE* fp)
{
	//SOI
	_write_word_(0xFFD8, fp);		// marker = 0xFFD8

	//APPO
	_write_word_(0xFFE0,fp);		// marker = 0xFFE0
	_write_word_(16, fp);			// length = 16 for usual JPEG, no thumbnail
	_write_("JFIF", 5, fp);			// 'JFIF\0'
	_write_byte_(1, fp);			// version_hi
	_write_byte_(1, fp);			// version_low
	_write_byte_(0, fp);			// xyunits = 0 no units, normal density
	_write_word_(1, fp);			// xdensity
	_write_word_(1, fp);			// ydensity
	_write_byte_(0, fp);			// thumbWidth
	_write_byte_(0, fp);			// thumbHeight

	//DQT
	_write_word_(0xFFDB, fp);		//marker = 0xFFDB
	_write_word_(132, fp);			//size=132
	_write_byte_(0, fp);			//QTYinfo== 0:  bit 0..3: number of QT = 0 (table for Y) 
									//				bit 4..7: precision of QT
									//				bit 8	: 0
	_write_(m_YTable, 64, fp);		//YTable
	_write_byte_(1, fp);			//QTCbinfo = 1 (quantization table for Cb,Cr)
	_write_(m_CbCrTable, 64, fp);	//CbCrTable

	//SOFO
	_write_word_(0xFFC0, fp);			//marker = 0xFFC0
	_write_word_(17, fp);				//length = 17 for a truecolor YCbCr JPG
	_write_byte_(8, fp);				//precision = 8: 8 bits/sample 
	_write_word_(m_height&0xFFFF, fp);	//height
	_write_word_(m_width&0xFFFF, fp);	//width
	_write_byte_(3, fp);				//nrofcomponents = 3: We encode a truecolor JPG

	_write_byte_(1, fp);				//IdY = 1
	_write_byte_(0x11, fp);				//HVY sampling factors for Y (bit 0-3 vert., 4-7 hor.)(SubSamp 1x1)
	_write_byte_(0, fp);				//QTY  Quantization Table number for Y = 0

	_write_byte_(2, fp);				//IdCb = 2
	_write_byte_(0x11, fp);				//HVCb = 0x11(SubSamp 1x1)
	_write_byte_(1, fp);				//QTCb = 1

	_write_byte_(3, fp);				//IdCr = 3
	_write_byte_(0x11, fp);				//HVCr = 0x11 (SubSamp 1x1)
	_write_byte_(1, fp);				//QTCr Normally equal to QTCb = 1
	
	//DHT
	_write_word_(0xFFC4, fp);		//marker = 0xFFC4
	_write_word_(0x01A2, fp);		//length = 0x01A2
	_write_byte_(0, fp);			//HTYDCinfo bit 0..3	: number of HT (0..3), for Y =0
									//			bit 4		: type of HT, 0 = DC table,1 = AC table
									//			bit 5..7	: not used, must be 0
	_write_(Standard_DC_Luminance_NRCodes, sizeof(Standard_DC_Luminance_NRCodes), fp);	//DC_L_NRC
	_write_(Standard_DC_Luminance_Values, sizeof(Standard_DC_Luminance_Values), fp);		//DC_L_VALUE
	_write_byte_(0x10, fp);			//HTYACinfo
	_write_(Standard_AC_Luminance_NRCodes, sizeof(Standard_AC_Luminance_NRCodes), fp);
	_write_(Standard_AC_Luminance_Values, sizeof(Standard_AC_Luminance_Values), fp); //we'll use the standard Huffman tables
	_write_byte_(0x01, fp);			//HTCbDCinfo
	_write_(Standard_DC_Chrominance_NRCodes, sizeof(Standard_DC_Chrominance_NRCodes), fp);
	_write_(Standard_DC_Chrominance_Values, sizeof(Standard_DC_Chrominance_Values), fp);
	_write_byte_(0x11, fp);			//HTCbACinfo
	_write_(Standard_AC_Chrominance_NRCodes, sizeof(Standard_AC_Chrominance_NRCodes), fp);
	_write_(Standard_AC_Chrominance_Values, sizeof(Standard_AC_Chrominance_Values), fp);

	//SOS
	_write_word_(0xFFDA, fp);		//marker = 0xFFC4
	_write_word_(12, fp);			//length = 12
	_write_byte_(3, fp);			//nrofcomponents, Should be 3: truecolor JPG

	_write_byte_(1, fp);			//Idy=1
	_write_byte_(0, fp);			//HTY	bits 0..3: AC table (0..3)
									//		bits 4..7: DC table (0..3)
	_write_byte_(2, fp);			//IdCb
	_write_byte_(0x11, fp);			//HTCb

	_write_byte_(3, fp);			//IdCr
	_write_byte_(0x11, fp);			//HTCr

	_write_byte_(0, fp);			//Ss not interesting, they should be 0,63,0
	_write_byte_(0x3F, fp);			//Se
	_write_byte_(0, fp);			//Bf
}

//////////////////////////////////////////////////
// CUDA Version
// // Color transform device function
// __device__ void colorTransform(
//     const unsigned char* rgbBuffer,
//     int stride,
//     char* yData,
//     char* cbData,
//     char* crData,
//     int pos) 
// {
//     unsigned char B = rgbBuffer[pos * 3];
//     unsigned char G = rgbBuffer[pos * 3 + 1];
//     unsigned char R = rgbBuffer[pos * 3 + 2];

//     yData[pos] = (char)(0.299f * R + 0.587f * G + 0.114f * B - 128);
//     cbData[pos] = (char)(-0.1687f * R - 0.3313f * G + 0.5f * B);
//     crData[pos] = (char)(0.5f * R - 0.4187f * G - 0.0813f * B);
// }

// DCT device function
__device__ void forwardDCT(
    const char* channel_data,
    short* fdc_data,
    const unsigned char* quant_table,
	int height,
    int width,
    int block_x,
    int block_y)
{
    __shared__ float block_data[8][8];
    const float PI = 3.1415926f;
    
    // 使用 2D thread 索引
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 載入數據到 shared memory
    int global_x = block_x * 8 + tx;
    int global_y = block_y * 8 + ty;

    block_data[ty][tx] = 0.0f;	

    if (global_x < width && global_y < height) {  // 確保不超出邊界
        block_data[ty][tx] = channel_data[global_y * width + global_x];
    }

    __syncthreads();
    
    // 計算 DCT
    int u = tx;
    int v = ty;
    
    float alpha_u = (u == 0) ? 1.0f/sqrtf(8.0f) : 0.5f;
    float alpha_v = (v == 0) ? 1.0f/sqrtf(8.0f) : 0.5f;
    
    float sum = 0.0f;
    #pragma unroll
    for (int y = 0; y < 8; y++) {
        #pragma unroll
        for (int x = 0; x < 8; x++) {
            float data = block_data[y][x];
            float cos_u = cosf((2.0f * x + 1.0f) * u * PI / 16.0f);
            float cos_v = cosf((2.0f * y + 1.0f) * v * PI / 16.0f);
            sum += data * cos_u * cos_v;
        }
    }
    
    sum *= alpha_u * alpha_v;
    
    // 修正輸出索引計算
    int zigzag_idx = ZigZag_d[v * 8 + u];
    float quantized = sum / quant_table[zigzag_idx];
    
    // 修正 block 偏移計算
    int block_idx = (block_y * (width/8) + block_x) * 64;
    fdc_data[block_idx + zigzag_idx] = (short)((short)(quantized + 16384.5f) - 16384);

}

// 修改主 kernel
__global__ void jpegCompressKernel(
    const unsigned char* rgb_buffer,
    char* y_channel,
    char* cb_channel,
    char* cr_channel,
    short* y_dct,
    short* cb_dct,
    short* cr_dct,
    int width,
    int height)
{
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    
    // 直接使用 threadIdx.x 和 threadIdx.y
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    
    // 計算在 original image 中的位置
    int global_x = block_x * 8 + local_x;
    int global_y = block_y * 8 + local_y;
    
    // Step 1: Color Transform
    // 有用 if 做邊界檢查
    if (global_x < width && global_y < height) {
        // 取得 channel R, G, B 的 value
        int rgb_pos = (global_y*width + global_x) * 3;
        float R = rgb_buffer[rgb_pos + 2];
        float G = rgb_buffer[rgb_pos + 1];
        float B = rgb_buffer[rgb_pos];
        
        int pos = global_y * width + global_x;
        y_channel[pos] = (char)(0.299f * R + 0.587f * G + 0.114f * B - 128);
        cb_channel[pos] = (char)(-0.1687f * R - 0.3313f * G + 0.5f * B);
        cr_channel[pos] = (char)(0.5f * R - 0.4187f * G - 0.0813f * B);
    }
        
    __syncthreads();
    
    // Step 2: DCT and Quantization
    forwardDCT(y_channel, y_dct, YTable_d, height, width, block_x, block_y);
    forwardDCT(cb_channel, cb_dct, CbCrTable_d, height, width, block_x, block_y);
    forwardDCT(cr_channel, cr_dct, CbCrTable_d, height, width, block_x, block_y);
}

// Modified encodeToJPG function
bool JpegEncoder::encodeToJPG(const char* fileName, int quality_scale)
{
    if(m_rgbBuffer==0 || m_width==0 || m_height==0) return false;
    
    FILE* fp = fopen(fileName, "wb");
    if(fp==0) return false;

    // 創建 CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float encode_time = 0.0f;


    _initQualityTables(quality_scale);
    _write_jpeg_header(fp);
    
    // 開始計時 - 記憶體配置階段
    cudaEventRecord(start);
    
    // Copy tables to device constant memory
    cudaMemcpyToSymbol(YTable_d, m_YTable, sizeof(m_YTable));
    cudaMemcpyToSymbol(CbCrTable_d, m_CbCrTable, sizeof(m_CbCrTable));
    cudaMemcpyToSymbol(ZigZag_d, ZigZag, sizeof(ZigZag));

    // Allocate device memory
    unsigned char *d_rgb;
    char *d_y, *d_cb, *d_cr;
    short *d_y_dct, *d_cb_dct, *d_cr_dct;
    
    size_t rgb_size = m_width * m_height * 3;
    size_t channel_size = m_width * m_height * sizeof(char);
    size_t dct_size = m_width * m_height * sizeof(short);
    
    // Allocate and copy data to device
    cudaMalloc(&d_rgb, rgb_size);
    cudaMalloc(&d_y, channel_size);
    cudaMalloc(&d_cb, channel_size);
    cudaMalloc(&d_cr, channel_size);
    cudaMalloc(&d_y_dct, dct_size);
    cudaMalloc(&d_cb_dct, dct_size);
    cudaMalloc(&d_cr_dct, dct_size);
    
    cudaMemcpy(d_rgb, m_rgbBuffer, rgb_size, cudaMemcpyHostToDevice);

    // 記錄記憶體配置時間
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float mem_time;
    cudaEventElapsedTime(&mem_time, start, stop);
    printf("Memory allocation and transfer time: %.2f ms\n", mem_time);
    encode_time += mem_time;

    // 啟動 kernel - 開始計時
    cudaEventRecord(start);
    
    // Launch kernel
    dim3 numBlocks((m_width + 7)/8, (m_height + 7)/8);
    dim3 threadsPerBlock(8, 8); // 64 threads for 8x8 block
    jpegCompressKernel<<<numBlocks, threadsPerBlock>>>(
        d_rgb, d_y, d_cb, d_cr,
        d_y_dct, d_cb_dct, d_cr_dct,
        m_width, m_height
    );
    
    // 記錄 kernel 執行時間
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, stop);
    printf("Kernel execution time: %.2f ms\n", kernel_time);
    encode_time += kernel_time;
    
    // 開始計時 - return value from device
    cudaEventRecord(start);

    // Allocate host memory for results
    short* h_y_dct = new short[m_width * m_height];
    short* h_cb_dct = new short[m_width * m_height];
    short* h_cr_dct = new short[m_width * m_height];
    
    // Copy results back to host
    cudaMemcpy(h_y_dct, d_y_dct, dct_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cb_dct, d_cb_dct, dct_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cr_dct, d_cr_dct, dct_size, cudaMemcpyDeviceToHost);

    // 記錄結果傳回時間
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float copy_back_time;
    cudaEventElapsedTime(&copy_back_time, start, stop);
    printf("Copy back time: %.2f ms\n", copy_back_time);
    encode_time += copy_back_time;

    // 開始計時 - Huffman 編碼階段
    cudaEventRecord(start);

    // Entropy encoding (kept on CPU)
    short prev_DC_Y = 0, prev_DC_Cb = 0, prev_DC_Cr = 0;
    int newByte = 0, newBytePos = 7;

    // Process each 8x8 block
    for(int yPos = 0; yPos < m_height; yPos += 8) {
        for(int xPos = 0; xPos < m_width; xPos += 8) {
			int block_idx = (yPos/8 * (m_width/8) + xPos/8) * 64;
            BitString outputBitString[128];
            int bitStringCounts;

            // Y channel Huffman encoding
            _doHuffmanEncoding(
                h_y_dct + block_idx,
                prev_DC_Y,
                m_Y_DC_Huffman_Table,
                m_Y_AC_Huffman_Table,
                outputBitString,
                bitStringCounts
            );
            _write_bitstring_(outputBitString, bitStringCounts, newByte, newBytePos, fp);

            // Cb channel Huffman encoding
            _doHuffmanEncoding(
                h_cb_dct + block_idx,
                prev_DC_Cb,
                m_CbCr_DC_Huffman_Table,
                m_CbCr_AC_Huffman_Table,
                outputBitString,
                bitStringCounts
            );
            _write_bitstring_(outputBitString, bitStringCounts, newByte, newBytePos, fp);

            // Cr channel Huffman encoding
            _doHuffmanEncoding(
                h_cr_dct + block_idx,
                prev_DC_Cr,
                m_CbCr_DC_Huffman_Table,
                m_CbCr_AC_Huffman_Table,
                outputBitString,
                bitStringCounts
            );
            _write_bitstring_(outputBitString, bitStringCounts, newByte, newBytePos, fp);
        }
    }

    // 記錄 Huffman 編碼時間
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float huffman_time;
    cudaEventElapsedTime(&huffman_time, start, stop);
    printf("Huffman encoding time: %.2f ms\n", huffman_time);
    encode_time += huffman_time;

    // Cleanup
    if (newBytePos != 7) {
        _write_byte_(newByte, fp);
    }
    _write_word_(0xFFD9, fp);
    
    // Free memory
    delete[] h_y_dct;
    delete[] h_cb_dct;
    delete[] h_cr_dct;
    
    cudaFree(d_rgb);
    cudaFree(d_y);
    cudaFree(d_cb);
    cudaFree(d_cr);
    cudaFree(d_y_dct);
    cudaFree(d_cb_dct);
    cudaFree(d_cr_dct);
    
    // 清理 CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Total execution time: %.2f ms\n", encode_time);

    fclose(fp);
    return true;
}