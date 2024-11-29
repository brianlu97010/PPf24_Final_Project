#include <stdio.h>
#include "jpeg_encoder_cuda.h"

//-------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if(argc < 3)
    {
        printf("Usage: %s {inputFile} {outputFile}\n\tInput file must be 24bit bitmap file.\n", argv[0]);
        return 1;
    }

    const char* inputFileName = argv[1];
    const char* outputFileName = argv[2];

    JpegEncoder encoder;
    if(!encoder.readFromBMP(inputFileName))
    {
        return 1;
    }

    if(!encoder.encodeToJPG(outputFileName, 50))
    {
        return 1;
    }
    return 0;
}