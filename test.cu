#include <stdio.h>
#include "common/CycleTimer.h"
#include "jpeg_encoder.h"

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

    // Serial version
    double startTime = CycleTimer::currentSeconds();
    if(!encoder.encodeToJPG(outputFileName, 50)) {
        return 1;
    }
    double endTime = CycleTimer::currentSeconds();

    // CUDA version
    double startTime_cuda = CycleTimer::currentSeconds();
    if(!encoder.encodeToJPG_CUDA(outputFileName, 50)) {
        return 1;
    }
    double endTime_cuda = CycleTimer::currentSeconds();

    double speedup = (endTime - startTime) / (endTime_cuda - startTime_cuda);

    // Print the result
    printf("\n============================= Performance =========================\n");
    printf("Serial Version     : %.3f seconds\n", (endTime - startTime));  
    printf("CUDA Version       : %.3f seconds\n", (endTime_cuda - startTime_cuda));
    printf("Speedup            : %.3fx faster\n", speedup);
    printf("===================================================================\n\n");
    return 0;
}