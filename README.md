# Profiling
## Steps
1. Generate bmp file
2. run the test script
3. See the result in `result.txt`

## Generate the testing BMP file
### Requirements
```bash
pip install numpy Pillow
```

### Generate test images
```bash
python generateBMP.py
```

## (Option 1.) Run the test script on local 
### Test with default image (img/sample3.bmp)
```bash
./run_test.sh
```


### Test all images in test_images directory
```bash
./run_test.sh -a
```
## (Option 2.) Run the test script on workstation

### Use workstation to run test script
```bash
./run_test.sh -s
```
### Use srun to test all images
```bash
./run_test.sh -s -a
```
## Performance Results
> See the Expiremental Results in `result.txt`


### Testing Hardware : 
待補

### Main Results
待補
| Test Image               | (Avg.) Original Processing Time | (Avg.) CUDA Version Processing Time | SpeedUp |
|--------------------------|---------------------------------|-------------------------------------|---------|
| img/sample_5184×3456.bmp | 4.526 s                         | 0.688 s                             | 6.579x  |

### CUDA Operation
| Test Image               | (Avg.) Memory Allocation & Transfer Time | (Avg.) Kernel Execution Time |
|--------------------------|------------------------------------------|------------------------------|
| img/sample_5184×3456.bmp | 0.01697 s                                | 0.0124 s                     | 