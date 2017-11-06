//
// File:       main.c
//
// Abstract:   A simple "Hello World" compute example showing basic usage of OpenCL which
//             calculates the mathematical square (X[i] = pow(X[i],2)) for a buffer of
//             floating point values.
//
//
// Version:    <1.0>
//
// Copyright ( C ) 2008 Apple Inc. All Rights Reserved.
//

////////////////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <OpenCL/opencl.h>

////////////////////////////////////////////////////////////////////////////////

// Use a static data size for simplicity
//
#define MAX_SOURCE_SIZE (0x100000)
#define MAX_TEXT_SOURCE_SIZE (0x10000000)




////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////

int *compute_prefix_function(char *pattern, unsigned long psize)
{
    int k = -1;
    int i = 1;
    int *pi = malloc(sizeof(int)*psize);
    if (!pi)
        return NULL;
    
    pi[0] = k;
    for (i = 1; i < psize; i++) {
        while (k > -1 && pattern[k+1] != pattern[i])
            k = pi[k];
        if (pattern[i] == pattern[k+1])
            k++;
        pi[i] = k;
    }
    return pi;
}

unsigned long findWhatLine(long *newLines, int max, unsigned long charNum) {
    for (int i = 1; i < max-1; i++) {
        if (*(newLines + i) < charNum && charNum < *(newLines + i + 1)){
            return i;
        }
    }
    return max;
}

void printd2Darray(int *array, int rowSize, int columnSize) {
    for (int i = 0; i < columnSize; i++) {
        for (int j = 0; j < rowSize; j++) {
            printf("%d ", *(array + (i * columnSize) + j));
        }
        printf("\n");
    }
}

void printResult(char* text_source, unsigned long text_source_size, int *result, unsigned long count, size_t local, unsigned long psize) {
    int numberOfFinds = 0;
    long *newLines = (long*)malloc(text_source_size * sizeof(int));
    
    int counter = 2;
    for(int j = 0; j < text_source_size; j++) {
        if (text_source[j] == '\n') {
            newLines[counter] = j;
            counter++;
        }
    }
    
    unsigned long partSize = count / local;
    if (partSize < psize) {
        partSize = psize;
    }
//    printd2Darray(result, local, partSize);
    for (int i = 0; i < local * partSize; i++) {
        if (result[i] != 0){
            numberOfFinds++;
            int threadId = floor(i / partSize);
            printf("Find match on char %lu\n", result[i] + 1 + (threadId * partSize));
            
//            printf("Find match on line %lu\n", findWhatLine(newLines, counter, result[i] + 1 + (threadId * partSize)));
        }
    }
    
    if (numberOfFinds > 0) {
        printf("\n-------------\nNumber of matches: %d\n", numberOfFinds);
    }else {
        printf("No match in file\n");
    }
}



int main(int argc, char** argv)
{
    int err;                            // error code returned from api calls
    
    size_t local;                       // local domain size for our calculation
    
    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    
    cl_mem input;                       // device memory used for the input array
    cl_mem patternMem;                       // device memory used for the input array
    cl_mem computePatternMem;
    cl_mem output;                      // device memory used for the output array
    
    
    FILE *fp;
    FILE *textfp;
    char fileName[] = "/Users/simonharvan/Documents/Development/C/aps/aps/main.cl";
    char textFileName[255];
//  /Users/simonharvan/Documents/Development/C/aps/aps/small.txt
    char pattern[255];
    
    
    char *source_str;
    size_t source_size;
    
    
    char *text_source_str;
    size_t text_source_size;
    
    printf("---------------\nHello, this is program for searching in text files\n---------------\nLoad file:\n");
    scanf("%s", textFileName);
    
    /* Load text file */
    textfp = fopen(textFileName, "r");
    if (!textfp) {
        fprintf(stderr, "Failed to load text file.\n");
        exit(1);
    }
    text_source_str = (char*)malloc(MAX_TEXT_SOURCE_SIZE);
    text_source_size = fread(text_source_str, 1, MAX_TEXT_SOURCE_SIZE, textfp);
    
    /* Load the source code containing the kernel*/
    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    
    printf("Pattern you want to find: \n");
    scanf("%s", pattern);
    
    int gpu = 1;
    printf("Compute on:\n0 - CPU\n1 - GPU\n");
    scanf("%d", &gpu);
    
    //Start time
    clock_t begin = clock();
    
    // Connect to a compute device
    //
    
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
    
    
    
    // Create a compute context
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }
    
    // Create a command commands
    //
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }
    
    // Create the compute program from the source buffer
    //
    program = clCreateProgramWithSource(context, 1, (const char **) & source_str, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }
    
    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }
    
    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, "run", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }
    
    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }
    
    unsigned long count = strlen(text_source_str);
    
    
    if (count == 0) {
        printf("Error: Text file is empty!");
        exit(1);
    }
    
    unsigned long psize = strlen(pattern);
    
    if (psize == 0) {
        printf("Error: Pattern is empty!");
        exit(1);
    }
    
    if (psize > count) {
        printf("Error: Pattern is longer than text in text file!");
        exit(1);
    }
    
    unsigned long resultCount = local;
    unsigned long partSize = count / resultCount;
    
    
    if (partSize < psize) {
        partSize = psize;
    }
    int results[local * partSize];
    
    // Create the input and output arrays in device memory for our calculation
    //
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(char) * count, NULL, NULL);
    patternMem = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(char) * strlen(pattern), NULL, NULL);
    computePatternMem = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(int) * strlen(pattern), NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * local * partSize, NULL, NULL);
    
    if (!input || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }
    
    // Write our data set into the input array in device memory
    //
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(char) * count, text_source_str, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }
    
    err = clEnqueueWriteBuffer(commands, patternMem, CL_TRUE, 0, sizeof(char) * psize, pattern, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source pattern!\n");
        exit(1);
    }
    
    int *pi = compute_prefix_function(pattern, psize);
    err = clEnqueueWriteBuffer(commands, computePatternMem, CL_TRUE, 0, sizeof(char) * psize, pi, 0, NULL, NULL);
    
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source pattern!\n");
        exit(1);
    }
    
    
    
    
    // Set the arguments to our compute kernel
    //
    
    
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &patternMem);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &computePatternMem);
    err |= clSetKernelArg(kernel, 4, sizeof(unsigned long), &psize);
    err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), &count);
    err |= clSetKernelArg(kernel, 6, sizeof(unsigned int), &local);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
    
    
    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &local, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }
    
    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands);
   
    
    
    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(int) * local * partSize, results, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
    
    //End time
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("\n-------------\nDuration - %fs\n", time_spent);
    printf("GPU - %d\n", gpu);
    printf("Input size - %luB\n", count);
    printf("Number of threads - %lu\n-------------\n", local);
    
    printResult(text_source_str, text_source_size, results, count, local, psize);
    
    // Shutdown and cleanup
    //
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseMemObject(patternMem);
    clReleaseMemObject(computePatternMem);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    
    return 0;
}


