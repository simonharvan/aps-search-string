//
// File:       main.c
//
// Abstract:   Simple searching string in large text files. User can choose whether he wants to execute on GPU or CPU. User can enter file, pattern he is searching and choose desired device for execution.
// Version:    <1.0>
//
// Copyright ( C ) 2017 Simon Harvan. All Rights Reserved.
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
#define MAX_TEXT_SOURCE_SIZE (0x100000000)




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

void printAdvResult(char* text_source, unsigned long text_source_size, int *result,int *resultUpper, int *resultLower,  size_t local, unsigned long psize) {
    int numberOfFinds = 0;
    long *newLines = (long*)malloc(text_source_size * sizeof(int));
    
    int counter = 2;
    for(int j = 0; j < text_source_size; j++) {
        if (text_source[j] == '\n') {
            newLines[counter] = j;
            counter++;
        }
    }
    
    unsigned long partSize = text_source_size / local;
    if (partSize < psize) {
        partSize = psize;
    }
    for (int i = 0; i < local; i++) {
        if (result[i] != 0){
            numberOfFinds++;
            //            printf("Find match on char %lu\n", result[i] + 1 + (i * partSize));
            printf("Find match on line %lu\n", findWhatLine(newLines, counter, result[i] + 1 + (i * partSize)));
        }
    }
    for (int i = 0; i < local; i++) {
        if (resultUpper[i] != 0){
            numberOfFinds++;
            //            printf("Find match on char %lu\n", result[i] + 1 + (i * partSize));
            printf("Find match on line %lu\n", findWhatLine(newLines, counter, resultUpper[i] + 1 + (i * partSize)));
        }
    }
    
    for (int i = 0; i < local; i++) {
        if (resultLower[i] != 0){
            numberOfFinds++;
            //            printf("Find match on char %lu\n", result[i] + 1 + (i * partSize));
            printf("Find match on line %lu\n", findWhatLine(newLines, counter, resultLower[i] + 1 + (i * partSize)));
        }
    }
    
    if (numberOfFinds > 0) {
        printf("\n-------------\nNumber of matches: %d\n", numberOfFinds);
    }else {
        printf("No match in file\n");
    }
}

void printResult(char* text_source, unsigned long text_source_size, int *result,  size_t local, unsigned long psize) {
    int numberOfFinds = 0;
    long *newLines = (long*)malloc(text_source_size * sizeof(int));
    
    int counter = 2;
    for(int j = 0; j < text_source_size; j++) {
        if (text_source[j] == '\n') {
            newLines[counter] = j;
            counter++;
        }
    }
    
    unsigned long partSize = text_source_size / local;
    if (partSize < psize) {
        partSize = psize;
    }
    for (int i = 0; i < local; i++) {
        if (result[i] != 0){
            numberOfFinds++;
//            printf("Find match on char %lu\n", result[i] + 1 + (i * partSize));
            printf("Find match on line %lu\n", findWhatLine(newLines, counter, result[i] + 1 + (i * partSize)));
        }
    }
    
    if (numberOfFinds > 0) {
        printf("\n-------------\nNumber of matches: %d\n", numberOfFinds);
    }else {
        printf("No match in file\n");
    }
}


void findString(cl_device_id device_id,
                cl_context context,
                cl_command_queue commands,
                cl_program program,
                cl_kernel kernel,
                size_t local,         // local domain size for our calculation
                char* text_source,
                unsigned long text_source_size,
                char* pattern,
                int *results) {
   
    int err;                            // error code returned from api calls
    
    
    cl_mem input;                       // device memory used for the input array
    cl_mem patternMem;                       // device memory used for the input array
    cl_mem computePatternMem;
    cl_mem output;                      // device memory used for the output array
    
    

    if (text_source_size == 0) {
        printf("Error: Text file is empty!");
        exit(1);
    }
    
    unsigned long pattern_size = strlen(pattern);
    
    if (pattern_size == 0) {
        printf("Error: Pattern is empty!");
        exit(1);
    }
    
    if (pattern_size > text_source_size) {
        printf("Error: Pattern is longer than text in text file!");
        exit(1);
    }
    
    
    // Create the input and output arrays in device memory for our calculation
    //
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(char) * text_source_size, NULL, NULL);
    patternMem = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(char) * strlen(pattern), NULL, NULL);
    computePatternMem = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(int) * strlen(pattern), NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * local, NULL, NULL);
    
    if (!input || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }
    
    // Write our data set into the input array in device memory
    //
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(char) * text_source_size, text_source, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }
    
    err = clEnqueueWriteBuffer(commands, patternMem, CL_TRUE, 0, sizeof(char) * strlen(pattern), pattern, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source pattern!\n");
        exit(1);
    }
    
    int *pi = compute_prefix_function(pattern, pattern_size);
    err = clEnqueueWriteBuffer(commands, computePatternMem, CL_TRUE, 0, sizeof(char) * strlen(pattern), pi, 0, NULL, NULL);
    
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source pattern!\n");
        exit(1);
    }
    
    
    
    
    // Set the arguments to our compute kernel
    //
    
    unsigned long resultCount = local;
    unsigned long partSize = text_source_size / resultCount;
    
    
    if (partSize < pattern_size) {
        partSize = pattern_size;
    }
    
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &patternMem);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &computePatternMem);
    err |= clSetKernelArg(kernel, 4, sizeof(unsigned long), &pattern_size);
    err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), &text_source_size);
    err |= clSetKernelArg(kernel, 6, sizeof(unsigned int), &resultCount);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
    
    
    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    size_t global = text_source_size + pattern_size;
    unsigned long temp = global / local;
    global = local * temp;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        exit(EXIT_FAILURE);
    }
    
    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands);
    
    
    
    // Read back the results from the device to verify the output
    //
    
    
    err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(int) * local, results, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
    
    
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseMemObject(patternMem);
    clReleaseMemObject(computePatternMem);
}

void strupp(char* beg)
{
    while (*beg++ = toupper(*beg));
}

void strlow(char* beg)
{
    while (*beg++ = tolower(*beg));
}


int main(int argc, char** argv)
{
   
    int err;
    
    FILE *fp;
    FILE *textfp;
    char fileName[] = "/Users/simonharvan/Documents/Development/C/aps/aps/main.cl";
    char textFileName[255];

    char pattern[255];
    
    size_t local;                       // local domain size for our calculation
    
    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    
    
    
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
        exit(EXIT_FAILURE);
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
    int results[local];
    int resultUpper[local];
    int resultLower[local];
    
    //Original
    findString(device_id, context, commands, program, kernel, local, text_source_str, text_source_size, pattern, results);
    
    //With all chars upper
    strupp(pattern);
    findString(device_id, context, commands, program, kernel, local, text_source_str, text_source_size, pattern, resultUpper);
    
    //With all chars lower
    strlow(pattern);
    findString(device_id, context, commands, program, kernel, local, text_source_str, text_source_size, pattern, resultLower);

    //End time
    clock_t end = clock();
    
    printAdvResult(text_source_str, text_source_size, results, resultUpper, resultLower, local, strlen(pattern));
    
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("\n-------------\nDuration - %fs\n", time_spent);
    printf("GPU - %d\n", gpu);
    printf("Input size - %luB\n", text_source_size);
    printf("Number of threads - %lu\n-------------\n", local);
    
    
    
    // Shutdown and cleanup
    //
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    
    return 0;
}

///Users/simonharvan/Documents/Development/C/aps/aps/text.txt
