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
#include <sys/mman.h>
#include <time.h>
#include <OpenCL/opencl.h>

////////////////////////////////////////////////////////////////////////////////

// Use a static data size for simplicity
//
#define MAX_SOURCE_SIZE (0x100000)

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

unsigned long findWhatLine(unsigned long *newLines, int max, unsigned long charNum, unsigned long *bonds) {
    for (int i = 1; i < max-1; i++) {
        if (*(newLines + i) <= charNum && charNum < *(newLines + i + 1)){
            bonds[0] = *(newLines + i);
            bonds[1] = *(newLines + i + 1);
            return i;
        }
    }
    bonds[0] = *(newLines + max - 1);
    bonds[1] = *(newLines + max);
    return max - 1;
}

char *substring(char *string, unsigned long position, unsigned long length)
{
    char *pointer;
    int c;
    
    pointer = (char*) malloc(length+1 * sizeof(char));
    
    if (pointer == NULL)
    {
        printf("Unable to allocate memory.\n");
        exit(1);
    }
    
    for (c = 0 ; c < length ; c++)
    {
        *(pointer+c) = *(string+position-1);
        string++;
    }
    
    *(pointer+c) = '\0';
    
    return pointer;
}

void printResult(char* text_source, unsigned long text_source_size, unsigned long *result, unsigned long resultSize, size_t local, unsigned long psize, unsigned long partSize, int linesOption, int offsetOption) {
    bool firsPart = true;
    int numberOfFinds = 0;
    if (linesOption){
        unsigned long *newLines = (unsigned long*)malloc(text_source_size * sizeof(unsigned long));
        newLines[1] = 0;
        int counter = 2;
        for(unsigned long j = 0; j < text_source_size; j++) {
            if (text_source[j] == '\n') {
                newLines[counter] = j;
                counter++;
            }
        }
        newLines[counter] = text_source_size;
        unsigned long lineNumber;
        unsigned long lastLineNumber = 0;
        unsigned long *lineBonds = (unsigned long*) malloc(2 * sizeof(unsigned long));
        for (unsigned long i = 0; i < resultSize; i++) {
            if (result[i] == -1 || (result[i] == 0 && !firsPart)){
                i = (i / partSize + 1) * partSize - 1;
                firsPart = false;
                continue;
            }
            numberOfFinds++;
            lineNumber = findWhatLine(newLines, counter, result[i], lineBonds);
            if (lineNumber > lastLineNumber){
                printf("Find match on line %lu\n", lineNumber);
                printf("%s\n", substring(text_source, lineBonds[0] + 1, lineBonds[1] - lineBonds[0]));
            }
            lastLineNumber = lineNumber;
            
        }
    } else if(offsetOption) {
        for (unsigned long i = 0; i < resultSize; i++) {
            if (result[i] == -1 || (result[i] == 0 && !firsPart)){
                i = (i / partSize + 1) * partSize - 1;
                firsPart = false;
                continue;
            }
            numberOfFinds++;
            printf("Find match with offset %lu\n", result[i] + 1);
        }
    } else {
        for (unsigned long i = 0; i < resultSize; i++) {
            if (result[i] == -1 || (result[i] == 0 && !firsPart)){
                i = (i / partSize + 1) * partSize - 1;
                firsPart = false;
                continue;
            }
            numberOfFinds++;
        }
    }
    
    
    
    
    if (numberOfFinds > 0) {
        printf("\nNumber of matches: %d\n", numberOfFinds);
    }else {
        printf("No match in file\n");
    }
}

void findStringCPU(char* text_source,
                   unsigned long text_source_size,
                   char* pattern,
                   unsigned long results[]) {
    int i;
    int k = -1;
    int counter = 0;
    unsigned long pattern_size = strlen(pattern);
    int *pi = compute_prefix_function(pattern, pattern_size);
    
    if (!pi){
        return;
    }
    
    for (i = 0; i < text_source_size; i++) {
        while (k > -1 && pattern[k+1] != text_source[i])
            k = pi[k];
        if (text_source[i] == pattern[k+1])
            k++;
        if (k == pattern_size - 1) {
            results[counter] = i - k;
            counter++;
            k = -1;
        }
    }
    results[counter] = -1;

    return;
    

}

size_t findStringGPU(cl_device_id device_id,
                cl_context context,
                cl_command_queue commands,
                cl_program program,
                cl_kernel kernel,
                size_t local,         // local domain size for our calculation
                cl_uint globalSize,
                size_t maxWorkGroupSize,
                char* text_source,
                unsigned long text_source_size,
                char* pattern,
                unsigned long results[],
                unsigned long resultSize,
                unsigned long *partSize) {
   
    int err;                            // error code returned from api calls
    
    
    cl_mem input;                       // device memory used for the input array
    cl_mem patternMem;                       // device memory used for the input array
    cl_mem computePatternMem;
    cl_mem output;// device memory used for the output array
    
    
    

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
    int error = 0;
    
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,  sizeof(char) * text_source_size, text_source, &error);
    if (error)
    {
        printf("Error: Failed to allocate device memory with code %d!\n", error);
        exit(1);
    }
    
    patternMem = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,  sizeof(char) * pattern_size, pattern, &error);
    if (error)
    {
        printf("Error: Failed to allocate device memory with code %d!\n", error);
        exit(1);
    }
    int *pi = compute_prefix_function(pattern, pattern_size);
    computePatternMem = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,  sizeof(int) * pattern_size, pi, &error);
    if (error)
    {
        printf("Error: Failed to allocate device memory with code %d!\n", error);
        exit(1);
    }
    
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, resultSize * sizeof(long), results, &error);
    if (error)
    {
        printf("Error: Failed to allocate device memory with code %d!\n", error);
        exit(1);
    }
    
    results = clEnqueueMapBuffer(commands, output, CL_TRUE, CL_MAP_WRITE, 0, resultSize * sizeof(unsigned long), 0, NULL, NULL, NULL);
    
    
    // Set the arguments to our compute kernel
    //
    
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &patternMem);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &computePatternMem);
    err |= clSetKernelArg(kernel, 4, sizeof(unsigned long), &pattern_size);
    err |= clSetKernelArg(kernel, 5, sizeof(unsigned long), &text_source_size);
    
    
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
    
    unsigned long tmp = text_source_size / 1024;
    size_t global = text_source_size / tmp;
    if ((text_source_size / (pattern_size * 4)) < global) {
        global = (text_source_size / (pattern_size * 4)) ;
    }
//    size_t global = (text_source_size / (pattern_size * 4)) ;
    *partSize = text_source_size / global;
    
    // Execute the kernel over the entire range of our 1d input data set
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    
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
    err = clEnqueueUnmapMemObject(commands, output, results, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
    
    
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseMemObject(patternMem);
    clReleaseMemObject(computePatternMem);
    return global;
}


void printArray(unsigned long *array, unsigned long arraySize) {
    for (int i = 0; i < arraySize; i++) {
        printf("%lu - %d\n", array[i], i);
    }
}


int main(int argc, char** argv)
{
    const char* kernelCode =
    "//"
    "//  string-search.cl"
    "//  hello"
    "//"
    "//  Created by Simon Harvan on 4.11.17."
    "//"
    "// Simple compute kernel which computes kmp of string given in input"
    "//"
    
    "// KNUTH–MORRIS–PRATT"
    "__kernel void kmp(__global char *target, unsigned long tsize, __global char* pattern, __global int *pi, unsigned long psize, __global unsigned long *output, unsigned long counter)"
    "{"
    
    
    "    unsigned long i;"
    "    int k = -1;"
    "    if (!pi){"
    "        return;"
    "    }"
    "    unsigned long index = counter;"
    
    
    "    for (i = 0; i < tsize; i++) {"
    "        while (k > -1 && pattern[k+1] != target[i])"
    "            k = pi[k];"
    "        if (target[i] == pattern[k+1])"
    "            k++;"
    "        if (k == psize - 1) {"
    "            output[counter] = index + i - k;"
    "            counter++;"
    "            k = -1;"
    
    "        }"
    "    }"
    "    output[counter] = -1;"
    "    return;"
    "}"
    
    "__kernel void run(__global char* input, __global unsigned long* output, __global char* pattern, __global int* pi,  unsigned long psize, unsigned long inputSize)"
    "{"
    "    int threadId = get_global_id(0);"
    "    unsigned long partSize = inputSize / get_global_size(0);"
    "    if (partSize < psize) {"
    "        partSize = psize;"
    "    }"
    
    "    if (threadId * partSize >= inputSize) {"
    "        return;"
    "    }"
    
    "    unsigned long index = threadId * partSize;"
    
    "    if (index + psize - 1 > inputSize) {"
    "        kmp(input + (index), partSize, pattern, pi, psize, output, index);"
    "    }else {"
    "        kmp(input + (index), partSize + psize - 1, pattern, pi, psize, output, index);"
    "    }"
    "}";
    char fileName[] = "/Users/simonharvan/Documents/Development/C/aps-search-string-2/aps/main.cl";
    int err;
    
    FILE *fp;
    char *textmemblock;
    
    char textFileName[255];
    
    int fd;
    struct stat sbuf;

    char pattern[255];
    
    size_t local = 0;                       // local domain size for our calculation
    cl_uint globalSize;
    
    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    
    
    
    char *source_str;
    size_t source_size;
    
    
    size_t text_source_size;
    int multithreading = 0;
    int linesOption = 0;
    int offsetOption = 0;
    int fileLoaded = 0;
    int patternLoaded = 0;
    int debugOption = 0;
    int i = 1;
    
    while (i < argc) {
        if (!strcmp(argv[i], "-p")) {
            strcpy(pattern, argv[i+1]);
            i += 2;
            patternLoaded = 1;
            continue;
        }else if (!strcmp(argv[i], "-f")) {
            strcpy(textFileName, argv[i+1]);
            i += 2;
            fileLoaded = 1;
            continue;
        }else if (!strcmp(argv[i], "-h")) {
            printf("aps [-tlod] [-p pattern] [-f file]\n\t-t\tmultithreading \n\t-l\touputs number of line and line itself\n\t-o\toutputs offset in bytes\n\t-d\touputs debug at the end\n\t-p\tpattern\n\t-f\tfile\n");
            return EXIT_SUCCESS;
        }else if (!strcmp(argv[i], "-t")) {
            multithreading = 1;
            i++;
            continue;
        }else if (!strcmp(argv[i], "-l")) {
            linesOption = 1;
            i++;
            continue;
        }else if (!strcmp(argv[i], "-o")) {
            offsetOption = 1;
            i++;
            continue;
        }else if (!strcmp(argv[i], "-d")){
            debugOption = 1;
            i++;
            continue;
        }else {
            printf("wrong argument %s\n", argv[i]);
            printf("aps [-tlod] [-p pattern] [-f file]\nNAME\naps \nDESCRIPTION\nFile pattern searcher. Utility searches any given input files, selecting lines that match one patterns.\n");
            return EXIT_FAILURE;
        }
    }
    
    if (!fileLoaded) {
        printf("no file defined!\n");
        return EXIT_FAILURE;
    }
    if (!patternLoaded) {
        printf("no pattern defined!\n");
        return EXIT_FAILURE;
    }
    
    if ((fd = open(textFileName, O_RDONLY)) == -1) {
        fprintf(stderr, "Error opening file\n");
        exit(EXIT_FAILURE);
    }
    
    if (stat(textFileName, &sbuf) == -1) {
        fprintf(stderr, "Stat error\n");
        exit(EXIT_FAILURE);
    }
    printf("Proccessing ...\n");
    

    
    
    /* Load text file */
    textmemblock = mmap(NULL, sbuf.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (textmemblock == (caddr_t)(-1)) {
        fprintf(stderr, "Failed to load text file.\n");
        exit(1);
    }

    text_source_size = sbuf.st_size;
    
    unsigned long psize = strlen(pattern);
   
    unsigned long resultSize = text_source_size;
    
    unsigned long *results = calloc(resultSize, sizeof(unsigned long));

    unsigned long partSize;
    
    size_t numOfThreads = 1;
    //Original
    if (!multithreading) {
        findStringCPU(textmemblock, text_source_size, pattern, results);
        partSize = text_source_size;
    } else {
        
        
        
        /* Load the source code containing the kernel*/
        fp = fopen(fileName, "r");
        if (!fp) {
            fprintf(stderr, "Failed to load kernel.\n");
            exit(1);
        }
        source_str = (char*)malloc(MAX_SOURCE_SIZE);
        source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
        
        
        // Connect to a compute device
        //
        err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
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
        
        err = clGetDeviceInfo(device_id, CL_DEVICE_ADDRESS_BITS, sizeof(globalSize), &globalSize, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to retrieve device info! %d\n", err);
            exit(1);
        }
        
        size_t maxWorkGroupSize;
        err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to retrieve device info! %d\n", err);
            exit(1);
        }
        
        
        cl_ulong maxMemAlloc;
        err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxMemAlloc), &maxMemAlloc, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to retrieve device info! %d\n", err);
            exit(1);
        }
        
        //If device cannot hold big enough array program ends
        //
        if (maxMemAlloc < text_source_size) {
            printf("Error: Too big file");
            exit(EXIT_FAILURE);
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
        
        
        numOfThreads = findStringGPU(device_id, context, commands, program, kernel, local, globalSize, maxWorkGroupSize, textmemblock, text_source_size, pattern, results, resultSize, &partSize);
        
        clReleaseProgram(program);
        clReleaseKernel(kernel);
        clReleaseCommandQueue(commands);
        clReleaseContext(context);
    }
    

    printResult(textmemblock, text_source_size, results, resultSize, local, psize, partSize, linesOption, offsetOption);

    if (debugOption) {
        printf("\n-------------\n");
        printf("Input size - %luB\n", text_source_size);
        printf("Number of threads - %lu\n-------------\n", numOfThreads);
    }
    
    
    // Shutdown and cleanup
    //
    
    return 0;
}

///Users/simonharvan/Documents/Development/C/aps/aps/tex.txt
