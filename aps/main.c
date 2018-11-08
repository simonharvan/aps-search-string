//
// File:        main.c
//
// Abstract:    Simple searching string in large text files. User can enter file, pattern he is searching.
//              Program uses Knuth-Morris-Pratt algorithm.
// Version:     <2.1>
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
#define OPTIMAL_NUMBER_OF_THREADS 2048
#define MIN_PART_SIZE_FACTOR 4
#define NEWLINE '\n'

/**
 Function for calculating prefix of pattern we are trying to find with KMP algorithm. Parameters are pattern and pattern length.
 */
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

/**
 Function returns what line character is on. It takes:
 newLines       - array of longs, which holds offsets of new line characters (\n).
 max            - size of newLines array
 charNum        - offset of character we want to find
 bonds          - array of two longs, where we can store bonds of line
 */
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


/**
 Function prints out results from both CPU singlethreaded or multithreaded calculation.
 text_source        - is haystack, where we want to find pattern (needle)
 text_source_size   - length of text_source
 result             - array of unsigned longs with results, they might be in uninterupted array or they might be splitted into more arrays, but then they have to have -1 at the end of each one.
 resultSize         - total length of result array
 partSize           - size of one part in result array
 linesOption        - type of output true for printing out lines
 offsetOption       - type of output true for printing out offset
 */
void printResult(char* text_source, unsigned long text_source_size, unsigned long *result, unsigned long resultSize, unsigned long partSize, int linesOption, int offsetOption) {
    
    
    bool firsPart = true;
    int numberOfFinds = 0;
    
    
    if (linesOption || offsetOption){
        unsigned long *newLines = (unsigned long*)malloc(text_source_size * sizeof(unsigned long));
        //set first line to 0
        newLines[1] = 0;
        
        //counter begins with 2, because of finding first linebreak is actually linebreak to 2nd line
        int counter = 2;
        
        for(unsigned long j = 0; j < text_source_size; j++) {
            if (text_source[j] == NEWLINE) {
                newLines[counter] = j;
                counter++;
            }
        }
        //set last line to end of file
        newLines[counter] = text_source_size;
        
        
        unsigned long lineNumber;
        unsigned long lineBonds[] = {0, 0};
        
        
        for (unsigned long i = 0; i < resultSize; i++) {
            //  when result is -1 we can jump to next part. when result is 0 and it is not in the first part, it is also jumping as a fail safe, because of last part, when its not divisible by partSize
            if (result[i] == -1 || (result[i] == 0 && !firsPart)){
                i = (i / partSize + 1) * partSize - 1;
                firsPart = false;
                continue;
            }
            numberOfFinds++;
            if (offsetOption) {
                printf("Offset %lu\n", result[i] + 1);
            }
            if (!linesOption){
                continue;
            }
            if (result[i] < lineBonds[0] || result[i] > lineBonds[1]){
                lineNumber = findWhatLine(newLines, counter, result[i], lineBonds);
                printf("Line %lu:", lineNumber);
                printf ("%.*s\n", lineBonds[1] - lineBonds[0], &(text_source[lineBonds[0]]));
            }
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

/**
 Function for searching string in string using KMP algorithm. Returns all occurances and their offset from beginning.
 text_source        - haystack array of characters.
 text_source_size   - text_source length
 pattern            - needle that we are trying to find
 results            - array of longs, where we are setting results
 */
void findStringSingleThread(char* text_source,
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


/**
 Function for searching string in string using KMP algorithm using OpenCL multithreading. Returns all occurances and their offset from beginning.
 device_id          - cl_device_id returned from OpenCL API call
 context            - cl_command_queue queue into which program is set
 program            - cl_program built program
 kernel             - cl_kernel the compute kernel in the program we wish to run
 text_source        - haystack array of characters
 text_source_size   - length of text_source
 pattern            - needle that we are trying to find array of char
 results            - array of longs, where we are setting results
 resultsSize        - length of results array
 partSize           - pointer to unsigned long variable, where we set size of one part
 */
size_t findStringMultiThread(cl_device_id device_id,
                cl_context context,
                cl_command_queue commands,
                cl_program program,
                cl_kernel kernel,
                char* text_source,
                unsigned long text_source_size,
                char* pattern,
                unsigned long results[],
                unsigned long resultSize,
                unsigned long *partSize) {
   
    int err;                            // error code returned from api calls
    
    
    cl_mem input;                       // device memory used for the input array
    cl_mem patternMem;                  // device memory used for the pattern array
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
    
    
    // Compute number of workers (threads) for this text file
    unsigned long tmp = round(text_source_size / OPTIMAL_NUMBER_OF_THREADS) + 1;
    size_t global = text_source_size / tmp;
    
    
    if ((text_source_size / (pattern_size * MIN_PART_SIZE_FACTOR)) < global) {
        global = (text_source_size / (pattern_size * MIN_PART_SIZE_FACTOR)) ;
    }
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

int main(int argc, char** argv)
{
    
    // source string of kernel
    const char* source_str =
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
    int err;
    
    
    char *textmemblock;
    
    char textFileName[255];
    
    int fd;
    struct stat sbuf;

    char pattern[255];
    
    size_t local = 0;                   // local domain size for our calculation
    cl_uint globalSize;
    
    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    
    
    
    
    
    
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
    
    unsigned long resultSize = text_source_size;
    
    unsigned long *results = calloc(resultSize, sizeof(unsigned long));

    unsigned long partSize;
    
    size_t numOfThreads = 1;
    //Original
    if (!multithreading) {
        findStringSingleThread(textmemblock, text_source_size, pattern, results);
        partSize = text_source_size;
    } else {
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
        
        
        numOfThreads = findStringMultiThread(device_id, context, commands, program, kernel, textmemblock, text_source_size, pattern, results, resultSize, &partSize);
        
        clReleaseProgram(program);
        clReleaseKernel(kernel);
        clReleaseCommandQueue(commands);
        clReleaseContext(context);
    }
    

    printResult(textmemblock, text_source_size, results, resultSize, partSize, linesOption, offsetOption);

    if (debugOption) {
        printf("\n-------------\n");
        printf("Input size - %luB\n", text_source_size);
        printf("Number of threads - %lu\n-------------\n", numOfThreads);
    }
    
    
    // Shutdown and cleanup
    //
    
    return 0;
}
