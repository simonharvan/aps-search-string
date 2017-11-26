//
//  string-search.cl
//  hello
//
//  Created by Simon Harvan on 4.11.17.
//
// Simple compute kernel which computes kmp of string given in input
//

// KNUTH–MORRIS–PRATT
__kernel void kmp(__global char *target, unsigned long tsize, __global char* pattern, __global int *pi, unsigned long psize, __global unsigned long *output, volatile __global int* counter)
{
    
    
    int i;
    int k = -1;
    if (!pi){
        output[*counter] =  0;
        return;
    }
    
    for (i = 0; i < tsize; i++) {
        while (k > -1 && pattern[k+1] != target[i])
            k = pi[k];
        if (target[i] == pattern[k+1])
            k++;
        if (k == psize - 1) {
            output[*counter] = i - k - 1;
            printf("%d at position %d\n", i - k - 1, *counter);
            atomic_inc(counter);
            k = -1;
        }
    }
    return;
}

__kernel void run(__global char* input, __global unsigned long* output, __global char* pattern, __global int* pi,  unsigned long psize, unsigned long inputSize, size_t numOfThreads, __global int *counterArg)
{
    
    int threadId = get_global_id(0);
    unsigned long partSize = inputSize / numOfThreads;
    if (partSize < psize) {
        partSize = psize;
    }

    volatile __global int *counter = counterArg;

    if (threadId * partSize >= inputSize) {
        return;
    }

    int index = threadId * partSize;
    
    if (index + psize - 1 > inputSize) {
        kmp(input + (index), partSize, pattern, pi, psize, output, counter);
    }else {
        kmp(input + (index), partSize + psize - 1, pattern, pi, psize, output, counter);
    }
}
