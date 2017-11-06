//
//  string-search.cl
//  hello
//
//  Created by Simon Harvan on 4.11.17.
//
// Simple compute kernel which computes the square of an input array
//


__global void kmp(__global char *target, int tsize, __global char* pattern, __global int *pi, unsigned long psize, __global int *output, int threadId)
{
    int i;
    int k = -1;
    int counter = threadId;
    if (!pi){
        output[counter] =  0;
        return;
    }
    for (i = 0; i < tsize; i++) {
        while (k > -1 && pattern[k+1] != target[i])
            k = pi[k];
        if (target[i] == pattern[k+1])
            k++;
        if (k == psize - 1) {
            output[counter] = i - k - 1;
            counter++;
        }
    }
    output[counter] =  0;
    counter++;
    return;
}

__kernel void run(__global char* input, __global int *output, __global char* pattern, __global int* pi, const unsigned long psize, const unsigned long inputSize, const unsigned long resultCount)
{
    
    int threadId = get_global_id(0);
    int partSize = inputSize / resultCount;
    if (partSize < psize) {
        partSize = psize;
    }
    // KNUTH–MORRIS–PRATT
    if (threadId * partSize >= inputSize) {
        return;
    }
    
    if (((threadId) * partSize) + psize - 1 > inputSize) {
        kmp(input + (threadId * partSize), partSize, pattern, pi, psize, output, threadId);
    }else {
        kmp(input + (threadId * partSize), partSize + psize - 1, pattern, pi, psize, output, threadId);
    }
}





