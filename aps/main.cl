//
//  string-search.cl
//  hello
//
//  Created by Simon Harvan on 4.11.17.
//
// Simple compute kernel which computes the square of an input array
//

// KNUTH–MORRIS–PRATT
__global int kmp(__global char *target, int tsize, __global char* pattern, __global int *pi, unsigned long psize)
{
    int i;
    int k = -1;
    if (!pi)
        return 0;
    for (i = 0; i < tsize; i++) {
        while (k > -1 && pattern[k+1] != target[i])
            k = pi[k];
        if (target[i] == pattern[k+1])
            k++;
        if (k == psize - 1) {
            return i-k - 1;
        }
    }
    return 0;
}

__kernel void run(__global char* input, __global int* output, __global char* pattern, __global int* pi, const unsigned long psize, const unsigned long inputSize, const unsigned long resultCount)
{
    
    int threadId = get_global_id(0);
    int partSize = inputSize / resultCount;
    if (partSize < psize) {
        partSize = psize;
    }
    
    
    
    
    if (threadId * partSize >= inputSize) {
        return;
    }
    
    
    if (((threadId) * partSize) + psize - 1 > inputSize) {
        output[threadId] = kmp(input + (threadId * partSize), partSize, pattern, pi, psize);
    }else {
        output[threadId] = kmp(input + (threadId * partSize), partSize + psize - 1, pattern, pi, psize);
    }
}





