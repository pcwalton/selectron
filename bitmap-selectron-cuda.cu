// CUDA Bitmap Selectron prototype
//
// Patrick Walton <pcwalton@mozilla.com>
//
// Copyright (c) 2015 Mozilla Corporation

#include "bitmap-selectron.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define THREAD_COUNT        256

__device__ unsigned hash_device(unsigned key, unsigned seed) {
    unsigned int hash = 2166136261;
    hash = hash ^ seed;
    hash = hash * 16777619;
    hash = hash ^ key;
    hash = hash * 16777619;
    return hash;
}

__global__ void match_selectors_device(const int *ids,
                                       const uint16_t *stylesheet,
                                       uint16_t *matched_rules) {
    int node = blockIdx.x * THREAD_COUNT + threadIdx.x;

    matched_rules[node] = stylesheet[hash_device(ids[node], 12345) % MAX_ID];
}

int get_cuda_device(bool cpu) {
    int device_count;
    checkCudaErrors(cudaGetDeviceCount(&device_count));

    for (int device = 0; device < device_count; device++) {
        cudaDeviceProp device_props;
        cudaGetDeviceProperties(&device_props, device);
        if (device_props.computeMode == cudaComputeModeProhibited)
            continue;
        fprintf(stderr, "found device: %s\n", device_props.name);
        return device;
    }

    fprintf(stderr, "no device found\n");
    return 1;
}

int main(int argc, char **argv) {
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int device_id = get_cuda_device(false);
    checkCudaErrors(cudaSetDevice(device_id));

    srand(time(NULL));

    // Create the stylesheet on the host.
    uint16_t *host_stylesheet = (uint16_t *)malloc(MAX_ID);
    for (int i = 0; i < MAX_ID; i++)
        host_stylesheet[i] = rand();

    // Create the DOM on the host.
    int *host_ids = (int *)malloc(sizeof(int) * NODE_COUNT);
    create_dom(host_ids);

    // Allocate the IDs and copy over.
    int *device_ids;
    checkCudaErrors(cudaMalloc((void **)&device_ids, sizeof(int) * NODE_COUNT));
    checkCudaErrors(cudaMemcpy(device_ids,
                               host_ids,
                               sizeof(int) * NODE_COUNT,
                               cudaMemcpyHostToDevice));

    // Allocate the stylesheet and copy over.
    uint16_t *device_stylesheet;
    checkCudaErrors(cudaMalloc((void **)&device_stylesheet, sizeof(uint16_t) * MAX_ID));
    checkCudaErrors(cudaMemcpy(device_stylesheet,
                               host_stylesheet,
                               sizeof(uint16_t) * MAX_ID,
                               cudaMemcpyHostToDevice));
    
    // Allocate the matched rules.
    uint16_t *device_matched_rules;
    uint16_t *device_matched_rules_host_mirror = (uint16_t *)malloc(sizeof(uint16_t) * NODE_COUNT);
    checkCudaErrors(cudaMalloc((uint16_t **)&device_matched_rules, sizeof(uint16_t) * NODE_COUNT));

    // Create start/stop events.
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Execute the kernel on the GPU.
    checkCudaErrors(cudaEventRecord(start));
    match_selectors_device<<<NODE_COUNT / THREAD_COUNT, THREAD_COUNT>>>(device_ids,
                                                                        device_stylesheet,
                                                                        device_matched_rules);
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    float gpu_elapsed = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&gpu_elapsed, start, stop));
    fprintf(stderr, "Elapsed time: %f ms\n", gpu_elapsed);

    for (int i = 0; i < 

    checkCudaErrors(cudaFree(device_matched_rules));
    checkCudaErrors(cudaFree(device_stylesheet));
    checkCudaErrors(cudaFree(device_ids));

    cudaDeviceReset();
    return 0;
}

