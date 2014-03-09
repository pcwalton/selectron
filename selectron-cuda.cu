// CUDA Selectron prototype
//
// Patrick Walton <pcwalton@mozilla.com>
//
// Copyright (c) 2014 Mozilla Corporation

#include "selectron.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

__device__ uint32_t css_rule_hash_device(uint32_t key, uint32_t seed) {
    CSS_RULE_HASH(key, seed);
}

__device__ const css_rule *__restrict__ css_cuckoo_hash_find_precomputed_device(
        const css_cuckoo_hash *__restrict__ hash,
        int32_t key,
        int32_t left_index,
        int32_t right_index) {
    CSS_CUCKOO_HASH_FIND_PRECOMPUTED(hash, key, left_index, right_index);
}

__device__ const css_rule *__restrict__ css_cuckoo_hash_find_device(
        const css_cuckoo_hash *__restrict__ hash,
        int32_t key) {
    CSS_CUCKOO_HASH_FIND(hash, key, css_rule_hash_device);
}

__global__ void match_selectors_device(dom_node *first,
                                       const css_stylesheet *__restrict__ stylesheet) {
#if 0
    MATCH_SELECTORS(first,
                    stylesheet,
                    blockIdx.x * THREAD_COUNT + threadIdx.x,
                    css_cuckoo_hash_find_device);
#endif
    MATCH_SELECTORS_PRECOMPUTED(first,
                                stylesheet,
                                blockIdx.x * THREAD_COUNT + threadIdx.x,
                                css_cuckoo_hash_find_precomputed_device,
                                css_rule_hash_device);
}

// Main routine

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

    // Create the rule tree on the host.
    css_stylesheet *host_stylesheet = (css_stylesheet *)malloc(sizeof(css_stylesheet));
    create_stylesheet(host_stylesheet);

    // Create the DOM tree on the host.
    int global_count = 0;
    dom_node *host_dom = (dom_node *)malloc(sizeof(struct dom_node) * NODE_COUNT);
    create_dom(host_dom, NULL, &global_count, 0);

    // Allocate the DOM tree and copy over.
    dom_node *device_dom;
    checkCudaErrors(cudaMalloc((void **)&device_dom, sizeof(struct dom_node) * NODE_COUNT));
    dom_node *device_dom_host_mirror = (dom_node *)malloc(sizeof(struct dom_node) * NODE_COUNT);
    memcpy(device_dom_host_mirror, host_dom, sizeof(struct dom_node) * NODE_COUNT);
    munge_dom_pointers(device_dom_host_mirror,
                       (ptrdiff_t)((ptrdiff_t)device_dom_host_mirror - (ptrdiff_t)device_dom));
    checkCudaErrors(cudaMemcpy(device_dom,
                               device_dom_host_mirror,
                               sizeof(struct dom_node) * NODE_COUNT,
                               cudaMemcpyHostToDevice));

    // Allocate the rule tree and copy over.
    css_stylesheet *device_stylesheet;
    checkCudaErrors(cudaMalloc((void **)&device_stylesheet, sizeof(css_stylesheet)));
    checkCudaErrors(cudaMemcpy(device_stylesheet,
                               host_stylesheet,
                               sizeof(css_stylesheet),
                               cudaMemcpyHostToDevice));

    // Create start/stop events.
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Execute the kernel on the GPU.
    checkCudaErrors(cudaEventRecord(start));
    match_selectors_device<<<NODE_COUNT / THREAD_COUNT, THREAD_COUNT>>>(device_dom,
                                                                        device_stylesheet);
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    float gpu_elapsed = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&gpu_elapsed, start, stop));

    // Execute the kernel on the CPU.
    uint64_t cpu_start = mach_absolute_time();
    for (int i = 0; i < NODE_COUNT; i++) {
        match_selectors(host_dom, host_stylesheet, i);
    }
    float cpu_elapsed = (double)(mach_absolute_time() - cpu_start) / 1000000.0;

    fprintf(stderr,
            "Selector matching: GPU %g ms, CPU %g ms (parallel CPU estimate %g ms)\n",
            (double)gpu_elapsed,
            (double)cpu_elapsed,
            (double)cpu_elapsed / ESTIMATED_PARALLEL_SPEEDUP);

    // Do frame construction.
    cpu_start = mach_absolute_time();
    for (int i = 0; i < NODE_COUNT; i++) {
        create_frame(host_dom, i);
    }
    float frame_construction_cpu_elapsed = (double)(mach_absolute_time() - cpu_start) / 1000000.0;

    fprintf(stderr,
            "Frame construction: CPU %g ms (parallel CPU estimate %g ms)\n",
            (double)frame_construction_cpu_elapsed,
            (double)frame_construction_cpu_elapsed / ESTIMATED_PARALLEL_SPEEDUP);

    uint64_t total_cpu_elapsed = cpu_elapsed + frame_construction_cpu_elapsed;
    fprintf(stderr,
            "Total CPU: %g ms (parallel CPU estimate %g ms)\n",
            (double)total_cpu_elapsed,
            (double)total_cpu_elapsed / ESTIMATED_PARALLEL_SPEEDUP);

    float best_case_elapsed = fmax(gpu_elapsed, frame_construction_cpu_elapsed);
    float best_case_parallel_elapsed = fmax(
            (double)gpu_elapsed,
            frame_construction_cpu_elapsed / ESTIMATED_PARALLEL_SPEEDUP);
    fprintf(stderr,
            "Best-case: %g ms (parallel estimate %g ms)\n",
            (double)best_case_elapsed,
            (double)best_case_parallel_elapsed);

    checkCudaErrors(cudaMemcpy(device_dom_host_mirror,
                               device_dom,
                               sizeof(struct dom_node) * NODE_COUNT,
                               cudaMemcpyDeviceToHost));

    check_dom(device_dom_host_mirror);

    checkCudaErrors(cudaFree(device_stylesheet));
    checkCudaErrors(cudaFree(device_dom));

    cudaDeviceReset();
    return 0;
}

