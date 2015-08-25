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

#if 0
__device__ const css_rule *__restrict__ css_cuckoo_hash_find_precomputed_device(
        const css_cuckoo_hash *__restrict__ hash,
        int32_t key,
        int32_t left_index,
        int32_t right_index) {
    CSS_CUCKOO_HASH_FIND_PRECOMPUTED(hash, key, left_index, right_index);
}
#endif

__device__ const css_rule *__restrict__ css_cuckoo_hash_find_device(
        const css_cuckoo_hash *__restrict__ hash,
        int32_t key,
        int32_t left_index,
        int32_t right_index) {
    CSS_CUCKOO_HASH_FIND(hash, key, left_index, right_index);
}

__device__ void sort_selectors_device(struct css_matched_property *matched_properties,
                                      int length) {
    SORT_SELECTORS(matched_properties, length);
}

__global__ void match_selectors_device(const int *ids,
                                       const int *tag_names,
                                       const int *class_counts,
                                       const int *classes,
                                       const css_stylesheet *__restrict__ stylesheet,
                                       const css_rule **matched_rules,
                                       int *matched_rule_counts) {
    MATCH_SELECTORS(ids,
                    tag_names,
                    class_counts,
                    classes,
                    stylesheet,
                    matched_rules,
                    matched_rule_counts,
                    blockIdx.x * THREAD_COUNT + threadIdx.x,
                    css_cuckoo_hash_find_device,
                    css_rule_hash_device,
                    sort_selectors_device,
                    );
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
    int property_index = 0;
    create_stylesheet(host_stylesheet, &property_index);

    // Create the DOM tree on the host.
    int class_count = 0, global_count = 0;
    int *host_ids = (int *)malloc(sizeof(int) * NODE_COUNT);
    int *host_tag_names = (int *)malloc(sizeof(int) * NODE_COUNT);
    int *host_class_counts = (int *)malloc(sizeof(int) * NODE_COUNT);
    int *host_classes = (int *)malloc(sizeof(int) * CLASS_COUNT);
    create_dom(host_ids,
               host_tag_names,
               host_class_counts,
               host_classes,
               &class_count,
               &global_count,
               0);

    // Allocate the IDs and copy over.
    int *device_ids;
    checkCudaErrors(cudaMalloc((void **)&device_ids, sizeof(int) * NODE_COUNT));
    checkCudaErrors(cudaMemcpy(device_ids,
                               host_ids,
                               sizeof(int) * NODE_COUNT,
                               cudaMemcpyHostToDevice));

    // Allocate the tag names and copy over.
    int *device_tag_names;
    checkCudaErrors(cudaMalloc((void **)&device_tag_names, sizeof(int) * NODE_COUNT));
    checkCudaErrors(cudaMemcpy(device_tag_names,
                               host_tag_names,
                               sizeof(int) * NODE_COUNT,
                               cudaMemcpyHostToDevice));

    // Allocate the class counts and copy over.
    int *device_class_counts;
    checkCudaErrors(cudaMalloc((void **)&device_class_counts, sizeof(int) * NODE_COUNT));
    checkCudaErrors(cudaMemcpy(device_class_counts,
                               host_class_counts,
                               sizeof(int) * NODE_COUNT,
                               cudaMemcpyHostToDevice));

    // Allocate the classes and copy over.
    int *device_classes;
    checkCudaErrors(cudaMalloc((void **)&device_classes, sizeof(int) * CLASS_COUNT));
    checkCudaErrors(cudaMemcpy(device_classes,
                               host_classes,
                               sizeof(int) * CLASS_COUNT,
                               cudaMemcpyHostToDevice));

    // Allocate the rule tree and copy over.
    css_stylesheet *device_stylesheet;
    checkCudaErrors(cudaMalloc((void **)&device_stylesheet, sizeof(css_stylesheet)));
    checkCudaErrors(cudaMemcpy(device_stylesheet,
                               host_stylesheet,
                               sizeof(css_stylesheet),
                               cudaMemcpyHostToDevice));

    // Allocate the matched rules.
    const css_rule **device_matched_rules;
    const css_rule **device_matched_rules_host_mirror = (const css_rule **)
        malloc(sizeof(const css_rule *) * MAX_MATCHED_RULES * NODE_COUNT);
    checkCudaErrors(cudaMalloc((void **)&device_matched_rules,
                               sizeof(css_rule *) * MAX_MATCHED_RULES * NODE_COUNT));

    // Allocate the matched property counts.
    int *device_matched_rule_counts;
    int *device_matched_rule_counts_host_mirror = (int *)malloc(sizeof(int) * NODE_COUNT);
    checkCudaErrors(cudaMalloc((void **)&device_matched_rule_counts, sizeof(int) * NODE_COUNT));

    // Create start/stop events.
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Execute the kernel on the GPU.
    checkCudaErrors(cudaEventRecord(start));
    match_selectors_device<<<NODE_COUNT / THREAD_COUNT, THREAD_COUNT>>>(
            device_ids,
            device_tag_names,
            device_class_counts,
            device_classes,
            device_stylesheet,
            device_matched_rules,
            device_matched_rule_counts);
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    float gpu_elapsed = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&gpu_elapsed, start, stop));

    report_timing("GPU", "Selector matching", gpu_elapsed, false, MODE_COPYING);

    // Copy the matched properties and matched property counts back.
    checkCudaErrors(cudaMemcpy(device_matched_rules_host_mirror,
                               device_matched_rules,
                               sizeof(const css_rule *) * MAX_MATCHED_RULES * NODE_COUNT,
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(device_matched_rule_counts_host_mirror,
                               device_matched_rule_counts,
                               sizeof(int) * NODE_COUNT,
                               cudaMemcpyDeviceToHost));

    check_dom(host_ids,
              host_tag_names,
              host_class_counts,
              host_classes,
              device_matched_rules_host_mirror,
              device_matched_rule_counts_host_mirror);

    checkCudaErrors(cudaFree(device_stylesheet));
    checkCudaErrors(cudaFree(device_matched_rule_counts));
    checkCudaErrors(cudaFree(device_matched_rules));
    checkCudaErrors(cudaFree(device_classes));
    checkCudaErrors(cudaFree(device_class_counts));
    checkCudaErrors(cudaFree(device_tag_names));
    checkCudaErrors(cudaFree(device_ids));

    cudaDeviceReset();
    return 0;
}

