// CUDA brute-force Selectron prototype
//
// Patrick Walton <pcwalton@mozilla.com>
//
// Copyright (c) 2015 Mozilla Corporation

#include <sys/time.h>
#include <limits.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef NO_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#endif

#define MAX_NODE_DEPTH      32
#define NODE_COUNT          (1024 * 50)
#define DOM_SIZE            (NODE_COUNT * MAX_NODE_DEPTH)
#define SELECTOR_COUNT      256
#define MAX_SELECTOR_LENGTH 32
#define STYLESHEET_SIZE     (SELECTOR_COUNT * MAX_SELECTOR_LENGTH)

#define THREAD_COUNT        1024
#define CPU_COUNT           4

#define CSS_SELECTOR_TYPE_ID    0
#define CSS_SELECTOR_TYPE_TAG   1
#define CSS_SELECTOR_TYPE_CLASS 2

struct dom_node {
    uint16_t value;
    uint8_t type;
};

struct css_rule_segment {
    uint16_t selector_value;
    uint8_t selector_type;
    uint8_t next_state_if_no_match;
};

struct job {
    uint16_t dom_node;
    uint16_t selector_index;
};

__device__ int device_get_node_index(int node_index, int ancestor_index) {
    //return node_index * MAX_NODE_DEPTH + ancestor_index;
    return ancestor_index * NODE_COUNT + node_index;
}

int get_node_index(int node_index, int ancestor_index) {
    //return node_index * MAX_NODE_DEPTH + ancestor_index;
    return ancestor_index * NODE_COUNT + node_index;
}

__device__ int device_get_selector_index(int selector_index, int segment_index) {
    return selector_index * MAX_SELECTOR_LENGTH + segment_index;
    //return segment_index * SELECTOR_COUNT + selector_index;
}

int get_selector_index(int selector_index, int segment_index) {
    return selector_index * MAX_SELECTOR_LENGTH + segment_index;
    //return segment_index * SELECTOR_COUNT + selector_index;
}

#ifndef NO_CUDA
__global__ void match_selectors_device(const struct dom_node *dom,
                                       const struct css_rule_segment *stylesheet,
                                       const int *selector_lengths,
                                       const job *plan,
                                       uint8_t *result_vector,
                                       int node_count,
                                       int selector_count,
                                       int ancestor_index) {
    int global_id = blockIdx.x * THREAD_COUNT + threadIdx.x;
    const job *job = &plan[global_id];
    int node_index = job->dom_node;
    int selector_index = job->selector_index;

    int selector_cursor = 0;
    int selector_length = selector_lengths[selector_index];
    while (ancestor_index < MAX_NODE_DEPTH && selector_cursor < selector_length) {
        const struct css_rule_segment *segment =
            &stylesheet[device_get_selector_index(selector_index, selector_cursor)];
        const struct dom_node *ancestor =
            &dom[device_get_node_index(node_index, ancestor_index)];
        if (ancestor->value != 0) {
            ancestor_index++;

            // FIXME(pcwalton): Need to sort classes.
            if (segment->selector_type == ancestor->type) {
                if (segment->selector_value == ancestor->value)
                    selector_cursor++;
                else if (segment->selector_type != CSS_SELECTOR_TYPE_CLASS)
                    selector_cursor = segment->next_state_if_no_match;
            } else {
                selector_cursor = segment->next_state_if_no_match;
            }
        } else {
            break;
        }
    }

    result_vector[global_id] = selector_cursor == selector_length;
}
#endif

#if 0
__global__ void match_selectors_device(const struct dom_node *dom,
                                       const struct css_rule_segment *stylesheet,
                                       uint8_t *result_vector,
                                       int node_count,
                                       int selector_count) {
    int global_id = blockIdx.x * THREAD_COUNT + threadIdx.x;
    int node_index = global_id / MAX_NODE_DEPTH / selector_count;
    int selector_index = global_id / MAX_NODE_DEPTH % selector_count;
    int ancestor_index = global_id % MAX_NODE_DEPTH;

    int selector_cursor = 0;
    bool matched = false;
    const struct css_rule_segment *segment =
        &stylesheet[selector_index * MAX_SELECTOR_LENGTH + selector_cursor];
    if (segment->selector_value == 0) {
        matched = true;
    } else {
        const struct dom_node *ancestor = &dom[device_get_node_index(node_index, ancestor_index)];
        if (ancestor->value != 0) {
            ancestor_index++;

            // FIXME(pcwalton): Need to sort classes.
            if (segment->selector_type == ancestor->type) {
                if (segment->selector_value == ancestor->value)
                    selector_cursor++;
                else if (segment->selector_type != CSS_SELECTOR_TYPE_CLASS)
                    selector_cursor = segment->next_state_if_no_match;
            } else {
                selector_cursor = segment->next_state_if_no_match;
            }
        }
    }

    result_vector[node_index * SELECTOR_COUNT + selector_index] = matched;
}
#endif

struct thread_start_info {
    const struct dom_node *dom;
    const struct css_rule_segment *stylesheet;
    const struct job *plan;
    uint8_t *result_vector;
    int node_count;
    int selector_count;
    int plan_length;
};

#if 0
const int STATES[3][3][2] = {
    {
        // didn't match, matched
        { 0, 0 },
        { 
};
#endif

void *match_selectors_host(void *userdata) {
    struct thread_start_info *info = (struct thread_start_info *)userdata;
    const struct dom_node *dom = info->dom;
    const struct css_rule_segment *stylesheet = info->stylesheet;
    const struct job *plan = info->plan;
    uint8_t *result_vector = info->result_vector;
    int plan_length = info->plan_length;

    for (int plan_index = 0; plan_index < plan_length; plan_index++) {
        int node_index = plan[plan_index].dom_node;
        int selector_index = plan[plan_index].selector_index;

        int selector_cursor = 0, ancestor_index = 0;
        bool matched = false;
        while (ancestor_index < MAX_NODE_DEPTH) {
            const struct css_rule_segment *segment =
                &stylesheet[get_selector_index(selector_index, selector_cursor)];
            if (segment->selector_value == 0) {
                matched = true;
                break;
            }

            const struct dom_node *ancestor = &dom[get_node_index(node_index, ancestor_index)];
            if (ancestor->value == 0)
                break;
            ancestor_index++;
            // FIXME(pcwalton): Need to sort classes.
            if (segment->selector_type == ancestor->type) {
                if (segment->selector_value == ancestor->value) {
                    selector_cursor++;
                } else if (segment->selector_type != CSS_SELECTOR_TYPE_CLASS) {
                    selector_cursor = segment->next_state_if_no_match;
                }
            } else {
                selector_cursor = segment->next_state_if_no_match;
            }
        }

        result_vector[plan_index] = matched;
    }
    return NULL;
}

int selector_sort_callback(const void *pa, const void *pb) {
    struct css_rule_segment *a = (struct css_rule_segment *)pa;
    struct css_rule_segment *b = (struct css_rule_segment *)pb;
    int a_len = 0;
    while (a_len < MAX_SELECTOR_LENGTH && a[a_len].selector_value != 0)
        a_len++;
    int b_len = 0;
    while (b_len < MAX_SELECTOR_LENGTH && b[b_len].selector_value != 0)
        b_len++;
    return a_len - b_len;
}

#ifndef NO_CUDA
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
#endif

void print_match_results(uint8_t *result_vector, int plan_length) {
    int n_printed = 0;
    printf("matching plans: ");
    for (int plan_index = 0; plan_index < plan_length; plan_index++) {
        bool result = result_vector[plan_index];
        if (result) {
            printf("%d ", plan_index);
            n_printed++;
            if ((n_printed % 25) == 0)
                //printf("\n");
                break;
        }
    }
    printf("\n");
}

int main(int argc, const char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: brute-selectron dom.txt css.txt\n");
        return 0;
    }

    srand(time(NULL));

#ifndef NO_CUDA
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int device_id = get_cuda_device(false);
    checkCudaErrors(cudaSetDevice(device_id));
#endif

    // Allocate and read in nodes on the host.
    struct dom_node *host_dom_nodes =
        (struct dom_node *)malloc(sizeof(struct dom_node) * NODE_COUNT * MAX_NODE_DEPTH);
    memset(host_dom_nodes, '\0', sizeof(struct dom_node) * NODE_COUNT * MAX_NODE_DEPTH);
    FILE *f = fopen(argv[1], "r");
    char line[1024];
    int node_count = 0;
    while (fgets(line, sizeof(line), f) != NULL) {
        int ancestor_index = 0;

        char *node_descriptor, *line_ptr = line;
        while ((node_descriptor = strsep(&line_ptr, " ")) != NULL) {
            if (strlen(node_descriptor) == 0)
                continue;

            int id = 0;
            int tag = 0;
            int class_count = 0;
            int classes[32] = { 0 };

            char *token;
            while ((token = strsep(&node_descriptor, ",")) != NULL) {
                switch (token[0]) {
                case '&':
                    tag = strtol(&token[1], NULL, 0);
                    break;
                case '#':
                    id = strtol(&token[1], NULL, 0);
                    break;
                case '.':
                    classes[class_count++] = strtol(&token[1], NULL, 0);
                    break;
                }

                if (class_count == sizeof(classes) / sizeof(classes[0]))
                    break;
            }

            struct dom_node node;

            if (ancestor_index < MAX_NODE_DEPTH && id != 0) {
                node.value = id;
                node.type = CSS_SELECTOR_TYPE_ID;
                host_dom_nodes[get_node_index(node_count, ancestor_index)] = node;
                ancestor_index++;
            }

            if (ancestor_index < MAX_NODE_DEPTH && tag != 0) {
                node.value = tag;
                node.type = CSS_SELECTOR_TYPE_TAG;
                host_dom_nodes[get_node_index(node_count, ancestor_index)] = node;
                ancestor_index++;
            }

            for (int class_index = 0; class_index < class_count; class_index++) {
                if (ancestor_index < MAX_NODE_DEPTH) {
                    node.value = classes[class_index];
                    node.type = CSS_SELECTOR_TYPE_CLASS;
                    host_dom_nodes[get_node_index(node_count, ancestor_index)] = node;
                    ancestor_index++;
                }
            }
        }
        node_count++;
    }
    fclose(f);

#if 0
    // Pad out the number of nodes.
    for (int i = node_count; i < NODE_COUNT; i++) {
        for (int j = 0; j < MAX_NODE_DEPTH; j++) {
            host_dom_nodes[get_node_index(i, j)] =
                host_dom_nodes[get_node_index(i % NODE_COUNT, j)];
        }
    }
    node_count = NODE_COUNT;
#endif

#if 0
    for (int i = 0; i < node_count; i++) {
        for (int j = 0; j < MAX_NODE_DEPTH; j++) {
            printf("%d ", host_dom_nodes[i * MAX_NODE_DEPTH + j]);
        }
        printf("\n");
    }
#endif

    // Create the stylesheet on the host.
    struct css_rule_segment *host_stylesheet = (struct css_rule_segment *)
        malloc(sizeof(struct css_rule_segment) * SELECTOR_COUNT * MAX_SELECTOR_LENGTH);
    memset(host_stylesheet,
           '\0',
           sizeof(struct css_rule_segment) * SELECTOR_COUNT * MAX_SELECTOR_LENGTH);
    f = fopen(argv[2], "r");
    int selector_count = 0;
    while (fgets(line, sizeof(line), f) != NULL) {
        int segment_index = 0;
        char *token, *line_ptr = line;
        while ((token = strsep(&line_ptr, " ")) != NULL) {
            if (strlen(token) == 0)
                continue;
            if (segment_index == MAX_SELECTOR_LENGTH)
                continue;

            struct css_rule_segment segment;
            switch (token[0]) {
            case '#':
                segment.selector_type = CSS_SELECTOR_TYPE_ID;
                break;
            case '&':
                segment.selector_type = CSS_SELECTOR_TYPE_TAG;
                break;
            default:
                segment.selector_type = CSS_SELECTOR_TYPE_CLASS;
            }
            segment.selector_value = strtol(&token[1], &token, 0);
            if (token[0] == '|') {
                segment.next_state_if_no_match = strtol(&token[1], NULL, 0);
            }
            printf("SC=%d SI=%d type=%d value=%d, nsinm=%d\n",
                   (int)selector_count,
                   (int)segment_index,
                   (int)segment.selector_type,
                   (int)segment.selector_value,
                   (int)segment.next_state_if_no_match);

            host_stylesheet[get_selector_index(selector_count, segment_index)] = segment;
            segment_index++;
        }

        if (segment_index < MAX_SELECTOR_LENGTH)
            selector_count++;
        if (selector_count == SELECTOR_COUNT)
            break;
    }

    /*qsort(host_stylesheet,
          selector_count,
          sizeof(struct css_rule_segment) * MAX_SELECTOR_LENGTH,
          selector_sort_callback);*/

    for (int i = 0; i < MAX_SELECTOR_LENGTH; i++) {
        struct css_rule_segment *segment = &host_stylesheet[get_selector_index(30, i)];
        printf("SC=%d SI=%d type=%d value=%d, nsinm=%d\n",
               (int)30,
               (int)i,
               (int)segment->selector_type,
               (int)segment->selector_value,
               (int)segment->next_state_if_no_match);
    }

    // Create the selector lengths.
    int *host_selector_lengths = (int *)malloc(sizeof(int) * SELECTOR_COUNT);
    for (int i = 0; i < SELECTOR_COUNT; i++) {
        int length = 0;
        while (length < MAX_SELECTOR_LENGTH &&
                host_stylesheet[get_selector_index(i, length)].selector_value != 0)
            length++;
        host_selector_lengths[i] = length;
    }

    // Create the plan on the host.
    struct job *host_plan = (struct job *)malloc(sizeof(struct job) * SELECTOR_COUNT * node_count);
    int plan_length = 0;
    for (int node_index = 0; node_index < node_count; node_index++) {
        // Find the last node.
        int last_ancestor_index = MAX_NODE_DEPTH - 1;
        struct dom_node *last_node =
            &host_dom_nodes[get_node_index(node_index, last_ancestor_index)];
        for (int prev_ancestor_index = MAX_NODE_DEPTH - 2;
             prev_ancestor_index >= 0;
             prev_ancestor_index--) {
            struct dom_node *prev_node =
                &host_dom_nodes[get_node_index(node_index, prev_ancestor_index)];
            if (last_node->value == 0) {
                last_node = prev_node;
                last_ancestor_index = prev_ancestor_index;
                continue;
            }
            if (prev_node->type < last_node->type) {
                last_node = prev_node;
                last_ancestor_index = prev_ancestor_index;
                continue;
            }
            break;
        }

        // Find all selectors that match the last node.
        for (int selector_index = 0; selector_index < selector_count; selector_index++) {
            // Find the last segment of the selector.
            int selector_length = host_selector_lengths[selector_index];
            int last_segment_index = selector_length - 1;
            struct css_rule_segment *last_segment =
                &host_stylesheet[get_selector_index(selector_index, last_segment_index)];
            for (int prev_segment_index = selector_length - 2;
                 prev_segment_index >= 0;
                 prev_segment_index--) {
                struct css_rule_segment *prev_segment =
                    &host_stylesheet[get_selector_index(selector_index, prev_segment_index)];
                if (last_segment->selector_value == 0) {
                    last_segment = prev_segment;
                    last_segment_index = prev_segment_index;
                    continue;
                }
                if (prev_segment->selector_type < last_segment->selector_type) {
                    last_segment = prev_segment;
                    last_segment_index = prev_segment_index;
                    continue;
                }
                break;
            }

            bool is_candidate = false;
            for (int ancestor_index = last_ancestor_index;
                 ancestor_index < MAX_NODE_DEPTH && !is_candidate;
                 ancestor_index++) {
                struct dom_node *node =
                    &host_dom_nodes[get_node_index(node_index, ancestor_index)];
                for (int segment_index = last_segment_index;
                     segment_index < selector_length && !is_candidate;
                     segment_index++) {
                    struct css_rule_segment *segment =
                        &host_stylesheet[get_selector_index(selector_index, segment_index)];
                    if (segment->selector_type == node->type &&
                            segment->selector_value == node->value) {
                        is_candidate = true;
                    }
                }
            }

            if (last_segment->selector_value == 113 && node_index == 2221) {
                printf("plan %d: last segment of selector %d = %d, ancestor index = %d, is "
                       "candidate=%d\n",
                       plan_length,
                       selector_index,
                       last_segment_index,
                       last_ancestor_index,
                       (int)is_candidate);
            }

            if (is_candidate) {
                host_plan[plan_length].dom_node = node_index;
                host_plan[plan_length].selector_index = selector_index;
                /*
                printf("plan %d: node %d selector %d\n",
                       (int)plan_length,
                       (int)node_index,
                       (int)selector_index);

                printf("nodes:\n");
                for (int ancestor_index = 0; ancestor_index < MAX_NODE_DEPTH; ancestor_index++) {
                    struct dom_node *node =
                        &host_dom_nodes[get_node_index(node_index, ancestor_index)];
                    printf("%d/%d ", node->type, node->value);
                }
                printf("\nrule:\n");
                for (int segment_index = 0; segment_index < selector_length; segment_index++) {
                    struct css_rule_segment *segment =
                        &host_stylesheet[get_selector_index(selector_index, segment_index)];
                    printf("%d/%d ", segment->selector_type, segment->selector_value);
                }
                printf("\n");
                */

                plan_length++;
            }
        }
    }
    printf("plan length=%d\n", plan_length);

#if 0
    // Pad out the plan length.
    for (int plan_index = plan_length; plan_index < NODE_COUNT; plan_index++)
        host_plan[plan_index] = host_plan[plan_index % plan_length];
    plan_length = NODE_COUNT;
#endif

#if 0
    printf("---\n");
    for (int i = 0; i < SELECTOR_COUNT; i++) {
        for (int j = 0; j < MAX_SELECTOR_LENGTH; j++) {
            printf("%d ", host_stylesheet[i * MAX_SELECTOR_LENGTH + j]);
        }
        printf("\n");
    }
#endif

#ifndef NO_CUDA
    // Allocate the DOM nodes and copy over.
    struct dom_node *device_dom_nodes;
    checkCudaErrors(cudaMalloc((void **)&device_dom_nodes, sizeof(struct dom_node) * DOM_SIZE));
    checkCudaErrors(cudaMemcpy(device_dom_nodes,
                               host_dom_nodes,
                               sizeof(struct dom_node) * DOM_SIZE,
                               cudaMemcpyHostToDevice));

    // Allocate the stylesheet and copy over.
    struct css_rule_segment *device_stylesheet;
    checkCudaErrors(cudaMalloc((void **)&device_stylesheet,
                               sizeof(struct css_rule_segment) * STYLESHEET_SIZE));
    checkCudaErrors(cudaMemcpy(device_stylesheet,
                               host_stylesheet,
                               sizeof(struct css_rule_segment) * STYLESHEET_SIZE,
                               cudaMemcpyHostToDevice));

    // Allocate the selector lengths and copy over.
    int *device_selector_lengths;
    checkCudaErrors(cudaMalloc((void **)&device_selector_lengths, sizeof(int) * SELECTOR_COUNT));
    checkCudaErrors(cudaMemcpy(device_selector_lengths,
                               host_selector_lengths,
                               sizeof(int) * SELECTOR_COUNT,
                               cudaMemcpyHostToDevice));

    // Allocate the plan and copy over.
    struct job *device_plan;
    checkCudaErrors(cudaMalloc((void **)&device_plan, sizeof(struct job) * plan_length));
    checkCudaErrors(cudaMemcpy(device_plan,
                               host_plan,
                               sizeof(struct job) * plan_length,
                               cudaMemcpyHostToDevice));

#endif

    // Allocate the result vector on the host.
    uint8_t *device_result_vector_host_mirror =
        (uint8_t *)malloc(sizeof(uint8_t) * node_count * SELECTOR_COUNT);
    memset(device_result_vector_host_mirror, '\0', sizeof(uint8_t) * node_count * SELECTOR_COUNT);

#ifndef NO_CUDA
    // Allocate the result vector on the device.
    uint8_t *device_result_vector;
    checkCudaErrors(cudaMalloc((uint8_t **)&device_result_vector,
                               sizeof(uint8_t) * node_count * SELECTOR_COUNT));
    checkCudaErrors(cudaMemcpy(device_result_vector,
                               device_result_vector_host_mirror,
                               sizeof(uint8_t) * node_count * SELECTOR_COUNT,
                               cudaMemcpyHostToDevice));

    // Create start/stop events.
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Execute the kernel on the GPU.
    checkCudaErrors(cudaEventRecord(start));
    match_selectors_device<<<plan_length / THREAD_COUNT, THREAD_COUNT>>>(
            device_dom_nodes,
            device_stylesheet,
            device_selector_lengths,
            device_plan,
            device_result_vector,
            node_count,
            selector_count,
            0);
    checkCudaErrors(cudaPeekAtLastError());

    // Print timings.
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    float gpu_elapsed = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&gpu_elapsed, start, stop));
    fprintf(stderr, "GPU elapsed time: %f ms\n", gpu_elapsed);

    // Copy back the result vector.
    checkCudaErrors(cudaMemcpy(device_result_vector_host_mirror,
                               device_result_vector,
                               sizeof(uint8_t) * node_count * SELECTOR_COUNT,
                               cudaMemcpyDeviceToHost));

    // Print out the first few, as a sanity check.
    print_match_results(device_result_vector_host_mirror, plan_length);
#endif

    // Run on the host.
    struct timeval cpu_start;
    gettimeofday(&cpu_start, NULL);

    pthread_t threads[CPU_COUNT];
    struct thread_start_info infos[CPU_COUNT];
    int plan_index = 0;
    for (int i = 0; i < CPU_COUNT; i++) {
        infos[i].dom = host_dom_nodes;
        infos[i].stylesheet = host_stylesheet;
        infos[i].result_vector = &device_result_vector_host_mirror[plan_index];
        infos[i].plan = &host_plan[plan_index];
        infos[i].node_count = node_count;
        infos[i].selector_count = selector_count;
        infos[i].plan_length = plan_length / CPU_COUNT;
        plan_index += infos[i].plan_length;
        pthread_create(&threads[i], NULL, match_selectors_host, (void *)&infos[i]);
    }
    for (int i = 0; i < CPU_COUNT; i++)
        pthread_join(threads[i], NULL);

    struct timeval cpu_end;
    gettimeofday(&cpu_end, NULL);

    // Print host timings.
    fprintf(stderr,
            "CPU elapsed time: %f ms\n",
            ((double)cpu_end.tv_sec - (double)cpu_start.tv_sec) * 1000.0 +
            ((double)cpu_end.tv_usec - (double)cpu_start.tv_usec) / 1000.0);

    // Print out the first few, as a sanity check.
    print_match_results(device_result_vector_host_mirror, plan_length);

#ifndef NO_CUDA
    // Clean up and exit.
    checkCudaErrors(cudaFree(device_result_vector));
    checkCudaErrors(cudaFree(device_selector_lengths));
    checkCudaErrors(cudaFree(device_plan));
    checkCudaErrors(cudaFree(device_stylesheet));
    checkCudaErrors(cudaFree(device_dom_nodes));
    cudaDeviceReset();
#endif

    return 0;
}

