// OpenCL Selectron prototype
//
// Patrick Walton <pcwalton@mozilla.com>
//
// Copyright (c) 2014 Mozilla Corporation

#include "selectron.h"

#define WIN32_LEAN_AND_MEAN

#include <stdio.h>
#include <time.h>

#ifndef __APPLE__
#include <windows.h>
#include <CL/opencl.h>
#else
#include <mach/mach_time.h>
#include <OpenCL/OpenCL.h>
#endif

// Hack to allow stringification of macro expansion

#define XSTRINGIFY(s)   STRINGIFY(s)
#define STRINGIFY(s)    #s

#ifndef __APPLE__
uint64_t mach_absolute_time() {
    static LARGE_INTEGER freq = { 0, 0 };
    if (!freq.QuadPart)
        QueryPerformanceFrequency(&freq);

    LARGE_INTEGER time;
    QueryPerformanceCounter(&time);
    return time.QuadPart * 1000000000 / freq.QuadPart;
}
#endif

const char *selector_matching_kernel_source = "\n"
    XSTRINGIFY(STRUCT_CSS_RULE) ";\n"
    XSTRINGIFY(STRUCT_CSS_CUCKOO_HASH) ";\n"
    XSTRINGIFY(STRUCT_CSS_MATCHED_PROPERTY) ";\n"
    XSTRINGIFY(STRUCT_CSS_STYLESHEET_SOURCE) ";\n"
    XSTRINGIFY(STRUCT_CSS_STYLESHEET) ";\n"
    "unsigned int css_rule_hash(unsigned int key, unsigned int seed) {\n"
    "   " XSTRINGIFY(CSS_RULE_HASH(key, seed)) ";\n"
    "}\n"
    "\n"
    "__global struct css_rule *css_cuckoo_hash_find(__global struct css_cuckoo_hash *hash,\n"
    "                                               int key,\n"
    "                                               int left_index,\n"
    "                                               int right_index) {\n"
    "   " XSTRINGIFY(CSS_CUCKOO_HASH_FIND(hash, key, left_index, right_index)) ";\n"
    "}\n"
    "\n"
    "void sort_selectors(struct css_matched_property *matched_properties, int length) {\n"
    "   " XSTRINGIFY(SORT_SELECTORS(matched_properties, length)) ";\n"
    "}\n"
    "\n"
    "__kernel void match_selectors(__global int *ids, \n"
    "                              __global int *tag_names, \n"
    "                              __global int *class_counts, \n"
    "                              __global int *classes, \n"
    "                              __global struct css_stylesheet *stylesheet, \n"
    "                              __global const struct css_rule *__global *matched_rules, \n"
    "                              __global int *matched_rule_counts) {\n"
    "   int index = get_global_id(0);\n"
    "   " XSTRINGIFY(MATCH_SELECTORS(ids,
                                     tag_names,
                                     class_counts,
                                     classes,
                                     stylesheet,
                                     matched_rules,
                                     matched_rule_counts,
                                     index,
                                     css_cuckoo_hash_find,
                                     css_rule_hash,
                                     sort_selectors,
                                     __global)) ";\n"
    "}\n";

#define CHECK_CL(call) \
    do { \
        int _err = call; \
        if (_err != CL_SUCCESS) { \
            fprintf(stderr, \
                    "OpenCL call failed (%s, %d): %d, %s\n", \
                    __FILE__, \
                    __LINE__, \
                    _err, \
                    #call); \
            abort(); \
        } \
    } while(0)

void abort_unless(int error) {
    if (!error) {
        fprintf(stderr, "OpenCL error");
    }
}

void abort_if_null(void *ptr, const char *msg = "") {
    if (!ptr) {
        fprintf(stderr, "OpenCL error: %s\n", msg);
    }
}

#define FIND_EXTENSION(name, platform) \
    static name##_fn name = NULL; \
    do { \
        if (!name) { \
            name = (name##_fn)clGetExtensionFunctionAddressForPlatform(platform, #name); \
            abort_if_null(name, "couldn't find extension " #name); \
        } \
    } while(0)

#define MALLOC(context, commands, err, mode, name, perm, type, count) \
    do { \
        device_##name = clCreateBuffer(context, perm, sizeof(type) * (count), NULL, NULL); \
        abort_if_null(device_##name); \
        if ((mode) == MODE_MAPPED) { \
            host_##name = (type *)clEnqueueMapBuffer(commands, \
                                                     device_##name, \
                                                     CL_TRUE, \
                                                     CL_MAP_READ | CL_MAP_WRITE, \
                                                     0, \
                                                     sizeof(type) * (count), \
                                                     0, \
                                                     NULL, \
                                                     NULL, \
                                                     &err); \
            CHECK_CL(err); \
        } else { \
            host_##name = (type *)malloc(sizeof(type) * count); \
        } \
        /*fprintf(stderr, "mapped " #name " to %p\n", host_##name);*/ \
    } while(0)

void go(cl_platform_id platform, time_t seed, cl_device_type device_type, int mode) {
#ifndef NO_SVM
    FIND_EXTENSION(clSVMAllocAMD, platform);
    FIND_EXTENSION(clSVMFreeAMD, platform);
    FIND_EXTENSION(clSetKernelArgSVMPointerAMD, platform);
#endif

    // Perform OpenCL initialization.
    cl_device_id device_id;
    CHECK_CL(clGetDeviceIDs(platform, device_type, 1, &device_id, NULL));
    size_t device_name_size;
    CHECK_CL(clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, NULL, &device_name_size));
    char *device_name = (char *)malloc(device_name_size);
    CHECK_CL(clGetDeviceInfo(device_id, CL_DEVICE_NAME, device_name_size, device_name, NULL));
    fprintf(stderr, "device found: %s\n", device_name);

    cl_context_properties props[6] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
#ifndef NO_SVM
        CL_HSA_ENABLED_AMD, (cl_context_properties)1,
#endif
        0, 0
    };

    cl_int err;
    cl_context context = clCreateContextFromType(props, device_type, NULL, NULL, &err);
    CHECK_CL(err);
    abort_if_null(context);

    cl_command_queue commands = clCreateCommandQueue(context, device_id, 0, &err);
    abort_if_null(commands);

    cl_program program = clCreateProgramWithSource(context,
                                                   1,
                                                   (const char **)&selector_matching_kernel_source,
                                                   NULL,
                                                   &err);
    CHECK_CL(err);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        cl_build_status build_status;
        CHECK_CL(clGetProgramBuildInfo(program,
                                       device_id,
                                       CL_PROGRAM_BUILD_STATUS,
                                       sizeof(cl_build_status),
                                       &build_status,
                                       NULL));
        fprintf(stderr, "Build status: %d\n", (int)build_status);

        size_t log_size;
        CHECK_CL(clGetProgramBuildInfo(program,
                                       device_id,
                                       CL_PROGRAM_BUILD_LOG,
                                       0,
                                       NULL,
                                       &log_size));

        char *log = (char *)malloc(log_size);
        CHECK_CL(clGetProgramBuildInfo(program,
                                       device_id,
                                       CL_PROGRAM_BUILD_LOG,
                                       log_size,
                                       log,
                                       NULL));
        fprintf(stderr, "Compilation error: %s\n", log);
        exit(1);
    }

    size_t binary_sizes_sizes;
    CHECK_CL(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, 0, NULL, &binary_sizes_sizes));
    size_t *binary_sizes = (size_t *)malloc(sizeof(size_t) * binary_sizes_sizes);
    CHECK_CL(clGetProgramInfo(program,
                              CL_PROGRAM_BINARY_SIZES,
                              binary_sizes_sizes,
                              binary_sizes,
                              NULL));
    char **binaries = (char **)malloc(binary_sizes_sizes / sizeof(size_t));
    for (int i = 0; i < binary_sizes_sizes / sizeof(size_t); i++)
        binaries[i] = (char *)malloc(binary_sizes[i]);
    CHECK_CL(clGetProgramInfo(program, CL_PROGRAM_BINARIES, binary_sizes_sizes, binaries, NULL));
    for (int i = 0; i < binary_sizes_sizes / sizeof(size_t); i++) {
        char *path = (char *)malloc(32);
        sprintf(path, "prg%c%02d.plist", (device_type == CL_DEVICE_TYPE_CPU) ? 'c' : 'g', i);
        FILE *f = fopen(path, "w");
        fwrite(binaries[i], binary_sizes[i], 1, f);
        fclose(f);
        free(path);
    }

    cl_kernel kernel = clCreateKernel(program, "match_selectors", &err);
    CHECK_CL(err);

    srand(seed);

    struct css_stylesheet *host_stylesheet = NULL;
    const struct css_rule **host_matched_rules = NULL;
    int *host_matched_rule_counts = NULL;
    int *host_ids = NULL;
    int *host_tag_names = NULL;
    int *host_class_counts = NULL;
    int *host_classes = NULL;

    cl_mem device_ids, device_tag_names, device_class_counts, device_stylesheet, device_classes;
    cl_mem device_matched_rules, device_matched_rule_counts;
    if (mode != MODE_SVM) {
        MALLOC(context,
               commands,
               err,
               mode,
               ids,
               CL_MEM_READ_WRITE,
               int,
               NODE_COUNT);
        MALLOC(context,
               commands,
               err,
               mode,
               tag_names,
               CL_MEM_READ_WRITE,
               int,
               NODE_COUNT);
        MALLOC(context,
               commands,
               err,
               mode,
               class_counts,
               CL_MEM_READ_WRITE,
               int,
               NODE_COUNT);
        MALLOC(context,
               commands,
               err,
               mode,
               classes,
               CL_MEM_READ_WRITE,
               int,
               CLASS_COUNT);
        MALLOC(context,
               commands,
               err,
               mode,
               stylesheet,
               CL_MEM_READ_WRITE,
               struct css_stylesheet,
               1);
        MALLOC(context,
               commands,
               err,
               mode,
               matched_rules,
               CL_MEM_READ_WRITE,
               const struct css_rule *,
               (MAX_MATCHED_RULES * NODE_COUNT));
        MALLOC(context,
               commands,
               err,
               mode,
               matched_rule_counts,
               CL_MEM_READ_WRITE,
               int,
               NODE_COUNT);
    } else {
#ifndef NO_SVM
        // Allocate the rule tree.
        host_stylesheet = (struct css_stylesheet *)clSVMAllocAMD(context,
                                                                 0,
                                                                 sizeof(struct css_stylesheet),
                                                                 16);
        abort_if_null(host_stylesheet, "failed to allocate host stylesheet");
        host_properties = (struct css_property *)clSVMAllocAMD(
            context,
            0,
            sizeof(struct css_property) * PROPERTY_COUNT,
            16);
        abort_if_null(host_properties, "failed to allocate host properties");

        // Allocate the DOM tree.
        host_dom = (struct dom_node *)clSVMAllocAMD(context,
                                                    0,
                                                    sizeof(struct dom_node) * NODE_COUNT,
                                                    16);
        abort_if_null(host_dom, "failed to allocate host DOM");

        // Allocate the classes.
        host_classes = (struct dom_node *)clSVMAllocAMD(context,
                                                        0,
                                                        sizeof(int) * CLASS_COUNT,
                                                        16);
        abort_if_null(host_dom, "failed to allocate host classes");
#endif
    }

    // Create the stylesheet and the DOM.
    uint64_t start = mach_absolute_time();
    int property_index = 0;
    create_stylesheet(host_stylesheet, &property_index);
    int class_count = 0, global_count = 0;
    create_dom(host_ids,
               host_tag_names,
               host_class_counts,
               host_classes,
               &class_count,
               &global_count,
               0);

    double elapsed = (double)(mach_absolute_time() - start) / 1000000.0;
    report_timing(device_name, "stylesheet/DOM creation", elapsed, false, mode);

    // Unmap or copy buffers if necessary.
    start = mach_absolute_time();
    switch (mode) {
    case MODE_MAPPED:
        CHECK_CL(clEnqueueUnmapMemObject(commands,
                                         device_stylesheet,
                                         host_stylesheet,
                                         0,
                                         NULL,
                                         NULL));
        CHECK_CL(clEnqueueUnmapMemObject(commands,
                                         device_matched_rules,
                                         host_matched_rules,
                                         0,
                                         NULL,
                                         NULL));
        CHECK_CL(clEnqueueUnmapMemObject(commands,
                                         device_class_counts,
                                         host_class_counts,
                                         0,
                                         NULL,
                                         NULL));
        CHECK_CL(clEnqueueUnmapMemObject(commands,
                                         device_tag_names,
                                         host_tag_names,
                                         0,
                                         NULL,
                                         NULL));
        CHECK_CL(clEnqueueUnmapMemObject(commands,
                                         device_ids,
                                         host_ids,
                                         0,
                                         NULL,
                                         NULL));
        CHECK_CL(clEnqueueUnmapMemObject(commands,
                                         device_classes,
                                         host_classes,
                                         0,
                                         NULL,
                                         NULL));
        break;
    case MODE_COPYING:
        CHECK_CL(clEnqueueWriteBuffer(commands,
                                      device_stylesheet,
                                      CL_TRUE,
                                      0,
                                      sizeof(struct css_stylesheet),
                                      host_stylesheet,
                                      0,
                                      NULL,
                                      NULL));
        CHECK_CL(clEnqueueWriteBuffer(
                    commands,
                    device_matched_rules,
                    CL_TRUE,
                    0,
                    sizeof(const struct css_rule *) * MAX_MATCHED_RULES * NODE_COUNT,
                    host_matched_rules,
                    0,
                    NULL,
                    NULL));
        CHECK_CL(clEnqueueWriteBuffer(commands,
                                      device_ids,
                                      CL_TRUE,
                                      0,
                                      sizeof(int) * NODE_COUNT,
                                      host_ids,
                                      0,
                                      NULL,
                                      NULL));
        CHECK_CL(clEnqueueWriteBuffer(commands,
                                      device_tag_names,
                                      CL_TRUE,
                                      0,
                                      sizeof(int) * NODE_COUNT,
                                      host_tag_names,
                                      0,
                                      NULL,
                                      NULL));
        CHECK_CL(clEnqueueWriteBuffer(commands,
                                      device_classes,
                                      CL_TRUE,
                                      0,
                                      sizeof(int) * CLASS_COUNT,
                                      host_classes,
                                      0,
                                      NULL,
                                      NULL));
        CHECK_CL(clEnqueueWriteBuffer(commands,
                                      device_class_counts,
                                      CL_TRUE,
                                      0,
                                      sizeof(int) * NODE_COUNT,
                                      host_class_counts,
                                      0,
                                      NULL,
                                      NULL));
    }

    // Set the arguments to the kernel.
    if (mode != MODE_SVM) {
        CHECK_CL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_ids));
        CHECK_CL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_tag_names));
        CHECK_CL(clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_class_counts));
        CHECK_CL(clSetKernelArg(kernel, 3, sizeof(cl_mem), &device_classes));
        CHECK_CL(clSetKernelArg(kernel, 4, sizeof(cl_mem), &device_stylesheet));
        CHECK_CL(clSetKernelArg(kernel, 5, sizeof(cl_mem), &device_matched_rules));
        CHECK_CL(clSetKernelArg(kernel, 6, sizeof(cl_mem), &device_matched_rule_counts));
    } else {
#ifndef NO_SVM
        CHECK_CL(clSetKernelArgSVMPointerAMD(kernel, 0, host_dom));
        CHECK_CL(clSetKernelArgSVMPointerAMD(kernel, 1, host_stylesheet));
        CHECK_CL(clSetKernelArgSVMPointerAMD(kernel, 2, host_properties));
        CHECK_CL(clSetKernelArgSVMPointerAMD(kernel, 3, host_classes));
#endif
    }

    elapsed = (double)(mach_absolute_time() - start) / 1000000.0;
    report_timing(device_name, "buffer copying", elapsed, false, mode);

    // Figure out the allowable size.
    size_t local_workgroup_size;
    CHECK_CL(clGetKernelWorkGroupInfo(kernel,
                                      device_id,
                                      CL_KERNEL_WORK_GROUP_SIZE,
                                      sizeof(local_workgroup_size),
                                      &local_workgroup_size,
                                      NULL));
    fprintf(stderr, "local workgroup size=%d\n", (int)local_workgroup_size);

    clFinish(commands);

    start = mach_absolute_time();
    size_t global_work_size = NODE_COUNT;
    CHECK_CL(clEnqueueNDRangeKernel(commands,
                                    kernel,
                                    1,
                                    NULL,
                                    &global_work_size,
                                    &local_workgroup_size,
                                    0,
                                    NULL,
                                    NULL));
    clFinish(commands);

    // Report timing.
    elapsed = (double)(mach_absolute_time() - start) / 1000000.0;
    report_timing(device_name, "kernel execution", elapsed, false, mode);

    if (mode != MODE_SVM) {
        // Retrieve the matched properties.
        const struct css_rule **device_matched_rules_mirror =
            (const struct css_rule **)malloc(
                    sizeof(const struct css_rule *) * MAX_MATCHED_RULES * NODE_COUNT);
        CHECK_CL(clEnqueueReadBuffer(
                    commands,
                    device_matched_rules,
                    CL_TRUE,
                    0,
                    sizeof(const struct css_rule *) * MAX_MATCHED_RULES * NODE_COUNT,
                    device_matched_rules_mirror,
                    0,
                    NULL,
                    NULL));
        int *device_matched_rule_counts_mirror = (int *)malloc(sizeof(int) * NODE_COUNT);
        CHECK_CL(clEnqueueReadBuffer(
                    commands,
                    device_matched_rule_counts,
                    CL_TRUE,
                    0,
                    sizeof(int) * NODE_COUNT,
                    device_matched_rule_counts_mirror,
                    0,
                    NULL,
                    NULL));
        clFinish(commands);

        check_dom(host_ids,
                  host_tag_names,
                  host_class_counts,
                  host_classes,
                  device_matched_rules_mirror,
                  device_matched_rule_counts_mirror);

        clReleaseMemObject(device_ids);
        clReleaseMemObject(device_tag_names);
        clReleaseMemObject(device_class_counts);
        clReleaseMemObject(device_classes);
        clReleaseMemObject(device_stylesheet);
        clReleaseMemObject(device_matched_rules);
        clReleaseMemObject(device_matched_rule_counts);
    } else {
#if 0
        check_dom(host_dom, host_classes);
#endif
    }

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
}

#define PRINT_PLATFORM_INFO(platform, name) \
    do { \
        size_t size; \
        CHECK_CL(clGetPlatformInfo(platform, CL_PLATFORM_##name, 0, NULL, &size)); \
        char *result = (char *)malloc(size); \
        CHECK_CL(clGetPlatformInfo(platform, CL_PLATFORM_##name, size, result, NULL)); \
        fprintf(stderr, "%s: %s\n", #name, result); \
        free(result); \
    } while(0)

int main() {
    cl_platform_id platforms[10];
    cl_uint num_platforms;
    CHECK_CL(clGetPlatformIDs(10, platforms, &num_platforms));
    fprintf(stderr,
            "%d platform(s) available: first ID %lx\n",
            (int)num_platforms,
            (uintptr_t)platforms[0]);
    cl_platform_id platform = platforms[0];

    PRINT_PLATFORM_INFO(platform, PROFILE);
    PRINT_PLATFORM_INFO(platform, VERSION);
    PRINT_PLATFORM_INFO(platform, NAME);
    PRINT_PLATFORM_INFO(platform, VENDOR);
    PRINT_PLATFORM_INFO(platform, EXTENSIONS);

    time_t seed = time(NULL);

    go(platform, seed, CL_DEVICE_TYPE_CPU, MODE_COPYING);
    go(platform, seed, CL_DEVICE_TYPE_CPU, MODE_MAPPED);
    //go(platform, seed, CL_DEVICE_TYPE_GPU, MODE_COPYING);
#ifndef NO_SVM
    go(platform, seed, CL_DEVICE_TYPE_GPU, MODE_SVM);
#endif
    //go(platform, seed, CL_DEVICE_TYPE_CPU, MODE_MAPPED);
    return 0;
}

