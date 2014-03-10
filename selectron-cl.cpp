// OpenCL Selectron prototype
//
// Patrick Walton <pcwalton@mozilla.com>
//
// Copyright (c) 2014 Mozilla Corporation

#include "selectron.h"

#define WIN32_LEAN_AND_MEAN

#include <stdio.h>
#include <time.h>
#include <windows.h>
#include <CL/opencl.h>
#include <CL/opencl.h>

// Hack to allow stringification of macro expansion

#define XSTRINGIFY(s)   STRINGIFY(s)
#define STRINGIFY(s)    #s

uint64_t mach_absolute_time() {
    static LARGE_INTEGER freq = { 0, 0 };
    if (!freq.QuadPart)
        QueryPerformanceFrequency(&freq);

    LARGE_INTEGER time;
    QueryPerformanceCounter(&time);
    return time.QuadPart * 1000000000 / freq.QuadPart;
}

const char *selector_matching_kernel_source = "\n"
    XSTRINGIFY(STRUCT_CSS_PROPERTY) ";\n"
    XSTRINGIFY(STRUCT_CSS_RULE) ";\n"
    XSTRINGIFY(STRUCT_CSS_CUCKOO_HASH) ";\n"
    XSTRINGIFY(STRUCT_CSS_MATCHED_PROPERTY) ";\n"
    XSTRINGIFY(STRUCT_CSS_STYLESHEET_SOURCE) ";\n"
    XSTRINGIFY(STRUCT_CSS_STYLESHEET) ";\n"
    XSTRINGIFY(STRUCT_DOM_NODE(__global)) ";\n"
    "unsigned int css_rule_hash(unsigned int key, unsigned int seed) {\n"
    "   " XSTRINGIFY(CSS_RULE_HASH(key, seed)) ";\n"
    "}\n"
    "\n"
    "__global struct css_rule *css_cuckoo_hash_find_precomputed(\n"
    "       __global struct css_cuckoo_hash *hash,\n"
    "       int key,\n"
    "       int left_index,\n"
    "       int right_index) {\n"
    "   " XSTRINGIFY(CSS_CUCKOO_HASH_FIND_PRECOMPUTED(hash, key, left_index, right_index)) ";\n"
    "}\n"
    "\n"
    "void sort_selectors(struct css_matched_property *matched_properties, int length) {\n"
    "   " XSTRINGIFY(SORT_SELECTORS(matched_properties, length)) ";\n"
    "}\n"
    "\n"
    "__kernel void match_selectors(__global struct dom_node *first, \n"
    "                              __global struct css_stylesheet *stylesheet, \n"
    "                              __global struct css_property *properties) {\n"
    "   int index = get_global_id(0);\n"
    "   " XSTRINGIFY(SCRAMBLE_NODE_ID(index)) ";\n"
    "   " XSTRINGIFY(MATCH_SELECTORS_PRECOMPUTED(first,
                                                 stylesheet,
                                                 properties,
                                                 index,
                                                 css_cuckoo_hash_find_precomputed,
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

void abort_if_null(void *ptr, char *msg = "") {
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

#define MALLOC(context, use_svm, size) \
    ((use_svm) ? clSVMAllocAMD((context), 0, (size), 4) : malloc((size)))

void go(cl_platform_id platform, cl_device_type device_type, bool use_svm) {
    FIND_EXTENSION(clSVMAllocAMD, platform);
    FIND_EXTENSION(clSVMFreeAMD, platform);
    FIND_EXTENSION(clSetKernelArgSVMPointerAMD, platform);

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
        CL_HSA_ENABLED_AMD, (cl_context_properties)1,
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

    srand(time(NULL));

    // Create the rule tree on the host.
    struct css_stylesheet *host_stylesheet =
        (struct css_stylesheet *)MALLOC(context, use_svm, sizeof(struct css_stylesheet));
    abort_if_null(host_stylesheet, "failed to allocate host stylesheet");
    struct css_property *host_properties =
        (struct css_property *)MALLOC(context,
                                      use_svm,
                                      sizeof(struct css_property) * PROPERTY_COUNT);
    abort_if_null(host_properties, "failed to allocate host properties");
    int property_index = 0;
    create_stylesheet(host_stylesheet, host_properties, &property_index);

    // Create the DOM tree on the host.
    int global_count = 0;
    struct dom_node *host_dom =
        (struct dom_node *)MALLOC(context, use_svm, sizeof(struct dom_node) * NODE_COUNT);
    abort_if_null(host_dom, "failed to allocate host DOM");
    create_dom(host_dom, NULL, &global_count, 0);

    cl_mem device_dom, device_stylesheet, device_properties;
    if (!use_svm) {
        device_dom = clCreateBuffer(context,
                                    CL_MEM_READ_WRITE,
                                    sizeof(struct dom_node) * NODE_COUNT,
                                    NULL,
                                    NULL);
        abort_if_null(device_dom);

        device_stylesheet = clCreateBuffer(context,
                                           CL_MEM_READ_ONLY,
                                           sizeof(struct css_stylesheet),
                                           NULL,
                                           NULL);
        abort_if_null(device_stylesheet);

        device_properties = clCreateBuffer(context,
                                           CL_MEM_READ_ONLY,
                                           sizeof(struct css_property) * PROPERTY_COUNT,
                                           NULL,
                                           NULL);
        abort_if_null(device_properties);

        // Copy over the DOM tree.
        CHECK_CL(clEnqueueWriteBuffer(commands,
                                      device_dom,
                                      CL_TRUE,
                                      0,
                                      sizeof(struct dom_node) * NODE_COUNT,
                                      host_dom,
                                      0,
                                      NULL,
                                      NULL));

        // Copy over the rule tree.
        CHECK_CL(clEnqueueWriteBuffer(commands,
                                      device_stylesheet,
                                      CL_TRUE,
                                      0,
                                      sizeof(struct css_stylesheet),
                                      host_stylesheet,
                                      0,
                                      NULL,
                                      NULL));

        // Copy over the properties.
        CHECK_CL(clEnqueueWriteBuffer(commands,
                                      device_properties,
                                      CL_TRUE,
                                      0,
                                      sizeof(struct css_property) * PROPERTY_COUNT,
                                      host_properties,
                                      0,
                                      NULL,
                                      NULL));
    }

    // Set the arguments to the kernel.
    if (!use_svm) {
        CHECK_CL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_dom));
        CHECK_CL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_stylesheet));
        CHECK_CL(clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_properties));
    } else {
        CHECK_CL(clSetKernelArgSVMPointerAMD(kernel, 0, host_dom));
        CHECK_CL(clSetKernelArgSVMPointerAMD(kernel, 1, host_stylesheet));
        CHECK_CL(clSetKernelArgSVMPointerAMD(kernel, 2, host_properties));
    }

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

    uint64_t start = mach_absolute_time();
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
    double elapsed = (double)(mach_absolute_time() - start) / 1000000.0;
    report_timing(device_name, elapsed, false, use_svm);

    if (!use_svm) {
        // Retrieve the DOM.
        struct dom_node *device_dom_mirror = (struct dom_node *)malloc(sizeof(struct dom_node) *
                                                                       NODE_COUNT);
        CHECK_CL(clEnqueueReadBuffer(commands,
                                     device_dom,
                                     CL_TRUE,
                                     0,
                                     sizeof(struct dom_node) * NODE_COUNT,
                                     device_dom_mirror,
                                     0,
                                     NULL,
                                     NULL));
        clFinish(commands);

        check_dom(device_dom_mirror);

        clReleaseMemObject(device_dom);
        clReleaseMemObject(device_stylesheet);
        clReleaseMemObject(device_properties);
    } else {
        check_dom(host_dom);
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
            "%d platform(s) available: first ID %d\n",
            (int)num_platforms,
            (int)platforms[0]);
    cl_platform_id platform = platforms[0];

    PRINT_PLATFORM_INFO(platform, PROFILE);
    PRINT_PLATFORM_INFO(platform, VERSION);
    PRINT_PLATFORM_INFO(platform, NAME);
    PRINT_PLATFORM_INFO(platform, VENDOR);
    PRINT_PLATFORM_INFO(platform, EXTENSIONS);

    fprintf(stderr, "Size of a DOM node: %d\n", (int)sizeof(struct dom_node));

    go(platform, CL_DEVICE_TYPE_GPU, true);
    go(platform, CL_DEVICE_TYPE_GPU, false);
    go(platform, CL_DEVICE_TYPE_CPU, false);
    return 0;
}
