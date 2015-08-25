// C bitmap control Selectron prototype
//
// Patrick Walton <pcwalton@mozilla.com>
//
// Copyright (c) 2015 Mozilla Corporation

#include "bitmap-selectron.h"

#include <sys/time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

uint32_t hash(uint32_t key, uint32_t seed) {
    unsigned int hash = 2166136261;
    hash = hash ^ seed;
    hash = hash * 16777619;
    hash = hash ^ key;
    hash = hash * 16777619;
    return hash;
}

void match_selectors(const int *ids, const uint16_t *stylesheet, uint16_t *matched_rules)
        __attribute__((noinline)) {
    for (int node = 0; node < NODE_COUNT; node++)
        matched_rules[node] = stylesheet[hash(ids[node], 12345) % MAX_ID];
}

int main(int argc, char **argv) {
    srand(time(NULL));

    // Create the stylesheet.
    uint16_t *stylesheet = (uint16_t *)malloc(MAX_ID);
    for (int i = 0; i < MAX_ID; i++)
        stylesheet[i] = rand();

    // Create the DOM on the host.
    int *ids = (int *)malloc(sizeof(int) * NODE_COUNT);
    create_dom(ids);

    // Allocate the matched rules.
    uint16_t *matched_rules = (uint16_t *)malloc(sizeof(uint16_t) * NODE_COUNT);

    // Time and execute the kernel.
    struct timeval start;
    gettimeofday(&start, NULL);
    match_selectors(ids, stylesheet, matched_rules);
    struct timeval end;
    gettimeofday(&end, NULL);

    fprintf(stderr,
            "Elapsed time: %f ms\n",
            ((double)end.tv_sec - (double)start.tv_sec) * 1000.0 +
            ((double)end.tv_usec - (double)start.tv_usec) / 1000.0);
}

