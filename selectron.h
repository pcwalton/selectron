// Selectron prototype
//
// Patrick Walton <pcwalton@mozilla.com>
//
// Copyright (c) 2014 Mozilla Corporation

#ifndef SELECTRON_H
#define SELECTRON_H

#include <mach/mach.h>
#include <mach/mach_time.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define NODE_COUNT 102400
#define THREAD_COUNT 1024
#define MAX_DOM_DEPTH 10

#define ESTIMATED_PARALLEL_SPEEDUP  2.7

#define RULE_ID_MAX 25
#define NODE_ID_MAX 50
#define RULE_TAG_NAME_MAX 8
#define NODE_TAG_NAME_MAX 12

#define CSS_SELECTOR_TYPE_NONE      0
#define CSS_SELECTOR_TYPE_ID        1
#define CSS_SELECTOR_TYPE_TAG_NAME  2

#define HASH_SIZE   4096

#ifdef MAX
#undef MAX
#endif
#define MAX(a,b)    ((a) > (b) ? (a) : (b))

#define LEFT_SEED   12345
#define RIGHT_SEED  67890

// FIXME(pcwalton): This is not really implemented properly; it should resize the table.

#define STRUCT_CSS_RULE \
    struct css_rule { \
        int type; \
        int value; \
    }

STRUCT_CSS_RULE;

#define STRUCT_CSS_CUCKOO_HASH \
    struct css_cuckoo_hash { \
        int left_seed; \
        int right_seed; \
        struct css_rule left[HASH_SIZE]; \
        struct css_rule right[HASH_SIZE]; \
    }

STRUCT_CSS_CUCKOO_HASH;

#define CSS_RULE_HASH(key, seed) \
    do {\
        unsigned int hash = 2166136261; \
        hash = hash ^ seed; \
        hash = hash * 16777619; \
        hash = hash ^ key; \
        hash = hash * 16777619; \
        return hash; \
    } while(0)

uint32_t css_rule_hash(uint32_t key, uint32_t seed) {
    CSS_RULE_HASH(key, seed);
}

void css_cuckoo_hash_reset(struct css_cuckoo_hash *hash) {
    for (int i = 0; i < HASH_SIZE; i++) {
        hash->left[i].type = 0;
        hash->right[i].type = 0;
    }
}

void css_cuckoo_hash_reseed(struct css_cuckoo_hash *hash) {
    hash->left_seed = rand();
    hash->right_seed = rand();
}

void css_cuckoo_hash_init(struct css_cuckoo_hash *hash) {
    css_cuckoo_hash_reset(hash);
    css_cuckoo_hash_reseed(hash);
}

void css_cuckoo_hash_rehash(struct css_cuckoo_hash *hash) {
    fprintf(stderr, "rehash unimplemented\n");
    abort();
}

bool css_cuckoo_hash_insert_internal(struct css_cuckoo_hash *hash,
                                     struct css_rule *rule,
                                     bool right) {
    int hashval = css_rule_hash(rule->value, right ? RIGHT_SEED : LEFT_SEED);
    int index = hashval % HASH_SIZE;
    struct css_rule *list = right ? hash->right : hash->left;
    if (list[index].type != 0) {
        if (!css_cuckoo_hash_insert_internal(hash, &list[index], !right))
            return false;
    }

    list[index] = *rule;
    return true;
}

void css_cuckoo_hash_insert(struct css_cuckoo_hash *hash, struct css_rule *rule) {
    if (css_cuckoo_hash_insert_internal(hash, rule, false))
        return;
    css_cuckoo_hash_reseed(hash);
    css_cuckoo_hash_rehash(hash);
    if (css_cuckoo_hash_insert_internal(hash, rule, false))
        return;
    fprintf(stderr, "rehashing failed\n");
    abort();
}

#define CSS_CUCKOO_HASH_FIND(hash, key, hashfn) \
    do {\
        int left_index = hashfn(key, LEFT_SEED) % HASH_SIZE; \
        if (hash->left[left_index].type != 0 && hash->left[left_index].value == key) \
            return &hash->left[left_index]; \
        int right_index = hashfn(key, RIGHT_SEED) % HASH_SIZE; \
        if (hash->right[right_index].type != 0 && hash->right[right_index].value == key) \
            return &hash->right[right_index]; \
        return NULL; \
    } while(0)

#define CSS_CUCKOO_HASH_FIND_PRECOMPUTED(hash, key, left_index, right_index) \
    do {\
        if (hash->left[left_index].type != 0 && hash->left[left_index].value == key) \
            return &hash->left[left_index]; \
        if (hash->right[right_index].type != 0 && hash->right[right_index].value == key) \
            return &hash->right[right_index]; \
        return NULL; \
    } while(0)

struct css_rule *css_cuckoo_hash_find(struct css_cuckoo_hash *hash, int32_t key) {
    CSS_CUCKOO_HASH_FIND(hash, key, css_rule_hash);
}

struct css_rule *css_cuckoo_hash_find_precomputed(struct css_cuckoo_hash *hash,
                                                  int32_t key,
                                                  int32_t left_index,
                                                  int32_t right_index) {
    CSS_CUCKOO_HASH_FIND_PRECOMPUTED(hash, key, left_index, right_index);
}

#define STRUCT_CSS_STYLESHEET_SOURCE \
    struct css_stylesheet_source { \
        struct css_cuckoo_hash ids; \
        struct css_cuckoo_hash tag_names; \
    }

STRUCT_CSS_STYLESHEET_SOURCE;

#define STRUCT_CSS_STYLESHEET \
    struct css_stylesheet { \
        struct css_stylesheet_source author; \
        struct css_stylesheet_source user_agent; \
    }

STRUCT_CSS_STYLESHEET;

#define STRUCT_CSS_MATCHED_PROPERTY \
    struct css_matched_property { \
        short type; \
        short specificity; \
        int value; \
    }

STRUCT_CSS_MATCHED_PROPERTY;

#define STRUCT_DOM_NODE(qualifier) \
    struct dom_node { \
        qualifier struct dom_node *parent; \
        qualifier struct dom_node *first_child; \
        qualifier struct dom_node *last_child; \
        qualifier struct dom_node *next_sibling; \
        qualifier struct dom_node *prev_sibling; \
        int id; \
        int tag_name; \
        int applicable_declaration_count; \
        struct css_matched_property applicable_declarations[16]; \
        int pad[24]; \
    }

STRUCT_DOM_NODE();

// Insertion sort.
#define SORT_SELECTORS(node) \
    do { \
        for (int i = 1; i < node->applicable_declaration_count; i++) { \
            for (int j = i; \
                    j > 0 && \
                    node->applicable_declarations[j - 1].specificity > \
                    node->applicable_declarations[j].specificity; \
                    j--) { \
                struct css_matched_property tmp = node->applicable_declarations[j - 1]; \
                node->applicable_declarations[j - 1] = node->applicable_declarations[j]; \
                node->applicable_declarations[j] = tmp; \
            } \
        } \
    } while(0)

#define MATCH_SELECTORS_HASH(node, hash, findfn) \
    do {\
        const css_rule *__restrict__ rule = findfn(hash, node->id); \
        if (rule != NULL) \
            node->applicable_declarations[node->applicable_declaration_count++] = *rule; \
    } while(0)

#define MATCH_SELECTORS(first, stylesheet, index, findfn) \
    do {\
        dom_node *node = &first[index]; \
        node->applicable_declaration_count = 0; \
        MATCH_SELECTORS_HASH(node, &stylesheet->author.ids, findfn); \
        MATCH_SELECTORS_HASH(node, &stylesheet->author.tag_names, findfn); \
        MATCH_SELECTORS_HASH(node, &stylesheet->user_agent.ids, findfn); \
        MATCH_SELECTORS_HASH(node, &stylesheet->user_agent.tag_names, findfn); \
    } while(0)

#define MATCH_SELECTORS_HASH_PRECOMPUTED(node, \
                                         hash, \
                                         spec, \
                                         findfn, \
                                         left_index, \
                                         right_index, \
                                         qualifier) \
    do {\
        qualifier const struct css_rule *__restrict__ rule = findfn(hash, \
                                                                    node->id, \
                                                                    left_index, \
                                                                    right_index); \
        if (rule != NULL) { \
            int index = node->applicable_declaration_count++; \
            node->applicable_declarations[index].type = rule->type; \
            node->applicable_declarations[index].specificity = spec; \
            node->applicable_declarations[index].value = rule->value; \
        } \
    } while(0)

#define MATCH_SELECTORS_PRECOMPUTED(first, stylesheet, index, findfn, hashfn, sortfn, qualifier) \
    do {\
        qualifier struct dom_node *node = &first[index]; \
        node->applicable_declaration_count = 0; \
        int left_id_index = hashfn(node->id, LEFT_SEED) % HASH_SIZE; \
        int right_id_index = hashfn(node->id, RIGHT_SEED) % HASH_SIZE; \
        int left_tag_name_index = hashfn(node->tag_name, LEFT_SEED) % HASH_SIZE; \
        int right_tag_name_index = hashfn(node->tag_name, RIGHT_SEED) % HASH_SIZE; \
        MATCH_SELECTORS_HASH_PRECOMPUTED(node, \
                                         &stylesheet->author.ids, \
                                         0, \
                                         findfn, \
                                         left_id_index, \
                                         right_id_index, \
                                         qualifier); \
        MATCH_SELECTORS_HASH_PRECOMPUTED(node, \
                                         &stylesheet->author.tag_names, \
                                         0, \
                                         findfn, \
                                         left_tag_name_index, \
                                         right_tag_name_index, \
                                         qualifier); \
        MATCH_SELECTORS_HASH_PRECOMPUTED(node, \
                                         &stylesheet->user_agent.ids, \
                                         1, \
                                         findfn, \
                                         left_id_index, \
                                         right_id_index, \
                                         qualifier); \
        MATCH_SELECTORS_HASH_PRECOMPUTED(node, \
                                         &stylesheet->user_agent.tag_names, \
                                         1, \
                                         findfn, \
                                         left_tag_name_index, \
                                         right_tag_name_index, \
                                         qualifier); \
        sortfn(node); \
    } while(0)

void sort_selectors(struct dom_node *node) {
    SORT_SELECTORS(node);
}

void match_selectors(struct dom_node *first, struct css_stylesheet *stylesheet, int32_t index) {
#if 0
    MATCH_SELECTORS(first, stylesheet, index, css_cuckoo_hash_find);
#endif
    MATCH_SELECTORS_PRECOMPUTED(first,
                                stylesheet,
                                index,
                                css_cuckoo_hash_find_precomputed,
                                css_rule_hash,
                                sort_selectors,
                                );
}

void create_stylesheet(struct css_stylesheet *stylesheet) {
    css_cuckoo_hash_init(&stylesheet->author.ids);
    css_cuckoo_hash_init(&stylesheet->author.tag_names);
    css_cuckoo_hash_init(&stylesheet->user_agent.ids);
    css_cuckoo_hash_init(&stylesheet->user_agent.tag_names);

    for (int i = 0; i < RULE_ID_MAX; i++) {
        struct css_rule rule = { CSS_SELECTOR_TYPE_ID, i };
        css_cuckoo_hash_insert(&stylesheet->author.ids, &rule);
    }
    for (int i = 0; i < RULE_ID_MAX; i++) {
        struct css_rule rule = { CSS_SELECTOR_TYPE_ID, i };
        css_cuckoo_hash_insert(&stylesheet->user_agent.ids, &rule);
    }
    for (int i = 0; i < RULE_TAG_NAME_MAX; i++) {
        struct css_rule rule = { CSS_SELECTOR_TYPE_TAG_NAME, i };
        css_cuckoo_hash_insert(&stylesheet->author.tag_names, &rule);
    }
    for (int i = 0; i < RULE_TAG_NAME_MAX; i++) {
        struct css_rule rule = { CSS_SELECTOR_TYPE_TAG_NAME, i };
        css_cuckoo_hash_insert(&stylesheet->user_agent.tag_names, &rule);
    }
}

void create_dom(struct dom_node *dest, struct dom_node *parent, int *global_count, int depth) {
    if (*global_count == NODE_COUNT)
        return;
    if (depth == MAX_DOM_DEPTH)
        return;

    struct dom_node *node = &dest[(*global_count)++];
    node->id = rand() % NODE_ID_MAX;
    node->tag_name = rand() % NODE_TAG_NAME_MAX;
    node->applicable_declaration_count = 0;

    node->first_child = node->last_child = node->next_sibling = NULL;
    if ((node->parent = parent) != NULL) {
        if (node->parent->last_child != NULL) {
            node->prev_sibling = node->parent->last_child;
            node->prev_sibling->next_sibling = node->parent->last_child = node;
        } else {
            node->parent->first_child = node->parent->last_child = node;
            node->prev_sibling = NULL;
        }
    }

    int child_count = rand() % (NODE_COUNT / 100);
    for (int i = 0; i < child_count; i++)
        create_dom(dest, node, global_count, depth + 1);
}

void munge_dom_pointers(struct dom_node *node, ptrdiff_t offset) {
    for (int i = 0; i < NODE_COUNT; i++) {
        node->parent = (struct dom_node *)((ptrdiff_t)node->parent + offset);
        node->first_child = (struct dom_node *)((ptrdiff_t)node->first_child + offset);
        node->last_child = (struct dom_node *)((ptrdiff_t)node->last_child + offset);
        node->next_sibling = (struct dom_node *)((ptrdiff_t)node->next_sibling + offset);
        node->prev_sibling = (struct dom_node *)((ptrdiff_t)node->prev_sibling + offset);
    }
}

void check_dom(struct dom_node *node) {
    for (int i = 0; i < 20; i++) {
        printf("%d -> %d\n", node[i].id, node[i].applicable_declaration_count);
    }
}

// Frame tree

struct frame {
    struct dom_node *node;
    int32_t type;
};

void create_frame(struct dom_node *first, int i) {
    struct frame *frame = (struct frame *)malloc(sizeof(struct frame));
    frame->node = &first[i];
    frame->type = 0;
}

// Misc.

void report_timing(const char *name, double ms, bool report_parallel_estimate) {
    if (report_parallel_estimate) {
        fprintf(stderr,
                "%s: %g ms (parallel estimate %g ms)\n",
                name,
                ms,
                ms / ESTIMATED_PARALLEL_SPEEDUP);
        return;
    }
    fprintf(stderr, "%s: %g ms\n", name, ms);
}

#endif

