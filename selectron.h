// Selectron prototype
//
// Patrick Walton <pcwalton@mozilla.com>
//
// Copyright (c) 2014 Mozilla Corporation

#ifndef SELECTRON_H
#define SELECTRON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define RULE_ID_MAX             25
#define NODE_ID_MAX             50
#define RULE_TAG_NAME_MAX       8
#define NODE_TAG_NAME_MAX       12
#define RULE_CLASS_MAX          25
#define NODE_CLASS_MAX          50
#define NODE_CLASS_COUNT_MAX    5

#define NODE_COUNT              (1024 * 100)
#define CLASS_COUNT             ((NODE_COUNT) * (NODE_CLASS_COUNT_MAX))
#define THREAD_COUNT            128
#define PROPERTY_COUNT          512
#define MAX_DOM_DEPTH           10
#define MAX_PROPERTIES_PER_RULE 5
#define MAX_STYLE_PROPERTIES    32
#define MAX_PROPERTY_VALUE      8
#define DOM_PADDING_PRE         0
#define DOM_PADDING_POST        0
#define MAX_MATCHED_RULES       16

#define ESTIMATED_PARALLEL_SPEEDUP  2.7

#define CSS_SELECTOR_TYPE_NONE      0
#define CSS_SELECTOR_TYPE_ID        1
#define CSS_SELECTOR_TYPE_TAG_NAME  2
#define CSS_SELECTOR_TYPE_CLASS     3

#define HASH_SIZE   256

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
        int property_index; \
        int property_count; \
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

#define CSS_CUCKOO_HASH_FIND(hash, key, left_index, right_index) \
    do {\
        if (hash->left[left_index].type != 0 && hash->left[left_index].value == key) \
            return &hash->left[left_index]; \
        if (hash->right[right_index].type != 0 && hash->right[right_index].value == key) \
            return &hash->right[right_index]; \
        return 0; \
    } while(0)

struct css_rule *css_cuckoo_hash_find(struct css_cuckoo_hash *hash,
                                      int32_t key,
                                      int32_t left_index,
                                      int32_t right_index) {
    CSS_CUCKOO_HASH_FIND(hash, key, left_index, right_index);
}

#define STRUCT_CSS_STYLESHEET_SOURCE \
    struct css_stylesheet_source { \
        struct css_cuckoo_hash ids; \
        struct css_cuckoo_hash tag_names; \
        struct css_cuckoo_hash classes; \
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
        int specificity; \
        int property_index; \
        int property_count; \
    }

STRUCT_CSS_MATCHED_PROPERTY;

// Insertion sort.
#define SORT_SELECTORS(matched_properties, count) \
    do { \
        for (int i = 1; i < count; i++) { \
            for (int j = i; \
                    j > 0 && \
                    matched_properties[j - 1].specificity > \
                    matched_properties[j].specificity; \
                    j--) { \
                struct css_matched_property tmp = matched_properties[j - 1]; \
                matched_properties[j - 1] = matched_properties[j]; \
                matched_properties[j] = tmp; \
            } \
        } \
    } while(0)

#define MATCH_SELECTORS_HASH(value, \
                             hash, \
                             spec, \
                             findfn, \
                             left_index, \
                             right_index, \
                             count, \
                             matched_rules, \
                             qualifier) \
    do {\
        qualifier const struct css_rule * rule = \
            findfn(hash, value, left_index, right_index); \
        if (rule != 0) { \
            int matched_rule_index = count++; \
            matched_rules[matched_rule_index] = rule; \
        } \
    } while(0)

#define MATCH_SELECTORS(ids, \
                        tag_names, \
                        class_counts, \
                        classes, \
                        stylesheet, \
                        matched_rule_buffer, \
                        matched_rule_counts, \
                        index, \
                        findfn, \
                        hashfn, \
                        sortfn, \
                        qualifier) \
    do { \
        int count = 0; \
        qualifier const struct css_rule *local_matched_rules[MAX_MATCHED_RULES]; \
        int left_id_index = hashfn(ids[index], LEFT_SEED) % HASH_SIZE; \
        int right_id_index = hashfn(ids[index], RIGHT_SEED) % HASH_SIZE; \
        int left_tag_name_index = hashfn(tag_names[index], LEFT_SEED) % HASH_SIZE; \
        int right_tag_name_index = hashfn(tag_names[index], RIGHT_SEED) % HASH_SIZE; \
        MATCH_SELECTORS_HASH(ids[index], \
                             &stylesheet->author.ids, \
                             0, \
                             findfn, \
                             left_id_index, \
                             right_id_index, \
                             count, \
                             local_matched_rules, \
                             qualifier); \
        MATCH_SELECTORS_HASH(tag_names[index], \
                             &stylesheet->author.tag_names, \
                             0, \
                             findfn, \
                             left_tag_name_index, \
                             right_tag_name_index, \
                             count, \
                             local_matched_rules, \
                             qualifier); \
        MATCH_SELECTORS_HASH(ids[index], \
                             &stylesheet->user_agent.ids, \
                             1, \
                             findfn, \
                             left_id_index, \
                             right_id_index, \
                             count, \
                             local_matched_rules, \
                             qualifier); \
        MATCH_SELECTORS_HASH(tag_names[index], \
                             &stylesheet->user_agent.tag_names, \
                             1, \
                             findfn, \
                             left_tag_name_index, \
                             right_tag_name_index, \
                             count, \
                             local_matched_rules, \
                             qualifier); \
        int class_count = class_counts[index]; \
        for (int i = 0; i < class_count; i++) { \
            int klass = classes[i * NODE_COUNT + index]; \
            int left_class_index = hashfn(klass, LEFT_SEED) % HASH_SIZE; \
            int right_class_index = hashfn(klass, RIGHT_SEED) % HASH_SIZE; \
            MATCH_SELECTORS_HASH(klass, \
                                 &stylesheet->author.classes, \
                                 0, \
                                 findfn, \
                                 left_class_index, \
                                 right_class_index, \
                                 count, \
                                 local_matched_rules, \
                                 qualifier); \
            MATCH_SELECTORS_HASH(klass, \
                                 &stylesheet->user_agent.classes, \
                                 0, \
                                 findfn, \
                                 left_class_index, \
                                 right_class_index, \
                                 count, \
                                 local_matched_rules, \
                                 qualifier); \
        } \
        count = min(count, MAX_MATCHED_RULES); \
        matched_rule_counts[index] = count; \
        for (int i = 0; i < count; i++) { \
            qualifier const struct css_rule *matched = local_matched_rules[i]; \
            matched_rule_buffer[index + NODE_COUNT * i] = matched; \
        } \
    } while(0)

#if 0
#define SCRAMBLE_NODE_ID(n) \
    do { \
        int nibble0 = (n & 0xf); \
        int nibble1 = (n & 0xf0) >> 4; \
        int rest = (n & 0xffffff00); \
        n = (rest | (nibble0 << 4) | nibble1); \
    } while(0)
//#define SCRAMBLE_NODE_ID(n)
#endif

void sort_selectors(struct css_matched_property *matched_properties, int length) {
   SORT_SELECTORS(matched_properties, length);
}

#if 0
void match_selectors(struct dom_node *first,
                     struct css_stylesheet *stylesheet,
                     struct css_property *properties,
                     int *classes,
                     int32_t index) {
    MATCH_SELECTORS(first,
                    stylesheet,
                    properties,
                    classes,
                    index,
                    css_cuckoo_hash_find,
                    css_rule_hash,
                    sort_selectors,
                    );
}
#endif

void create_properties(struct css_rule *rule, int *property_index) {
    int n_properties = rand() % MAX_PROPERTIES_PER_RULE;
    rule->property_index = *property_index;
    rule->property_count = n_properties;
    *property_index += n_properties;
}

void init_rule_hash(struct css_cuckoo_hash *hash,
                    int *property_index,
                    int type,
                    int max) {
    css_cuckoo_hash_init(hash);
    for (int i = 0; i < max; i++) {
        struct css_rule rule = { type, i, 0, 0 };
        create_properties(&rule, property_index);
        css_cuckoo_hash_insert(hash, &rule);
    }
}

void create_stylesheet(struct css_stylesheet *stylesheet, int *property_index) {
    init_rule_hash(&stylesheet->author.ids,
                   property_index,
                   CSS_SELECTOR_TYPE_ID,
                   RULE_ID_MAX);
    init_rule_hash(&stylesheet->author.tag_names,
                   property_index,
                   CSS_SELECTOR_TYPE_TAG_NAME,
                   RULE_TAG_NAME_MAX);
    init_rule_hash(&stylesheet->author.classes,
                   property_index,
                   CSS_SELECTOR_TYPE_TAG_NAME,
                   RULE_CLASS_MAX);
    init_rule_hash(&stylesheet->user_agent.ids,
                   property_index,
                   CSS_SELECTOR_TYPE_ID,
                   RULE_ID_MAX);
    init_rule_hash(&stylesheet->user_agent.tag_names,
                   property_index,
                   CSS_SELECTOR_TYPE_TAG_NAME,
                   RULE_TAG_NAME_MAX);
    init_rule_hash(&stylesheet->user_agent.classes,
                   property_index,
                   CSS_SELECTOR_TYPE_TAG_NAME,
                   RULE_CLASS_MAX);
}

void create_dom(int *dest_ids,
                int *dest_tag_names,
                int *dest_class_counts,
                int *dest_classes,
                int *class_count,
                int *global_count,
                int depth) {
    if (*global_count == NODE_COUNT)
        return;
    if (depth == MAX_DOM_DEPTH)
        return;

    int index = (*global_count)++;
    dest_ids[index] = rand() % NODE_ID_MAX;
    dest_tag_names[index] = rand() % NODE_TAG_NAME_MAX;

    dest_class_counts[index] = rand() % NODE_CLASS_COUNT_MAX;
    for (int i = 0; i < dest_class_counts[index]; i++)
        dest_classes[index + i * NODE_COUNT] = rand() % NODE_CLASS_MAX;

#if 0
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
#endif

    int child_count = rand() % (NODE_COUNT / 100);
    for (int i = 0; i < child_count; i++)
        create_dom(dest_ids,
                   dest_tag_names,
                   dest_class_counts,
                   dest_classes,
                   class_count,
                   global_count,
                   depth + 1);
}

#if 0
void munge_dom_pointers(struct dom_node *node, ptrdiff_t offset) {
    for (int i = 0; i < NODE_COUNT; i++) {
        node->parent = (struct dom_node *)((ptrdiff_t)node->parent + offset);
        node->first_child = (struct dom_node *)((ptrdiff_t)node->first_child + offset);
        node->last_child = (struct dom_node *)((ptrdiff_t)node->last_child + offset);
        node->next_sibling = (struct dom_node *)((ptrdiff_t)node->next_sibling + offset);
        node->prev_sibling = (struct dom_node *)((ptrdiff_t)node->prev_sibling + offset);
    }
}
#endif

void check_dom(const int *ids,
               const int *tag_names,
               const int *class_counts,
               const int *classes,
               const css_rule **matched_rules,
               const int *matched_rule_counts) {
    for (int i = 0; i < 20; i++) {
        printf("%d (id %d; tag %d; classes", i, ids[i], tag_names[i]);
        for (int j = 0; j < class_counts[i]; j++) {
            printf("%s%d", j == 0 ? " " : ", ", classes[i + j * NODE_COUNT]);
        }
        printf(") -> ");
        for (int j = 0; j < matched_rule_counts[i]; j++) {
            printf("%p ", matched_rules[i + j * NODE_COUNT]);
        }
        printf("\n");
    }
}

// Frame tree

#if 0
struct frame {
    struct dom_node *node;
    int32_t type;
};

void create_frame(struct dom_node *first, int i) {
    struct frame *frame = (struct frame *)malloc(sizeof(struct frame));
    frame->node = &first[i];
    frame->type = 0;
}
#endif

// Misc.

#define MODE_COPYING    0
#define MODE_MAPPED     1
#define MODE_SVM        2

const char *mode_to_string(int mode) {
    switch (mode) {
    case MODE_COPYING:  return " (copying)";
    case MODE_MAPPED:   return " (mapped)";
    default:            return " (SVM)";
    }
}

void report_timing(const char *name,
                   const char *operation,
                   double ms,
                   bool report_parallel_estimate,
                   int mode) {
    if (report_parallel_estimate) {
        fprintf(stderr,
                "%s%s %s: %g ms (parallel estimate %g ms)\n",
                name,
                mode_to_string(mode),
                operation,
                ms,
                ms / ESTIMATED_PARALLEL_SPEEDUP);
        return;
    }
    fprintf(stderr, "%s%s %s: %g ms\n", name, mode_to_string(mode), operation, ms);
}

#endif

