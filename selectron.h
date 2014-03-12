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
#define THREAD_COUNT            1024
#define PROPERTY_COUNT          512
#define MAX_DOM_DEPTH           10
#define MAX_PROPERTIES_PER_RULE 5
#define MAX_STYLE_PROPERTIES    32
#define MAX_PROPERTY_VALUE      8
#define DOM_PADDING_PRE         0
#define DOM_PADDING_POST        0

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

#define STRUCT_CSS_PROPERTY \
    struct css_property { \
        int name; \
        int value; \
    }

STRUCT_CSS_PROPERTY;

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

#define STRUCT_DOM_NODE(qualifier) \
    struct dom_node { \
        struct dom_node *parent; \
        int pad_pre[DOM_PADDING_PRE]; \
        int id; \
        int tag_name; \
        int class_count; \
        int first_class; \
        int style[MAX_STYLE_PROPERTIES]; \
        int pad_post[DOM_PADDING_POST]; \
    }

STRUCT_DOM_NODE();

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
                             matched_properties, \
                             qualifier) \
    do {\
        qualifier const struct css_rule * rule = findfn(hash, \
                                                                    value, \
                                                                    left_index, \
                                                                    right_index); \
        if (rule != 0) { \
            int index = count++; \
            matched_properties[index].specificity = spec; \
            matched_properties[index].property_index = rule->property_index; \
            matched_properties[index].property_count = rule->property_count; \
        } \
    } while(0)

#define MATCH_SELECTORS(first, \
                        stylesheet, \
                        properties, \
                        classes, \
                        index, \
                        findfn, \
                        hashfn, \
                        sortfn, \
                        qualifier) \
    do {\
        qualifier struct dom_node *node = &first[index]; \
        int count = 0; \
        struct css_matched_property matched_properties[16]; \
        int left_id_index = hashfn(node->id, LEFT_SEED) % HASH_SIZE; \
        int right_id_index = hashfn(node->id, RIGHT_SEED) % HASH_SIZE; \
        int left_tag_name_index = hashfn(node->tag_name, LEFT_SEED) % HASH_SIZE; \
        int right_tag_name_index = hashfn(node->tag_name, RIGHT_SEED) % HASH_SIZE; \
        MATCH_SELECTORS_HASH(node->id, \
                             &stylesheet->author.ids, \
                             0, \
                             findfn, \
                             left_id_index, \
                             right_id_index, \
                             count, \
                             matched_properties, \
                             qualifier); \
        MATCH_SELECTORS_HASH(node->tag_name, \
                             &stylesheet->author.tag_names, \
                             0, \
                             findfn, \
                             left_tag_name_index, \
                             right_tag_name_index, \
                             count, \
                             matched_properties, \
                             qualifier); \
        MATCH_SELECTORS_HASH(node->id, \
                             &stylesheet->user_agent.ids, \
                             1, \
                             findfn, \
                             left_id_index, \
                             right_id_index, \
                             count, \
                             matched_properties, \
                             qualifier); \
        MATCH_SELECTORS_HASH(node->tag_name, \
                             &stylesheet->user_agent.tag_names, \
                             1, \
                             findfn, \
                             left_tag_name_index, \
                             right_tag_name_index, \
                             count, \
                             matched_properties, \
                             qualifier); \
        int class_count = node->class_count; \
        int first_class = node->first_class; \
        for (int i = 0; i < class_count; i++) { \
            int klass = classes[first_class + i]; \
            int left_class_index = hashfn(klass, LEFT_SEED) % HASH_SIZE; \
            int right_class_index = hashfn(klass, RIGHT_SEED) % HASH_SIZE; \
            MATCH_SELECTORS_HASH(klass, \
                                 &stylesheet->author.classes, \
                                 0, \
                                 findfn, \
                                 left_class_index, \
                                 right_class_index, \
                                 count, \
                                 matched_properties, \
                                 qualifier); \
            MATCH_SELECTORS_HASH(klass, \
                                 &stylesheet->user_agent.classes, \
                                 0, \
                                 findfn, \
                                 left_class_index, \
                                 right_class_index, \
                                 count, \
                                 matched_properties, \
                                 qualifier); \
        } \
        sortfn(matched_properties, count); \
        for (int i = 0; i < count; i++) { \
            struct css_matched_property *matched = &matched_properties[i]; \
            int count = matched->property_count; \
            for (int j = 0; j < count; j++) { \
                qualifier struct css_property *property = \
                    &properties[matched->property_index + j]; \
                node->style[property->name] = property->value; \
            } \
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

void create_properties(struct css_rule *rule,
                       struct css_property *properties,
                       int *property_index) {
    int n_properties = rand() % MAX_PROPERTIES_PER_RULE;
    rule->property_index = *property_index;
    rule->property_count = n_properties;

    for (int i = 0; i < n_properties; i++) {
        properties[*property_index].name = rand() % MAX_STYLE_PROPERTIES;
        properties[*property_index].value = rand() % MAX_PROPERTY_VALUE;
        if (++*property_index >= PROPERTY_COUNT) {
            fprintf(stderr, "out of properties, try increasing PROPERTY_COUNT\n");
            exit(1);
        }
    }
}

void init_rule_hash(struct css_cuckoo_hash *hash,
                    struct css_property *properties,
                    int *property_index,
                    int type,
                    int max) {
    css_cuckoo_hash_init(hash);
    for (int i = 0; i < max; i++) {
        struct css_rule rule = { type, i, 0, 0 };
        create_properties(&rule, properties, property_index);
        css_cuckoo_hash_insert(hash, &rule);
    }
}

void create_stylesheet(struct css_stylesheet *stylesheet,
                       struct css_property *properties,
                       int *property_index) {
    init_rule_hash(&stylesheet->author.ids,
                   properties,
                   property_index,
                   CSS_SELECTOR_TYPE_ID,
                   RULE_ID_MAX);
    init_rule_hash(&stylesheet->author.tag_names,
                   properties,
                   property_index,
                   CSS_SELECTOR_TYPE_TAG_NAME,
                   RULE_TAG_NAME_MAX);
    init_rule_hash(&stylesheet->author.classes,
                   properties,
                   property_index,
                   CSS_SELECTOR_TYPE_TAG_NAME,
                   RULE_CLASS_MAX);
    init_rule_hash(&stylesheet->user_agent.ids,
                   properties,
                   property_index,
                   CSS_SELECTOR_TYPE_ID,
                   RULE_ID_MAX);
    init_rule_hash(&stylesheet->user_agent.tag_names,
                   properties,
                   property_index,
                   CSS_SELECTOR_TYPE_TAG_NAME,
                   RULE_TAG_NAME_MAX);
    init_rule_hash(&stylesheet->user_agent.classes,
                   properties,
                   property_index,
                   CSS_SELECTOR_TYPE_TAG_NAME,
                   RULE_CLASS_MAX);
}

void create_dom(struct dom_node *dest,
                int *dest_classes,
                struct dom_node *parent,
                int *class_count,
                int *global_count,
                int depth) {
    if (*global_count == NODE_COUNT)
        return;
    if (depth == MAX_DOM_DEPTH)
        return;

    struct dom_node *node = &dest[(*global_count)++];
    node->id = rand() % NODE_ID_MAX;
    node->tag_name = rand() % NODE_TAG_NAME_MAX;

    node->class_count = rand() % NODE_CLASS_COUNT_MAX;
    node->first_class = *class_count;
    *class_count += node->class_count;
    for (int i = 0; i < node->class_count; i++)
        dest_classes[node->first_class + i] = rand() % NODE_CLASS_MAX;

    for (int i = 0; i < MAX_STYLE_PROPERTIES; i++)
        node->style[i] = 0;

    node->parent = parent;

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
        create_dom(dest, dest_classes, node, class_count, global_count, depth + 1);
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

void check_dom(struct dom_node *node, int *classes) {
    for (int i = 0; i < 20; i++) {
        printf("%d (id %d; tag %d; classes", i, node[i].id, node[i].tag_name);
        for (int j = 0; j < node[i].class_count; j++) {
            printf("%s%d", j == 0 ? " " : ", ", classes[node[i].first_class + j]);
        }
        printf(") -> ");
        for (int j = 0; j < MAX_STYLE_PROPERTIES; j++) {
            if (node[i].style[j] != 0)
                printf("%d=%d ", j, node[i].style[j]);
        }
        printf("\n");
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

