"""
CoNLL-2003 Official Evaluation Script for Chunking/NER Evaluation
Evaluates Strict Entity-level Precision, Recall, and Micro-F1.
Adapted from sighsmile/conlleval for Vietnamese PAP_NER.
"""

from __future__ import division, print_function, unicode_literals

import sys
from collections import defaultdict
from .constant import LOGGER


def split_tag(chunk_tag):
    """
    Split chunk tag into BIO prefix and chunk_type
    e.g.
    B-VBPL -> ('B', 'VBPL')
    I-VBPL -> ('I', 'VBPL')
    O      -> ('O', None)
    """
    if chunk_tag == 'O':
        return ('O', None)
    parts = chunk_tag.split('-', maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return ('O', None)


def is_chunk_end(prev_tag, tag):
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix1 == 'O':
        return False
    if prefix2 == 'O':
        return prefix1 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']


def is_chunk_start(prev_tag, tag):
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix2 == 'O':
        return False
    if prefix1 == 'O':
        return prefix2 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']


def calc_metrics(tp, p, t, percent=False):
    precision = tp / p if p else 0
    recall = tp / t if t else 0
    fb1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    if percent:
        return 100 * precision, 100 * recall, 100 * fb1
    else:
        return precision, recall, fb1


def count_chunks(true_seqs, pred_seqs):
    correct_chunks = defaultdict(int)
    true_chunks = defaultdict(int)
    pred_chunks = defaultdict(int)

    correct_counts = defaultdict(int)
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)

    prev_true_tag, prev_pred_tag = 'O', 'O'
    in_correct = False

    for true_tag, pred_tag in zip(true_seqs, pred_seqs):
        # Check chunk ends
        if in_correct:
            if is_chunk_end(prev_true_tag, true_tag) and is_chunk_end(prev_pred_tag, pred_tag):
                in_correct = False
                correct_chunks[split_tag(prev_true_tag)[1]] += 1
            elif is_chunk_end(prev_true_tag, true_tag) != is_chunk_end(prev_pred_tag, pred_tag):
                in_correct = False

        if is_chunk_end(prev_true_tag, true_tag):
            true_chunks[split_tag(prev_true_tag)[1]] += 1
        if is_chunk_end(prev_pred_tag, pred_tag):
            pred_chunks[split_tag(prev_pred_tag)[1]] += 1

        # Check chunk starts
        if is_chunk_start(prev_true_tag, true_tag) and is_chunk_start(prev_pred_tag, pred_tag):
            if split_tag(true_tag)[1] == split_tag(pred_tag)[1]:
                in_correct = True
            else:
                in_correct = False

        if is_chunk_start(prev_true_tag, true_tag):
            true_chunks[split_tag(true_tag)[1]] += 1
        if is_chunk_start(prev_pred_tag, pred_tag):
            pred_chunks[split_tag(pred_tag)[1]] += 1

        if true_tag == pred_tag:
            correct_counts[true_tag] += 1
        true_counts[true_tag] += 1
        pred_counts[pred_tag] += 1

        prev_true_tag, prev_pred_tag = true_tag, pred_tag

    if in_correct:
        correct_chunks[split_tag(prev_true_tag)[1]] += 1
    if is_chunk_end(prev_true_tag, 'O'):
        true_chunks[split_tag(prev_true_tag)[1]] += 1
    if is_chunk_end(prev_pred_tag, 'O'):
        pred_chunks[split_tag(prev_pred_tag)[1]] += 1

    return correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts, pred_counts


def get_result(correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts, pred_counts, verbose=True):
    sum_correct_chunks = sum(correct_chunks.values())
    sum_true_chunks = sum(true_chunks.values())
    sum_pred_chunks = sum(pred_chunks.values())

    sum_correct_counts = sum(correct_counts.values())
    sum_true_counts = sum(true_counts.values())

    nonO_correct_counts = sum(v for k, v in correct_counts.items() if k != 'O')
    nonO_true_counts = sum(v for k, v in true_counts.items() if k != 'O')

    chunk_types = sorted(list(set(list(true_chunks) + list(pred_chunks))))

    prec, rec, f1 = calc_metrics(sum_correct_chunks, sum_pred_chunks, sum_true_chunks, percent=False)
    res = (prec, rec, f1)
    if not verbose:
        return res

    LOGGER.info(f"Processed {sum_true_counts} tokens with {sum_true_chunks} gold phrases; "
                f"Found: {sum_pred_chunks} phrases; Correct: {sum_correct_chunks}.")
    if nonO_true_counts > 0:
        LOGGER.info("Accuracy (excl. 'O'): %0.4f" % (nonO_correct_counts / nonO_true_counts))
    if sum_true_counts > 0:
        LOGGER.info("Token Accuracy: %0.4f | Strict Precision: %0.4f | Strict Recall: %0.4f | Strict F1: %0.4f"
                    % (sum_correct_counts / sum_true_counts, prec, rec, f1))
    
    for t in chunk_types:
        p_t, r_t, f1_t = calc_metrics(correct_chunks[t], pred_chunks[t], true_chunks[t], percent=False)
        LOGGER.info("%17s: Precision: %0.4f; Recall: %0.4f; F1-score: %0.4f (Gold: %d, Pred: %d)" 
                    % (t, p_t, r_t, f1_t, true_chunks[t], pred_chunks[t]))
    return res


def evaluate(true_seqs, pred_seqs, verbose=True):
    correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts, pred_counts = count_chunks(true_seqs, pred_seqs)
    return get_result(correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts, pred_counts, verbose=verbose)

