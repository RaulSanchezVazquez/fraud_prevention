#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd


def compute(y_true, y_score, weights):
    """Compute model score threshold table.

    Parameters
    -----------
    y_true : pandas.Series
        Array containing the ground-truth.
    y_score : pandas.Series
        Array containing the model scores

    Returns
    -------
    threshold_table : pandas.DataFrame
        The pandas dataframe with the cut-table.
    """

    nb_transactions = len(y_true)

    is_fraud = y_true == 1

    score_candidates_for_threshold = (
        y_score.round(3).drop_duplicates().sort_values())

    threshold_table = []
    for score in score_candidates_for_threshold:
        is_accepted = (y_score <= score)
        is_rejected = (y_score > score)

        nb_accepted = is_accepted.sum()
        nb_rejected = is_rejected.sum()

        acceptance_rate = nb_accepted / nb_transactions

        accepted_nb_fraud = (is_fraud & is_accepted).sum()
        rejected_nb_fraud = (is_fraud & is_rejected).sum()

        accepted_nb_no_fraud = ((~is_fraud) & is_accepted).sum()
        rejected_nb_no_fraud = ((~is_fraud) & is_rejected).sum()

        if nb_accepted != 0:
            accepted_fraud_percent = accepted_nb_fraud / nb_accepted
        else:
            accepted_fraud_percent = 0

        if nb_rejected != 0:
            rejected_fraud_percent = rejected_nb_fraud / nb_rejected
        else:
            rejected_fraud_percent = 0

        threshold_table.append({
            'score': score,
            'acceptance_rate': acceptance_rate,
            'nb_accepted': nb_accepted,
            'nb_rejected': nb_rejected,
            'accepted_nb_fraud': accepted_nb_fraud,
            'rejected_nb_fraud': rejected_nb_fraud,
            'accepted_nb_no_fraud': accepted_nb_no_fraud,
            'rejected_nb_no_fraud': rejected_nb_no_fraud,
            'accepted_fraud_percent': accepted_fraud_percent,
            'rejected_fraud_percent': rejected_fraud_percent
        })

    threshold_table = pd.DataFrame(threshold_table).set_index('score')

    return threshold_table
