#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute(y_true, y_score, cc_id, weight=None):
    """Compute model score threshold table.

    Parameters
    -----------
    y_true : pandas.Series
        Array containing the ground-truth.
    y_score : pandas.Series
        Array containing the model scores
    cc_id : pandas.Series
        The credit card ids.
    weight : pandas.Series
        The credit card ids.

    Returns
    -------
    threshold_table : pandas.DataFrame
        The pandas dataframe with the cut-table.
    """

    nb_ccs = float(cc_id.nunique())

    is_fraud = y_true == 1

    score_candidates_for_threshold = (
        y_score.round(3).drop_duplicates().sort_values())

    threshold_table = []
    for score in score_candidates_for_threshold:
        is_accepted = (y_score <= score)
        is_rejected = (y_score > score)

        nb_accepted = float(cc_id[is_accepted].nunique())
        nb_rejected = float(cc_id[is_rejected].nunique())

        weight_fraud_ratio = None
        if weight is not None:
            weight_accepted = weight[is_accepted].sum()
            weight_accepted_fraud = weight[is_fraud & is_accepted].sum()

            weight_fraud_ratio = weight_accepted_fraud / weight_accepted

        acceptance_rate = nb_accepted / nb_ccs

        accepted_nb_fraud = cc_id[is_fraud & is_accepted].nunique()
        rejected_nb_fraud = cc_id[is_fraud & is_rejected].nunique()

        accepted_nb_no_fraud = cc_id[(~is_fraud) & is_accepted].nunique()
        rejected_nb_no_fraud = cc_id[(~is_fraud) & is_rejected].nunique()

        if nb_accepted != 0:
            accepted_fraud_percent = accepted_nb_fraud / nb_accepted
        else:
            accepted_fraud_percent = 0

        if nb_rejected != 0:
            rejected_fraud_percent = rejected_nb_fraud / nb_rejected
        else:
            rejected_fraud_percent = 0

        threshold_table.append({
            'min_score': score,
            'acceptance_rate': acceptance_rate,
            'nb_accepted': nb_accepted,
            'nb_rejected': nb_rejected,
            'accepted_nb_fraud': accepted_nb_fraud,
            'rejected_nb_fraud': rejected_nb_fraud,
            'accepted_nb_no_fraud': accepted_nb_no_fraud,
            'rejected_nb_no_fraud': rejected_nb_no_fraud,
            'accepted_fraud_percent': accepted_fraud_percent,
            'rejected_fraud_percent': rejected_fraud_percent,
            'weight_fraud_ratio': weight_fraud_ratio
        })

    threshold_table = pd.DataFrame(threshold_table).set_index('min_score')

    return threshold_table


def plot(cut_table, accept_th=None, reject_th=None, title=''):
    """Visualization for the cut-table.

    Parameters
    ----------
    cut_table: pd.DataFrame.
        The cut-table.
    accept_th: float, None
        Acceptance threshold.
    reject_th: float, None
        Rejection threshold.

    Returns
    -------
    fig : matplotlib.figure.Figure | None
        The figure in whith to plot.

    ax : matplotlib.axes._subplots.AxesSubplot | None
        The axe in whith to plot.
    """
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))

    # Fraud percent in accepted.
    cut_table['accepted_fraud_percent'].plot(
        marker='o',
        grid=True,
        ax=ax[0])
    ax[0].set_title((
        '%s (%s)\n'
        'Fraud Percent in ACCEPTED\n'
        'Y-axis right: Gray-line denotes Acceptance-rate, '
        'Red-line denotes LR due to fraud') % (
            title,
            int(cut_table['nb_accepted'].max())),
        fontsize=15
    )

    ax[0].set_xlabel('Fraud-score threshold.')
    ax[0].set_ylabel('% of Fraud.')

    # Fraud percent in rejected.
    cut_table['rejected_fraud_percent'].plot(
        marker='o',
        grid=True,
        ax=ax[1])
    ax[1].set_title((
        '%s (%s)\n'
        'Fraud Percent in REJECTED\n'
        'Y-axis right: Gray-line denotes Acceptance-rate, '
        'Red-line denotes LR due to fraud') % (
            title,
            int(cut_table['nb_accepted'].max())),
        fontsize=15)

    ax[1].set_xlabel('Fraud-score threshold.')
    ax[1].set_ylabel('% of Fraud.')

    # Aux. line for acceptance
    for x in ax.flatten():
        ax_ = x.twinx()
        cut_table['acceptance_rate'].plot(
            marker='',
            color='gray',
            alpha=.5,
            ax=ax_)

        weight_fraud_ratio = cut_table['weight_fraud_ratio'].dropna()
        if weight_fraud_ratio.shape[0] > 0:
            weight_fraud_ratio.plot(
                marker='',
                color='red',
                alpha=.5,
                ax=ax_)

    # Reference lines for acceptance rate.
    ref_cut_points = {}
    for acceptance_rate in np.arange(0, 1.05, .05):
        abs_diff = (
            acceptance_rate - cut_table['acceptance_rate']
        ).abs().values

        ref_cut_points[
            cut_table.index[np.argmin(abs_diff)]
        ] = acceptance_rate

    ref_cut_points = pd.Series(ref_cut_points).round(2)

    # Plot reference lines
    for it_plot, x in enumerate(list(ax)):
        for it, score in enumerate(ref_cut_points.index):
            x.axvline(
                score,
                c='blue',
                alpha=.5,
                linestyle=':' if (it % 2) == 0 else '-')

        if reject_th is not None:
            x.axvline(
                reject_th,
                c='red')

        if accept_th is not None:
            x.axvline(
                accept_th,
                c='green')

    # Plot metadata in acceptance and
    if accept_th is not None:
        ct_accept = cut_table.loc[accept_th]

        th_metadata = 'score: %s\nAR: %s\n%% Fraud in Acc.: %s' % (
            round(accept_th, 2),
            round(ct_accept['acceptance_rate'], 2),
            round(ct_accept['accepted_nb_fraud'] / ct_accept['nb_accepted'], 2))
        ax[0].annotate(
            xy=(
                accept_th + .01,
                cut_table['accepted_fraud_percent'].quantile(.1)),
            text=th_metadata,
            fontsize=15)

        th_metadata = 'score: %s\nRR: %s\n%% Fraud in Rej.: %s' % (
            round(accept_th, 2),
            round(1 - ct_accept['acceptance_rate'], 2),
            round(ct_accept['rejected_nb_fraud'] / ct_accept['nb_rejected'], 2))
        ax[1].annotate(
            xy=(
                accept_th + .01,
                cut_table['rejected_fraud_percent'].quantile(.1)),
            text=th_metadata,
            fontsize=15)

    if reject_th is not None:
        ct_reject = cut_table.loc[reject_th]
        th_metadata = 'score: %s\nAR: %s\n%% Fraud in Acc.: %s' % (
            round(reject_th, 2),
            round(ct_reject['acceptance_rate'], 2),
            round(ct_reject['accepted_nb_fraud'] / ct_reject['nb_accepted'], 2))

        ax[0].annotate(
            xy=(
                reject_th + .01,
                cut_table['accepted_fraud_percent'].quantile(.5)),
            text=th_metadata,
            fontsize=15)

        th_metadata = 'score: %s\nRR: %s\n%% Fraud in Rej.: %s' % (
            round(reject_th, 2),
            round(1 - cut_table.loc[reject_th]['acceptance_rate'], 2),
            round(ct_reject['rejected_nb_fraud'] / ct_reject['nb_rejected'], 2))

        ax[1].annotate(
            xy=(
                reject_th + .01,
                cut_table['rejected_fraud_percent'].quantile(.5)),
            text=th_metadata,
            fontsize=15)

    fig.set_tight_layout('tight')

    return fig, ax


def cut_point_selection_stats(cut_table, accept_th, reject_th):
    """Given a selection of cut points, get relevant stats.
    """
    cut_table_reject_th = cut_table.loc[reject_th]
    rejection_rate = 1 - cut_table_reject_th['acceptance_rate']
    acceptance_rate = cut_table_reject_th['acceptance_rate']

    accepted_fraud_percent = cut_table_reject_th['accepted_fraud_percent']
    rejected_fraud_percent = cut_table_reject_th['rejected_fraud_percent']

    stats = {
        'accept_th': round(accept_th, 2),
        'reject_th': round(reject_th, 2),
        'rejection_rate': round(rejection_rate, 2),
        'acceptance_rate': round(acceptance_rate, 2),
        'fraud_in_rejected': round(rejected_fraud_percent, 2),
        'fraud_in_accepted': round(accepted_fraud_percent, 2)}

    return stats