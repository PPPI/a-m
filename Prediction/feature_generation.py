from datetime import timedelta
from typing import Set
import numpy as np
import os
from collections import Counter

import math

from Prediction.named_tuple_util import link_datapoint
from Prediction.text_utils import ps, text_pipeline, preprocess_text
from gitMine.VCClasses import Issue, PullRequest, IssueStates


def cosine_similarity(vec1, vec2):
    """
    Return the cosine similarity between two vectors.

    Args:
        vec1 (:class:`numpy.array`)
        vec2 (:class:`numpy.array`)

    Returns:
        float
    """
    c = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return .0 if math.isnan(c) else c


# TODO: Separate each feature into own function
def generate_features(issue_: Issue, pr: PullRequest, stopwords_: Set[str], fingerprint_, dict_, model_, linked,
                      min_len, net_size_in_days) -> link_datapoint:
    author = pr.comments[0].author if pr.comments else None
    files_in_pr = {ps.stem(str(os.path.basename(diff[0]).split('.')[0])) for diff in pr.diffs}
    words_in_pr = set(text_pipeline(pr, stopwords_, min_len))
    words_in_issue = set(text_pipeline(issue_, stopwords_, min_len))

    pr_vector = np.zeros((len(dict_.token2id),))
    for index, value in model_[dict_.doc2bow(text_pipeline(pr, stopwords_, min_len))]:
        pr_vector[index] += value
    i_vector = np.zeros((len(dict_.token2id),))
    for index, value in model_[dict_.doc2bow(text_pipeline(issue_, stopwords_, min_len))]:
        i_vector[index] += value

    cosine = cosine_similarity(pr_vector, i_vector)

    pr_title_vector = np.zeros((len(dict_.token2id),))
    for index, value in model_[dict_.doc2bow(preprocess_text(pr.title, stopwords_, min_len))]:
        pr_title_vector[index] += value
    pr_comment_vector = np.zeros((len(dict_.token2id),))
    try:
        for index, value in model_[dict_.doc2bow(preprocess_text(pr.comments[0].body, stopwords_, min_len))]:
            pr_comment_vector[index] += value
    except IndexError:
        pass
    i_title_vector = np.zeros((len(dict_.token2id),))
    for index, value in model_[dict_.doc2bow(preprocess_text(issue_.title, stopwords_, min_len))]:
        i_title_vector[index] += value
    i_comment_vector = np.zeros((len(dict_.token2id),))
    for index, value in model_[dict_.doc2bow(preprocess_text(issue_.original_post.body, stopwords_, min_len))]:
        i_comment_vector[index] += value

    cosine_tt = cosine_similarity(pr_title_vector, i_title_vector)
    cosine_tc = cosine_similarity(pr_title_vector, i_comment_vector)
    cosine_ct = cosine_similarity(pr_comment_vector, i_title_vector)
    cosine_cc = cosine_similarity(pr_comment_vector, i_comment_vector)

    words_in_common = len(words_in_pr.intersection(words_in_issue)) / len(words_in_pr.union(words_in_issue)) \
        if len(words_in_pr.union(words_in_issue)) > 0 else .0
    files_in_issue = len(files_in_pr.intersection(words_in_issue)) / (len(issue_.replies) + 1)

    issue_id = issue_.id_
    pr_id = pr.number

    reporter = issue_.original_post.author == author
    assignee = issue_.assignee == author
    engagement = len([reply
                      for reply
                      in [issue_.original_post] + issue_.replies
                      if reply.author == author]) / (len(issue_.replies) + 1)
    in_top_2 = author in [a for a, _
                          in Counter(map(lambda c: c.author, [issue_.original_post] + issue_.replies)).most_common(2)]
    comments = engagement > .0

    try:
        lag = min([abs(entity.timestamp - pr.comments[0].timestamp)
                   if entity.timestamp and pr.comments
                   else timedelta(seconds=fingerprint_['AVG'])
                   for entity in [issue_.original_post]
                   + [reply for reply in issue_.replies if reply.author == author]
                   + [state for state in issue_.states if state.to_ == IssueStates.closed]]).total_seconds()
        # Scale lag on a per developer basis
        try:
            lag /= fingerprint_[author] if fingerprint_[author] > .0 else fingerprint_['AVG']
        except KeyError:
            lag /= fingerprint_['AVG']
        # And centre around 0
        lag -= 1
    except ValueError:
        lag = .0

    if math.isnan(lag):
        lag = .0

    # Also absolute lag as a sanity check
    try:
        lag_close = min([abs(entity.timestamp - pr.comments[0].timestamp)
                         for entity in
                         [state for state in issue_.states if state.to_ == IssueStates.closed]]).total_seconds()
    except (ValueError, IndexError):
        lag_close = timedelta(days=net_size_in_days, seconds=1).total_seconds()

    try:
        lag_open = (abs(issue_.original_post.timestamp - pr.comments[0].timestamp)).total_seconds() \
            if pr.comments else timedelta(days=net_size_in_days, seconds=1).total_seconds()
    except TypeError:
        lag_open = timedelta(days=net_size_in_days, seconds=1).total_seconds()

    no_pr_desc = len(pr.comments) == 0 or pr.comments[0].body == 'No description provided.' or pr.comments[0].body == ''

    return link_datapoint(issue=issue_id, pr=pr_id, in_comments=comments, is_reporter=reporter, is_assignee=assignee,
                          engagement=engagement, lag=lag, linked=linked, jaccard=words_in_common,
                          files=files_in_issue, top_2=in_top_2, cosine=cosine, cosine_tt=cosine_tt,
                          cosine_tc=cosine_tc, cosine_ct=cosine_ct, cosine_cc=cosine_cc,
                          lag_open=lag_open, lag_close=lag_close, pr_commits=len(pr.commits), no_pr_desc=no_pr_desc,
                          pr_files=len(pr.diffs))
