from datetime import timedelta
from typing import Dict, Any
import numpy as np
import os
from collections import Counter

import math

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


def jaccard_similarity(set1, set2):
    return len(set1.intersection(set2)) / len(set1.union(set2)) \
            if len(set1.union(set2)) > 0 else .0


class FeatureGenerator(object):
    def __init__(self, use_sim_cs, use_sim_j, use_social, use_temporal, use_file, use_pr_only, use_issue_only,
                 similarity_config=None, temporal_config=None):
        self.use_sim_cs = use_sim_cs
        self.use_sim_j = use_sim_j
        self.use_file = use_file
        if self.use_sim_j or self.use_sim_cs or self.use_file:
            assert similarity_config
            self.dictionary = similarity_config['dict']
            self.model = similarity_config['model']
            self.min_len = similarity_config['min_len']
            self.stopwords = similarity_config['stopwords']
        self.use_social = use_social
        self.use_temporal = use_temporal
        if self.use_temporal:
            assert temporal_config
            self.fingerprint = temporal_config['fingerprint']
            self.net_size_in_days = temporal_config['net_size_in_days']
        self.use_pr_only = use_pr_only
        self.use_issue_only = use_issue_only

    def generate_features(self, issue_: Issue, pr: PullRequest, linked: bool) -> Dict[str, Any]:
        issue_id = issue_.id_
        pr_id = pr.number
        features = {'issue': issue_id, 'pr': pr_id, 'linked': linked}
        if self.use_sim_cs or self.use_sim_j or self.use_file:
            full_issue_text = text_pipeline(issue_, self.stopwords_, self.min_len)
        if self.use_sim_cs or self.use_sim_j:
            full_pr_text = text_pipeline(pr, self.stopwords_, self.min_len)
            pr_title_text = preprocess_text(pr.title, self.stopwords_, self.min_len)
            pr_desc_text = preprocess_text(pr.comments[0].body, self.stopwords_, self.min_len)
            issue_title_text = preprocess_text(issue_.title, self.stopwords_, self.min_len)
            issue_report_text = preprocess_text(issue_.original_post.body, self.stopwords_, self.min_len)

        if self.use_sim_cs:
            pr_vector = np.zeros((len(self.dict_.token2id),))
            for index, value in self.model_[self.dict_.doc2bow(full_pr_text)]:
                pr_vector[index] += value
            i_vector = np.zeros((len(self.dict_.token2id),))
            for index, value in self.model_[self.dict_.doc2bow(full_issue_text)]:
                i_vector[index] += value

            cosine = cosine_similarity(pr_vector, i_vector)

            pr_title_vector = np.zeros((len(self.dict_.token2id),))
            for index, value in self.model_[self.dict_.doc2bow(pr_title_text)]:
                pr_title_vector[index] += value
            pr_comment_vector = np.zeros((len(self.dict_.token2id),))
            try:
                for index, value in self.model_[self.dict_.doc2bow(pr_desc_text)]:
                    pr_comment_vector[index] += value
            except IndexError:
                pass
            i_title_vector = np.zeros((len(self.dict_.token2id),))
            for index, value in self.model_[self.dict_.doc2bow(issue_title_text)]:
                i_title_vector[index] += value
            i_comment_vector = np.zeros((len(self.dict_.token2id),))
            for index, value in self.model_[self.dict_.doc2bow(issue_report_text)]:
                i_comment_vector[index] += value

            cosine_tt = cosine_similarity(pr_title_vector, i_title_vector)
            cosine_tc = cosine_similarity(pr_title_vector, i_comment_vector)
            cosine_ct = cosine_similarity(pr_comment_vector, i_title_vector)
            cosine_cc = cosine_similarity(pr_comment_vector, i_comment_vector)
            features['cosine'] = cosine
            features['cosine_tt'] = cosine_tt
            features['cosine_tc'] = cosine_tc
            features['cosine_ct'] = cosine_ct
            features['cosine_cc'] = cosine_cc

        if self.use_file or self.use_sim_j:
            words_in_issue = set(full_issue_text)

        if self.use_sim_j:
            words_in_pr = set(full_pr_text)
            jaccard = jaccard_similarity(words_in_pr, words_in_issue)

            words_in_pr_title = set(pr_title_text)
            words_in_pr_desc = set(pr_desc_text)
            words_in_issue_title = set(issue_title_text)
            words_in_issue_report = set(issue_report_text)

            jaccard_tt = jaccard_similarity(words_in_pr_title, words_in_issue_title)
            jaccard_tc = jaccard_similarity(words_in_pr_title, words_in_issue_report)
            jaccard_ct = jaccard_similarity(words_in_pr_desc, words_in_issue_title)
            jaccard_cc = jaccard_similarity(words_in_pr_desc, words_in_issue_report)
            features['jaccard'] = jaccard
            features['jaccard_tt'] = jaccard_tt
            features['jaccard_tc'] = jaccard_tc
            features['jaccard_ct'] = jaccard_ct
            features['jaccard_cc'] = jaccard_cc

        if self.use_file:
            files_in_pr = {ps.stem(str(os.path.basename(diff[0]).split('.')[0])) for diff in pr.diffs}
            files_in_issue = len(files_in_pr.intersection(words_in_issue)) / (len(issue_.replies) + 1)
            features['files_shared'] = files_in_issue

        if self.use_social or self.use_temporal:
            author = pr.comments[0].author if pr.comments else None

        if self.use_social:
            reporter = issue_.original_post.author == author
            assignee = issue_.assignee == author
            engagement = len([reply
                              for reply
                              in [issue_.original_post] + issue_.replies
                              if reply.author == author]) / (len(issue_.replies) + 1)
            in_top_2 = author in [a for a, _
                                  in Counter(map(lambda c: c.author, [issue_.original_post] + issue_.replies)).most_common(2)]
            comments = engagement > .0
            features['is_reporter'] = reporter
            features['is_assignee'] = assignee
            features['engagement'] = engagement
            features['in_top_2'] = in_top_2
            features['in_comments'] = comments

        if self.use_temporal:
            try:
                lag = min([abs(entity.timestamp - pr.comments[0].timestamp)
                           if entity.timestamp and pr.comments
                           else timedelta(seconds=self.fingerprint_['AVG'])
                           for entity in [issue_.original_post]
                           + [reply for reply in issue_.replies if reply.author == author]
                           + [state for state in issue_.states if state.to_ == IssueStates.closed]]).total_seconds()
                # Scale lag on a per developer basis
                try:
                    lag /= self.fingerprint_[author] if self.fingerprint_[author] > .0 else self.fingerprint_['AVG']
                except KeyError:
                    lag /= self.fingerprint_['AVG']
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
                lag_close = timedelta(days=self.net_size_in_days, seconds=1).total_seconds()

            try:
                lag_open = (abs(issue_.original_post.timestamp - pr.comments[0].timestamp)).total_seconds() \
                    if pr.comments else timedelta(days=self.net_size_in_days, seconds=1).total_seconds()
            except TypeError:
                lag_open = timedelta(days=self.net_size_in_days, seconds=1).total_seconds()

            features['developer_normalised_lag'] = lag
            features['lag_from_issue_open_to_pr_submission'] = lag_open
            features['lag_from_last_issue_update_to_pr_submission'] = lag_close

        if self.use_pr_only:
            no_pr_desc = len(pr.comments) == 0 \
                         or pr.comments[0].body == 'No description provided.' \
                         or pr.comments[0].body == ''
            features['no_pr_desc'] = no_pr_desc
            features['branch_size'] = len(pr.commits)
            features['files_touched_by_pr'] = len(pr.diffs)

        if self.use_issue_only:
            features['report_size'] = len(issue_.original_post)
            features['participants'] = len(set([c.author for c in issue_.replies + [issue_.original_post]]))
            features['bounces'] = len([s.to_ == IssueStates.open for s in issue_.states])
            features['existing_links'] = len(issue_.commits)

        return features
