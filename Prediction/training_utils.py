import math
from multiprocessing.pool import Pool
from random import shuffle

import multiprocessing
import numpy as np

from datetime import datetime, timedelta
from typing import List, Tuple, Union, Set, Dict, Any

from gensim.corpora import Dictionary
from gensim.models import tfidfmodel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.pipeline import Pipeline
from skgarden import MondrianForestClassifier

from Prediction.text_utils import text_pipeline
from gitMine.VCClasses import Repository, Commit, Comment, StateChange, Reference, Issue, PullRequest, IssueStates

null_issue = Issue(assignee='', id_='null_issue', original_post=Comment(author='', body='', id_='null_comment',
                                                                        timestamp=None), repository='', title='')
null_issue.commits = set()

null_pr = PullRequest(number='null_pr', title='', assignee='', comments=list(), commits=list(), diffs=list(),
                      from_branch='', to_branch='', from_repo='', labels=list(), state=IssueStates.open, to_repo='')

null_commit = Commit(parent='', author=None, title='', desc='', branches=list(), repository='', c_hash='null_commit',
                      diff=list(), timestamp=None)


# The property that the first element of the tuple has is '.timestamp'
def flatten_events(repo_: Repository) \
        -> List[Tuple[Union[Commit, Comment, StateChange, Reference], Union[Commit, Issue, PullRequest]]]:
    if repo_.prs is None:
        repo_.prs = list()

    if repo_.issues is None:
        repo_.issues = list()

    if repo_.commits is None:
        repo_.commits = list()

    result = [(c, c) for c in repo_.commits]
    result += [(comment, i) for i in repo_.issues for comment in [i.original_post] + i.replies]
    result += [(states, i) for i in repo_.issues for states in i.states]
    result += [(action, i) for i in repo_.issues for action in i.actions if isinstance(action, Reference)]
    result += [(comment, pr) for pr in repo_.prs for comment in pr.comments]
    result += [(commit, pr) for pr in repo_.prs for commit in pr.commits]
    result = sorted(map(lambda e: [e, setattr(e[0], 'timestamp', e[0].timestamp.replace(tzinfo=None))][0], result),
                    key=lambda e: e[0].timestamp)
    return result


def filter_flat_events_by_time(
        events: List[Tuple[Union[Commit, Comment, StateChange, Reference], Union[Commit, Issue, PullRequest]]],
        start: datetime.timestamp,
        end: datetime.timestamp) \
        -> List[Tuple[Union[Commit, Comment, StateChange, Reference], Union[Commit, Issue, PullRequest]]]:
    return [p for p in events if start <= p[0].timestamp <= end]


def update(event: Tuple[Union[Comment, StateChange, Reference], Union[Issue, PullRequest]],
           list_: List[Union[Issue, PullRequest]]) -> List[Union[Issue, PullRequest]]:
    event_id = event[1].id_ if isinstance(event[1], Issue) else event[1].number
    for elem in list_:
        id_ = elem.id_ if isinstance(elem, Issue) else elem.number
        if id_ == event_id:
            if isinstance(event[0], Comment):
                elem.replies.append(event[0]) if isinstance(elem, Issue) else elem.comments.append(event[0])
            elif isinstance(event[0], StateChange):
                if isinstance(elem, Issue):
                    elem.states.append(event[0])
                else:
                    elem.state = event[0].to_
            elif isinstance(event[0], Reference):
                elem.actions.append(event[0])
            elif isinstance(event[0], Commit):
                elem.commits.append(event[0])
            return list_
    new_event = None
    if isinstance(event[0], Comment):
        new_event = Issue(id_=event_id, assignee=event[1].assignee, original_post=event[0],
                          repository=event[1].repository, title=event[1].title) if isinstance(event[1], Issue) \
            else PullRequest(assignee=event[1].assignee, labels=event[1].labels, diffs=event[1].diffs,
                             comments=[event[0]], commits=list(),
                             from_branch=event[1].from_branch, from_repo=event[1].from_repo,
                             number=event_id, state=event[1].state, title=event[1].title,
                             to_branch=event[1].to_branch, to_repo=event[1].to_repo)
    elif isinstance(event[0], StateChange):
        # This happens only when the event started before the prefix
        pass
    elif isinstance(event[0], Reference):
        # This happens only when the event started before the prefix
        pass
    elif isinstance(event[0], Commit):
        new_event = PullRequest(assignee=event[1].assignee, labels=event[1].labels, diffs=event[1].diffs,
                                comments=list(), commits=[event[0]],
                                from_branch=event[1].from_branch, from_repo=event[1].from_repo,
                                number=event_id, state=event[1].state, title=event[1].title,
                                to_branch=event[1].to_branch, to_repo=event[1].to_repo)
    if new_event:
        list_.append(new_event)
    return list_


def inflate_events(events: List[Tuple[Union[Commit, Comment, StateChange, Reference],
                                      Union[Commit, Issue, PullRequest]]],
                   langs: List[str], name: str) -> Repository:
    prs = list()
    issues = list()
    commits = list()
    for event in events:
        if isinstance(event[1], Commit):
            commits.append(event[0])
        else:
            update(event, issues) if isinstance(event[1], Issue) else update(event, prs)
    return Repository(commits=commits, issues=issues, langs=langs, name=name, prs=prs)


def generate_batches(repo_, n_batches_):
    event_stream = flatten_events(repo_)

    batches_ = list()
    sorted_prs = sorted(repo_.prs, key=lambda pr: pr.comments[0].timestamp)
    chunk_size = int(len(sorted_prs) / n_batches_)
    chunked_prs = [list(t) for t in zip(*[iter(sorted_prs)] * chunk_size)]
    chunked_prs[0] = [event_stream[0][0]] + chunked_prs[0]
    chunked_prs[-1].append(event_stream[-1][0])
    for chunk in chunked_prs:
        try:
            start = chunk[0].comments[0].timestamp
        except AttributeError:
            start = chunk[0].timestamp
        try:
            end = chunk[-1].comments[0].timestamp
        except AttributeError:
            end = chunk[-1].timestamp

        batch_ = filter_flat_events_by_time(event_stream, start, end)
        batches_.append(batch_)

    return batches_


def generate_dev_fingerprint(repo_: Repository):
    devs = {commit.author for commit in repo_.commits}
    devs = devs.union({issue_.assignee for issue_ in repo_.issues})
    dev_tick_rate = dict()

    average_joe = list()
    for dev in devs:
        pr_td = list()
        for pr in [pr for pr in repo_.prs if (pr.comments[0].author == dev if pr.comments else False)]:
            for commit in pr.commits:
                for issue_ in repo_.issues:
                    if any([commit.c_hash.startswith(candidate) for candidate in issue_.commits]):
                        pr_td.append(min([abs(entity.timestamp - pr.comments[0].timestamp)
                                          for entity in [issue_.original_post]
                                          + [reply for reply in issue_.replies if reply.author == dev]
                                          + [state for state in issue_.states if
                                             state.to_ == IssueStates.closed]]).total_seconds())
                        break
        if pr_td:
            avg = float(np.mean(pr_td))
            dev_tick_rate[dev] = avg if not (math.isnan(avg)) else 1.0
            average_joe.append(dev_tick_rate[dev])
    avg = float(np.mean(average_joe))
    dev_tick_rate['AVG'] = avg if not (math.isnan(avg)) else 1.0
    return dev_tick_rate


def undersample_naively(mult_, arg_list):
    true_links = [d for d in arg_list if d[-1]]
    false_links = [d for d in arg_list if not d[-1]]
    shuffle(false_links)
    false_links = false_links[:min(len(false_links), mult_ * len(true_links))]
    arg_list = true_links + false_links
    shuffle(arg_list)
    return arg_list


def generate_pr_issue_interest_pairs(pr_, issue_list, truth_, net_size_in_days):
    author = pr_.comments[0].author if pr_.comments else ''
    considered = [i for i in issue_list
                  if (min([abs(entity.timestamp - pr_.comments[0].timestamp)
                           if entity.timestamp and pr_.comments
                           else timedelta(days=net_size_in_days, seconds=1)
                           for entity in
                           [i.original_post]
                           + [r for r in i.replies if r.author == author]
                           + i.states
                           + i.actions]) <= timedelta(days=net_size_in_days))]
    link_data = list()
    for issue_ in considered:
        try:
            link_data.append(('#' + pr_.number[len('issue_'):]) in truth_['#' + issue_.id_[len('issue_'):]])
        except KeyError:
            link_data.append(False)
    return zip(considered, [pr_] * len(considered), link_data)


def generate_commit_issue_interest_pairs(commit_, issue_list, net_size_in_days):
    author = commit_.author
    considered = [i for i in issue_list
                  if (min([abs(entity.timestamp - commit_.timestamp)
                           if hasattr(entity, 'timestamp') and entity.timestamp
                           else timedelta(days=net_size_in_days, seconds=1)
                           for entity in
                           [i.original_post]
                           + [r for r in i.replies if r.author == author]
                           + i.states
                           + i.actions]) <= timedelta(days=net_size_in_days))]
    link_data = list()
    for issue_ in considered:
        try:
            link_data.append(any([commit_.c_hash.startswith(c) for c in issue_.commits]))
        except KeyError:
            link_data.append(False)
    return zip(considered, [commit_] * len(considered), link_data)


def generate_pi_wrapper(args):
    return generate_pr_issue_interest_pairs(*args)


def generate_ci_wrapper(args):
    return generate_commit_issue_interest_pairs(*args)


class Feature_Closure(object):
    def __init__(self, feature_generator):
        self.feature_generator = feature_generator

    def __call__(self, args):
        return self.feature_generator.generate_features(*args)


class Feature_Closure_Commit(object):
    def __init__(self, feature_generator):
        self.feature_generator = feature_generator

    def __call__(self, args):
        return self.feature_generator.generate_features_commit(*args)


def generate_training_data(training_repo_: Repository,
                           feature_generator,
                           net_size_in_days,
                           truth_,
                           mult_) -> List[Dict[str, Any]]:
    with Pool(processes=multiprocessing.cpu_count() - 1) as wp:
        arg_list = list()
        for v in wp.imap_unordered(generate_pi_wrapper,
                                   zip(training_repo_.prs, len(training_repo_.prs) * [training_repo_.issues],
                                       len(training_repo_.prs) * [truth_],
                                       len(training_repo_.prs) * [net_size_in_days]),
                                   chunksize=128):
            for i, p, linked in v:
                arg_list.append((i, p, linked))

        # Explicitly add no_link to the training data
        issue_map = {i[0]: any([t[-1] for t in arg_list]) for i in arg_list}
        pr_map = {i[1]: any([t[-1] for t in arg_list]) for i in arg_list}
        for issue, any_link in issue_map.items():
            if not any_link:
                arg_list.append((issue, null_pr, True))
            else:
                arg_list.append((issue, null_pr, False))
        for pr, any_link in pr_map.items():
            if not any_link:
                arg_list.append((null_issue, pr, True))
            else:
                arg_list.append((null_issue, pr, False))

        arg_list = undersample_naively(mult_, arg_list)

        training_data_ = list()
        for point in wp.imap_unordered(Feature_Closure(feature_generator), arg_list, chunksize=128):
            training_data_.append(point)
    return training_data_


def generate_training_data_commit(training_repo_: Repository,
                                  feature_generator,
                                  net_size_in_days,
                                  mult_) -> List[Dict[str, Any]]:
    with Pool(processes=multiprocessing.cpu_count() - 1) as wp:
        arg_list = list()
        for v in wp.imap_unordered(generate_ci_wrapper,
                                   zip(training_repo_.commits, len(training_repo_.commits) * [training_repo_.issues],
                                       len(training_repo_.commits) * [net_size_in_days]),
                                   chunksize=128):
            for i, c, linked in v:
                arg_list.append((i, c, linked))

        # Explicitly add no_link to the training data
        issue_map = {i[0]: any([t[-1] for t in arg_list]) for i in arg_list}
        commit_map = {i[1]: any([t[-1] for t in arg_list]) for i in arg_list}
        for issue, any_link in issue_map.items():
            if not any_link:
                arg_list.append((issue, null_commit, True))
            else:
                arg_list.append((issue, null_commit, False))
        for pr, any_link in commit_map.items():
            if not any_link:
                arg_list.append((null_issue, pr, True))
            else:
                arg_list.append((null_issue, pr, False))

        arg_list = undersample_naively(mult_, arg_list)

        training_data_ = list()
        for point in wp.imap_unordered(Feature_Closure_Commit(feature_generator), arg_list, chunksize=128):
            training_data_.append(point)
    return training_data_


def generate_training_data_seq(training_repo_: Repository,
                               feature_generator,
                               net_size_in_days,
                               truth_,
                               mult_) -> List[Dict[str, Any]]:
    training_data_ = list()
    arg_list = list()
    for pr in training_repo_.prs:
        for v in generate_pr_issue_interest_pairs(pr, training_repo_.issues, truth_, net_size_in_days):
            i, p, linked = v
            arg_list.append((i, p, linked))

    # Explicitly add no_link to the training data
    issue_map = {i[0]: any([t[-1] for t in arg_list]) for i in arg_list}
    pr_map = {i[1]: any([t[-1] for t in arg_list]) for i in arg_list}
    for issue, any_link in issue_map.items():
        if not any_link:
            arg_list.append((issue, null_pr, True))
    for pr, any_link in pr_map.items():
        if not any_link:
            arg_list.append((null_issue, pr, True))

    arg_list = undersample_naively(mult_, arg_list)

    for i, p, linked in arg_list:
        training_data_.append(feature_generator.generate_features(i, p, linked))
    return training_data_


def train_classifier(training_data_: List[Dict[str, Any]], perform_feature_selection: bool = False) \
        -> MondrianForestClassifier:
    X = list()
    y = list()
    for point in training_data_:
        X.append(tuple([v for k, v in point.items() if k not in ['linked', 'issue', 'pr', 'commit']]))
        y.append(1 if point['linked'] else -1)
    if perform_feature_selection:
        clf_ = Pipeline([
            ('feature_selection', SelectFromModel(RFE(
                RandomForestClassifier(n_estimators=128, class_weight='balanced_subsample'), 5, step=1))),
            ('classification', MondrianForestClassifier(n_estimators=128, ))
        ])
    else:
        clf_ = MondrianForestClassifier(n_estimators=128, )
    clf_.partial_fit(X, y)
    return clf_


def generate_tfidf(repository: Repository, stopwords_: Set[str], min_len, cache=None) -> Tuple[tfidfmodel.TfidfModel, Dictionary, Dict]:
    if cache is None:
        cache = dict()

    texts = list()
    for pr in repository.prs:
        if pr.number in cache.keys():
            texts.append(cache[pr.number])
        else:
            text = text_pipeline(pr, stopwords_, min_len)
            texts.append(text)
            cache[pr.number] = text
    for issue_ in repository.issues:
        if issue_.id_ in cache.keys():
            texts.append(cache[issue_.id_])
        else:
            text = text_pipeline(issue_, stopwords_, min_len)
            texts.append(text)
            cache[issue_.id_] = text

    dictionary_ = Dictionary(texts)
    dictionary_.filter_extremes(no_below=3, no_above=0.95)
    working_corpus = [dictionary_.doc2bow(text, return_missing=True) for text in texts]
    # Convert UNK from explicit dictionary to UNK token (id = -1)
    working_corpus = [val[0] + [(-1, sum(val[1].values()))] for val in working_corpus]
    return tfidfmodel.TfidfModel(working_corpus, id2word=dictionary_), dictionary_, cache


def generate_tfidf_commit(repository: Repository, stopwords_: Set[str], min_len, cache=None) -> Tuple[tfidfmodel.TfidfModel, Dictionary, Dict]:
    if cache is None:
        cache = dict()

    texts = list()
    for commit in repository.commits:
        if commit.c_hash in cache.keys():
            texts.append(cache[commit.c_hash])
        else:
            text = text_pipeline(commit, stopwords_, min_len)
            texts.append(text)
            cache[commit.c_hash] = text
    for issue_ in repository.issues:
        if issue_.id_ in cache.keys():
            texts.append(cache[issue_.id_])
        else:
            text = text_pipeline(issue_, stopwords_, min_len)
            texts.append(text)
            cache[issue_.id_] = text

    dictionary_ = Dictionary(texts)
    dictionary_.filter_extremes(no_below=3, no_above=0.95)
    working_corpus = [dictionary_.doc2bow(text, return_missing=True) for text in texts]
    # Convert UNK from explicit dictionary to UNK token (id = -1)
    working_corpus = [val[0] + [(-1, sum(val[1].values()))] for val in working_corpus]
    return tfidfmodel.TfidfModel(working_corpus, id2word=dictionary_), dictionary_, cache