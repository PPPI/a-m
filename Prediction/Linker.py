import pickle
import os
import math
from multiprocessing.pool import Pool
from time import sleep, time

import jsonpickle
import numpy as np
from datetime import timedelta, datetime

from gensim.corpora import Dictionary
from gensim.models import TfidfModel

from Prediction.feature_generation import FeatureGenerator
from Prediction.gitScraper import get_all_commit_hashes, process_a_commit
from Prediction.training_utils import generate_training_data as generate_training_data
from Prediction.training_utils import train_classifier, generate_dev_fingerprint, \
    generate_tfidf, update, inflate_events, generate_batches, null_issue, flatten_events, null_pr
from Util import utils_
from Util.ReservedKeywords import java_reserved, c_reserved, cpp_reserved, javascript_reserved, python_reserved
from Util.github_api_methods import parse_pr_ref, parse_issue_ref
from Util.heuristic_methods import extract_issue_numbers
from gitMine.VCClasses import IssueStates, Commit, Issue, PullRequest, StateChange


def evaluate_at_threshold(result, th, truth):
    result_ = [(pr, sorted([(issue, prob)
                            for issue, prob in pred.items() if prob >= th], key=lambda p: (p[1], p[0]), reverse=True))
               for pr, pred in result.items()]
    hit = 0
    ap = 0
    mrr = .0
    dcg = 0
    fpr = 0
    fnr = 0
    renorm = 0

    for pr_nr, predictions in result_:
        predictions = [p[0] for p in predictions]
        if 'null_issue' in predictions:
            predictions = predictions[:predictions.index('null_issue')]

        try:
            expected_truth = truth['#' + pr_nr[len('issue_'):]]
        except KeyError:
            expected_truth = set()

        predictions = [('#' + val[len('issue_'):] in expected_truth) for val in predictions]

        if (len(expected_truth) > 0) and (len(predictions) > 0):
            renorm += 1
            hit += any(predictions)
            curr_ap = np.mean([len([val for val in predictions[:(i + 1)] if val is True]) / (i + 1)
                               for i, j in enumerate(predictions) if j is True])
            if not math.isnan(curr_ap):
                ap += curr_ap
            try:
                mrr += 1 / (predictions.index(True) + 1)
            except ValueError:
                mrr += 0
            dcg += sum([1 / (math.log(i + 1, 2) if i > 1 else 1) if j is True else 0
                        for i, j in enumerate(predictions)])
        elif (len(expected_truth) == 0) and (len(predictions) > 0):
            fpr += 1
        elif (len(expected_truth) > 0) and (len(predictions) == 0):
            fnr += 1
        elif (len(expected_truth) == 0) and (len(predictions) == 0):
            hit += 1

    mrr = mrr / renorm if renorm > 0 else mrr
    ap = ap / renorm if renorm > 0 else ap
    dcg = dcg / renorm if renorm > 0 else dcg
    hit /= len(result_)
    fpr /= len(result_)
    fnr /= len(result_)
    return mrr, ap, dcg, hit, fpr, fnr


# Closures needed to be able to generate features in a multi-threaded way.
class Issue_Closure(object):
    def __init__(self, prediction_object, feature_generator):
        self.prediction_object = prediction_object
        self.feature_generator = feature_generator

    def __call__(self, issue):
        return self.feature_generator.generate_features(issue, self.prediction_object, False)


class PR_Closure(object):
    def __init__(self, prediction_object, feature_generator):
        self.prediction_object = prediction_object
        self.feature_generator = feature_generator

    def __call__(self, pr):
        return self.feature_generator.generate_features(self.prediction_object, pr, False)


# Basic wrapper to re-attempt with exponential back-off
def __try_and_get__(via, max_attempts, args):
    attempt = 0
    while attempt < max_attempts:
        try:
            return via(*args)
        except Exception:
            sleep(2**attempt)
        attempt += 1


class Linker(object):
    """
    Main class that encapsulates the concept of a link prediction engine. It is constructed using a configuration as
    follows:
    net_size_in_days : The size of the window of interest around a specific artefact's timestamp when shortlisting
        candidates
    undersample_multiplicity : What is the desired class imbalance we wish to reach by false links uniformly at random
    feature_config : A dictionary of str to bool that indicates which features will be used in classification
    predictions_between_updates : How many new artefacts do we admit to the corpus before recomputing tf-idf values
    min_tok_len : the length that a token must have to be included in the corpus
    stopwords : a set of words to be excluded from the corpus
    """
    def __init__(self, net_size_in_days, undersample_multiplicity, feature_config, predictions_between_updates=None,
                 min_tok_len=None, stopwords=None):
        self.repository_obj = None
        self.truth = None
        self.clf = None
        self.fingerprint = None
        self.model = None
        self.dictionary = None
        self.feature_generator = None
        self.features = None
        self.prediction_threshold = 1e-5
        self.predictions_from_last_tf_idf_update = 0
        self.predictions_between_updates = predictions_between_updates
        self.stopwords = stopwords
        self.net_size_in_days = net_size_in_days
        self.min_tok_len = min_tok_len
        self.undersample_multiplicity = undersample_multiplicity
        self.use_sim_cs = feature_config['use_sim_cs']
        self.use_sim_j = feature_config['use_sim_j']
        self.use_sim_d = feature_config['use_sim_d']
        self.use_social = feature_config['use_social']
        self.use_temporal = feature_config['use_temporal']
        self.use_file = feature_config['use_file']
        self.use_pr_only = feature_config['use_pr_only']
        self.use_issue_only = feature_config['use_issue_only']
        if self.use_sim_cs or self.use_sim_j or self.use_sim_d or self.use_file:
            assert self.predictions_between_updates is not None
            assert self.min_tok_len is not None
            assert self.stopwords is not None

    def fit(self, repository_obj, truth, features=None):
        self.repository_obj = repository_obj
        self.truth = truth
        self.features = features

        similarity_config = None
        temporal_config = None
        cache = self.feature_generator.text_cache if self.feature_generator else None
        if self.use_sim_cs or self.use_sim_j or self.use_sim_d or self.use_file:
            self.model, self.dictionary, cache = generate_tfidf(self.repository_obj, self.stopwords, self.min_tok_len,
                                                                cache=cache)
            similarity_config = {
                'dict': self.dictionary,
                'model': self.model,
                'min_len': self.min_tok_len,
                'stopwords': self.stopwords,
            }
        if self.use_temporal:
            self.fingerprint = generate_dev_fingerprint(self.repository_obj)
            temporal_config = {
                'fingerprint': self.fingerprint,
                'net_size_in_days': self.net_size_in_days,
            }
        self.feature_generator = FeatureGenerator(
            use_file=self.use_file,
            use_sim_cs=self.use_sim_cs,
            use_sim_j=self.use_sim_j,
            use_sim_d=self.use_sim_d,
            use_social=self.use_social,
            use_temporal=self.use_temporal,
            use_pr_only=self.use_pr_only,
            use_issue_only=self.use_issue_only,
            similarity_config=similarity_config,
            temporal_config=temporal_config,
            text_cache=cache,
            selected=self.features,
        )
        self.clf = train_classifier(generate_training_data(self.repository_obj,
                                                           self.feature_generator,
                                                           self.net_size_in_days,
                                                           truth,
                                                           self.undersample_multiplicity))

    def predict(self, prediction_object):
        threshold = self.prediction_threshold
        predictions = list()
        if isinstance(prediction_object, PullRequest):
            # Predict
            prediction_object.comments = prediction_object.comments[:1]
            open_issues = [i for i in self.repository_obj.issues
                           if
                           (len(i.states) == 0 or i.states[-1].to_ == IssueStates.open)
                           or
                           (min([abs(entity.timestamp - prediction_object.comments[0].timestamp)
                                 if entity.timestamp and prediction_object.comments
                                 else timedelta(days=self.net_size_in_days, seconds=1)
                                 for entity in
                                 [i.original_post]
                                 + i.states
                                 + i.actions]) <= timedelta(days=self.net_size_in_days))]
            open_issues += [null_issue]
            prediction_data = list()

            if len(open_issues) > 128:
                with Pool(processes=os.cpu_count() - 1) as wp:
                    for point in wp.map(func=Issue_Closure(prediction_object, self.feature_generator),
                                        iterable=open_issues, chunksize=128):
                        prediction_data.append(point)
            else:
                for issue in open_issues:
                    prediction_data.append(self.feature_generator.generate_features(issue, prediction_object, False))

            for point in prediction_data:
                probabilities = self.clf.predict_proba(np.array(tuple([v for k, v in point.items()
                                                                       if k not in ['linked', 'issue', 'pr']]))
                                                       .reshape(1, -1))
                if point['issue'] == 'null_issue':
                    threshold = max(threshold, probabilities[0][1])
                else:
                    prediction = (point['issue'], float(probabilities[0][1]))
                    predictions.append(prediction)
            predictions = sorted([p for p in predictions if p[1] >= threshold],
                                 key=lambda p: (p[1], p[0]),
                                 reverse=True)
            response = prediction_object.number, predictions
        elif isinstance(prediction_object, Issue):
            # Predict
            candidates = [p for p in self.repository_obj.prs
                          if
                          (min([abs(entity.timestamp - p.comments[0].timestamp)
                                if entity.timestamp and p.comments
                                else timedelta(days=self.net_size_in_days, seconds=1)
                                for entity in
                                [prediction_object.original_post]
                                + prediction_object.states
                                + prediction_object.actions]) <= timedelta(days=self.net_size_in_days))]
            candidates += [null_pr]
            prediction_data = list()

            if len(candidates) > 128:
                with Pool(processes=os.cpu_count() - 1) as wp:
                    for point in wp.map(func=PR_Closure(prediction_object, self.feature_generator),
                                        iterable=candidates, chunksize=128):
                        prediction_data.append(point)
            else:
                for pr in candidates:
                    prediction_data.append(self.feature_generator.generate_features(prediction_object, pr, False))

            for point in prediction_data:
                probabilities = self.clf.predict_proba(np.array(tuple([v for k, v in point.items()
                                                                       if k not in ['linked', 'issue', 'pr']]))
                                                       .reshape(1, -1))
                if point['pr'] == 'null_pr':
                    threshold = max(threshold, probabilities[0][1])
                else:
                    prediction = (point['pr'], float(probabilities[0][1]))
                    predictions.append(prediction)
            predictions = sorted([p for p in predictions if p[1] >= threshold],
                                 key=lambda p: (p[1], p[0]),
                                 reverse=True)
            response = prediction_object.id_, predictions
        if self.use_sim_cs or self.use_sim_j or self.use_sim_d or self.use_file:
            if self.predictions_from_last_tf_idf_update < self.predictions_between_updates:
                self.predictions_from_last_tf_idf_update += 1
            else:
                self.predictions_from_last_tf_idf_update = 0
                temporal_config = None
                self.model, self.dictionary, new_cache = generate_tfidf(self.repository_obj, self.stopwords,
                                                                        self.min_tok_len,
                                                                        cache=self.feature_generator.text_cache)
                similarity_config = {
                    'dict': self.dictionary,
                    'model': self.model,
                    'min_len': self.min_tok_len,
                    'stopwords': self.stopwords,
                }
                if self.use_temporal:
                    self.fingerprint = generate_dev_fingerprint(self.repository_obj)
                    temporal_config = {
                        'fingerprint': self.fingerprint,
                        'net_size_in_days': self.net_size_in_days,
                    }
                self.feature_generator = FeatureGenerator(
                    use_file=self.use_file,
                    use_sim_cs=self.use_sim_cs,
                    use_sim_j=self.use_sim_j,
                    use_sim_d=self.use_sim_d,
                    use_social=self.use_social,
                    use_temporal=self.use_temporal,
                    use_pr_only=self.use_pr_only,
                    use_issue_only=self.use_issue_only,
                    similarity_config=similarity_config,
                    temporal_config=temporal_config,
                    text_cache=new_cache,
                    selected=self.features,
                )
        return response

    def id_to_title(self, id_):
        title = [e.title for e in self.repository_obj.issues + self.repository_obj.prs
                 if (e.number if isinstance(e, PullRequest) else e.id_) == id_][0]
        return title

    def most_recent_sha(self):
        return sorted(map(lambda c: [c, setattr(c, 'timestamp', c.timestamp.replace(tzinfo=None))][0],
                          self.repository_obj.commits), key=lambda c: c.timestamp)[-1].c_hash

    def most_recent_timestamp(self):
        return sorted(flatten_events(self.repository_obj), key=lambda e: e[0].timestamp)[-1][0].timestamp

    def update_from_flat_repo_and_predict(self, event):
        if isinstance(event[1], Commit):
            if event[0] not in self.repository_obj.commits:
                self.repository_obj.commits.append(event[0])
        elif isinstance(event[1], Issue):
            prediction = None
            if len([i for i in self.repository_obj.issues if i.id_ == event[1].id_]) == 0 or \
                (isinstance(event[0], StateChange) and event[0].to_ == IssueStates.closed):
                prediction = self.predict(event[1])
            update(event, self.repository_obj.issues)
            return prediction
        else:
            prediction = None
            if len([p for p in self.repository_obj.prs if p.number == event[1].id_]) == 0 or \
                (isinstance(event[0], StateChange) and event[0].to_ == IssueStates.merged):
                prediction = self.predict(event[1])
            update(event, self.repository_obj.prs)
            return prediction

    def request_prediction(self, issue_or_pr):
        # if isinstance(issue_or_pr, Issue):
        #     old_issue = [i for i in self.repository_obj.issues if i.id_ == issue_or_pr.id_]
        #     if len(old_issue) == 1:
        #         self.repository_obj.issues.remove(old_issue)
        #     self.repository_obj.issues.append(issue_or_pr)
        # elif isinstance(issue_or_pr, PullRequest):
        #     old_pr = [p for p in self.repository_obj.prs if p.number == issue_or_pr.number]
        #     if len(old_pr) == 1:
        #         self.repository_obj.prs.remove(old_pr)
        #     self.repository_obj.prs.append(issue_or_pr)
        return self.predict(issue_or_pr)[1]

    def validate_over_suffix(self, suffix):
        scores = list()
        unk_rate = list()
        for event in suffix:
            result = self.update_from_flat_repo_and_predict(event)
            if result:
                scores.append(result)
                id_, predictions = result
                predictions = [t[0][len('issue_'):] for t in predictions[:5]]
                id_ = id_[len('issue_'):]
                UNKs = self.feature_generator.get_tf(self.feature_generator.via_text_cache(id_, event[1]))[-1][-1]
                unk_rate.append(UNKs)
                if isinstance(event[1], Issue):
                    for other in predictions:
                        try:
                            if ('#' + other) in self.truth['#' + id_]:
                                self.update_truth((id_, other))
                            else:
                                self.update_truth((id_, other), is_true=False)
                        except KeyError:
                            self.update_truth((id_, other), is_true=False)
                elif isinstance(event[1], PullRequest):
                    for other in predictions:
                        try:
                            if ('#' + id_) in self.truth['#' + other]:
                                self.update_truth((other, id_))
                            else:
                                self.update_truth((other, id_), is_true=False)
                        except KeyError:
                            self.update_truth((other, id_), is_true=False)
        return scores, unk_rate

    def update_from_github(self, gh, since):
        """
        Update the PR and Issue internal state of the backend using the GitHub REST API
        :param gh: A GitHub client to use during the update
        :param since: Only issues updated at or after this time are returned.
                      This is a timestamp in ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ.
        """
        repo = gh.get_repo(self.repository_obj.name.replace('_', '/')[1:])
        pr_numbers = [pr.number for pr in self.repository_obj.prs]
        issue_ids = [issue.id_ for issue in self.repository_obj.issues]
        pr_refs = [ref for ref in repo.get_pulls(state='all')]
        for pr_ref in pr_refs:
            pr = None
            while pr is None:
                if gh.rate_limiting[0] == 0:
                    sleep(gh.rate_limiting_resettime - time())
                pr = __try_and_get__(parse_pr_ref, 20, (pr_ref, self.repository_obj.name))
            if pr_ref.number in pr_numbers:
                old_pr = [pr for pr in self.repository_obj.prs if pr.number == pr_ref.number][0]
                self.repository_obj.prs.remove(old_pr)
            self.repository_obj.prs.append(pr)
            try:
                all_text = '\n'.join([c.body for c in pr.comments] + [c.title + c.desc for c in pr.commits])
                issue_numbers = extract_issue_numbers(all_text)
                issue_numbers = list(filter(lambda id_: ('issue_' + id_[1:]) in issue_ids, issue_numbers))
                for issue_id in issue_numbers:
                    self.update_truth((issue_id[1:], pr.number[len('issue_'):]))
            except TypeError:
                pass

        issue_refs = [ref for ref in repo.get_issues(state='all', since=since)]
        for issue_ref in issue_refs:
            issue = None
            while issue is None:
                if gh.rate_limiting[0] == 0:
                    sleep(gh.rate_limiting_resettime - time())
                issue = __try_and_get__(parse_issue_ref, 20, tuple([issue_ref]))
            if issue.id_ in issue_ids:
                existing_issue = [i for i in self.repository_obj.issues if i.id_ == issue.id_][0]
                self.repository_obj.issues.remove(existing_issue)
            self.repository_obj.issues.append(issue)
            try:
                all_text = '\n'.join([c.body for c in pr.comments] + [c.title + c.desc for c in pr.commits])
                issue_numbers = extract_issue_numbers(all_text)
                issue_numbers = list(filter(lambda id_: ('issue_' + id_[1:]) in pr_numbers, issue_numbers))
                for pr_id in issue_numbers:
                    self.update_truth((issue.id_[len('issue_'):], pr_id))
            except TypeError:
                pass

    def update_from_local_git(self, git_location, since_sha):
        """
        Update the internal state of the commits in the backend using the local git repository.
        :param git_location: Location in the filesystem where the git repository is located
        :param since_sha: The sha after which we wish to update
        """
        repo_name = self.repository_obj.name
        hashes = get_all_commit_hashes(git_location)
        try:
            hashes = hashes[:hashes.index(since_sha):]
        except ValueError:
            current_index = -1
            sorted_commits = sorted(self.repository_obj.commits, key=lambda c: c.timestamp)
            while sorted_commits[current_index].c_hash not in hashes:
                current_index -= 1
            since_sha = sorted_commits[current_index].c_hash
            hashes = hashes[:hashes.index(since_sha):]
        commits = list(map(lambda h: process_a_commit(h, repo_name, git_location), hashes))
        for commit in commits:
            if len([_ for _ in self.repository_obj.commits if _.c_hash.startswith(commit.c_hash)]) > 0:
                self.repository_obj.commits.append(commit)

    def update_truth(self, link, is_true=True):
        """
        Update the inner representation with a new link
        :param link: a link tuple, issue id in first and pr number in second
        :param is_true: If the link is a true link or if we only wish to do a model update
        """
        if is_true:
            try:
                if ('#' + link[1]) not in self.truth['#' + link[0]]:
                    self.truth['#' + link[0]].append('#' + link[1])
            except KeyError:
                self.truth['#' + link[0]] = ['#' + link[1]]
        try:
            point = self.feature_generator.generate_features(
                    [i for i in self.repository_obj.issues if i.id_[len('issue_'):] == link[0]][0],
                    [p for p in self.repository_obj.prs if p.number[len('issue_'):] == link[1]][0],
                    linked=True)
            self.clf = self.clf.partial_fit([tuple([v for k, v in point.items() if k not in ['linked', 'issue', 'pr']])], [1])
        except IndexError:
            pass

    def trim_truth(self):
        """
        Method to remove ground truth data regarding entities that no longer exist in the internal representation.
        """
        issue_ids = [i.id_ for i in self.repository_obj.issues]
        temp = dict()
        for issue_id in self.truth.keys():
            if ('issue_' + issue_id[1:]) in issue_ids:
                temp[issue_id] = self.truth[issue_id]
        self.truth = temp

    def forget_older_than(self, max_age_in_days):
        """
        Trim model state to a set age and retrain classifier
        :param max_age_in_days: The age (in days) of the oldest artifacts to keep
        """
        self.fit(inflate_events(
            list(filter(lambda e: (e[0].timestamp - datetime.now(tz=None)) <= timedelta(days=max_age_in_days),
                        flatten_events(self.repository_obj))),
            name=self.repository_obj.name, langs=self.repository_obj.langs), self.truth)
        self.trim_truth()

    def get_mean_probability_of_true_link(self):
        prediction_data = list()
        predictions = list()

        for issue_id, pr_ids in self.truth.items():
            issue = [i for i in self.repository_obj.issues if i.id_ == 'issue_' + issue_id[1:]][0]
            for pr_id in pr_ids:
                pr = [p for p in self.repository_obj.prs if p.number == 'issue_' + pr_id[1:]][0]
                prediction_data.append(self.feature_generator.generate_features(issue, pr, True))

        for point in prediction_data:
            probabilities = self.clf.predict_proba(np.array(tuple([v for k, v in point.items()
                                                                   if k not in ['linked', 'issue', 'pr']]))
                                                   .reshape(1, -1))
            predictions.append(probabilities[0][1])

        return float(np.mean(predictions))

    def persist_to_disk(self, path):
        """
        Function to save the class to disc
        :param path: The location where the class should be saved, creates folders if necessary
        :return: None
        """
        # Build config
        name = self.repository_obj.name.translate({ord(c): '_' for c in '\\/'})
        params = {
            'name': name,
            'net_size_in_days': self.net_size_in_days,
            'min_tok_len': self.min_tok_len,
            'undersample_multiplicity': self.undersample_multiplicity,
            'prediction_threshold': self.prediction_threshold,
            'use_sim_cs': self.use_sim_cs,
            'use_sim_j': self.use_sim_j,
            'use_sim_d': self.use_sim_d,
            'use_social': self.use_social,
            'use_temporal': self.use_temporal,
            'use_file': self.use_file,
            'use_pr_only': self.use_pr_only,
            'use_issue_only': self.use_issue_only,
            'predictions_between_updates': self.predictions_between_updates,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(os.path.join(path, 'config.json'), 'w') as f:
            f.write(jsonpickle.encode(params))
        # Persist corpora
        if self.repository_obj and self.truth:
            os.makedirs(os.path.dirname(os.path.join(path, name, 'repository_data.json')), exist_ok=True)
            with open(os.path.join(path, name, 'repository_data.json'), 'w') as f:
                f.write(jsonpickle.encode(self.repository_obj))
            with open(os.path.join(path, name, 'truth_data.json'), 'w') as f:
                f.write(jsonpickle.encode(self.truth))
        if self.fingerprint:
            with open(os.path.join(path, name, 'fingerprint_data.json'), 'w') as f:
                f.write(jsonpickle.encode(self.fingerprint))
        if self.model and self.dictionary:
            os.makedirs(os.path.join(path, 'tfidf'), exist_ok=True)
            with open(os.path.join(path, name, 'stopwords_data.json'), 'w') as f:
                f.write(jsonpickle.encode(self.stopwords))
            self.dictionary.save_as_text(os.path.join(path, 'tfidf', 'term2id.txt'))
            self.model.save(os.path.join(path, 'tfidf', 'model.tfidf'))
        # Persist CLF
        if self.clf:
            os.makedirs(os.path.dirname(os.path.join(path, 'clf_model', 'model.p')), exist_ok=True)
            pickle.dump(self.clf, open(os.path.join(path, 'clf_model', 'model.p'), 'wb'))
        if self.feature_generator:
            os.makedirs(os.path.dirname(os.path.join(path, 'feature_generator', 'gen.p')), exist_ok=True)
            pickle.dump(self.feature_generator, open(os.path.join(path, 'feature_generator', 'gen.p'), 'wb'))

    def __load_from_disk(self, path):
        """
        Function that is used internally to load and set-up the class state
        :param path: Location from where the class internal state should be loaded
        :return: None, side-effect on the class on which this is called
        """
        # Read config,
        with open(os.path.join(path, 'config.json')) as f:
            params = jsonpickle.decode(f.read())
        self.net_size_in_days = params['net_size_in_days']
        self.min_tok_len = params['min_tok_len']
        self.undersample_multiplicity = params['undersample_multiplicity']
        self.prediction_threshold = params['prediction_threshold']
        self.use_sim_cs = params['use_sim_cs']
        self.use_sim_j = params['use_sim_j']
        self.use_sim_d = params['use_sim_d']
        self.use_social = params['use_social']
        self.use_temporal = params['use_temporal']
        self.use_file = params['use_file']
        self.use_pr_only = params['use_pr_only']
        self.use_issue_only = params['use_issue_only']
        self.predictions_between_updates = params['predictions_between_updates']
        name = params['name']
        try:
            with open(os.path.join(path, name, 'repository_data.json')) as f:
                self.repository_obj = jsonpickle.decode(f.read())
            with open(os.path.join(path, name, 'truth_data.json')) as f:
                self.truth = jsonpickle.decode(f.read())
        except FileNotFoundError:
            pass
        try:
            with open(os.path.join(path, name, 'fingerprint_data.json')) as f:
                self.fingerprint = jsonpickle.decode(f.read())
        except FileNotFoundError:
            pass
        try:
            self.dictionary = Dictionary.load_from_text(os.path.join(path, 'tfidf', 'term2id.txt'))
            self.model = TfidfModel.load(os.path.join(path, 'tfidf', 'model.tfidf'))
            with open(os.path.join(path, name, 'stopwords_data.json')) as f:
                self.stopwords = jsonpickle.decode(f.read())
        except FileNotFoundError:
            pass
        try:
            self.clf = pickle.load(open(os.path.join(path, 'clf_model', 'model.p'), 'rb'))
        except FileNotFoundError:
            pass
        try:
            self.feature_generator = pickle.load(open(os.path.join(path, 'feature_generator', 'gen.p'), 'rb'))
        except FileNotFoundError:
            pass

    @staticmethod
    def load_from_disk(path):
        """
        Static function to load a TopicalLinker class from disc
        :param path: The location from where the internal state of tha class should be loaded
        :return: A TopicalLinker object that has the internal state specified by the files at path
        """
        # We only want to ignore the type-check here
        # noinspection PyTypeChecker
        stub = Linker(None, None, feature_config={
            'use_issue_only': True,
            'use_pr_only': True,
            'use_temporal': True,
            'use_sim_cs': True,
            'use_sim_j': True,
            'use_sim_d': True,
            'use_file': True,
            'use_social': True
        }, predictions_between_updates=-1, min_tok_len=-1, stopwords=set())
        stub.__load_from_disk(path)
        return stub


if __name__ == '__main__':
    # import pandas as pd

    location_format = '../data/dev_set/%s.json'
    projects = [
        'PhilJay_MPAndroidChart',
        # 'ReactiveX_RxJava',
        'palantir_plottable',
        # 'tensorflow_tensorflow',
    ]
    config = {
        'use_issue_only': True,
        'use_pr_only': True,
        'use_temporal': True,
        'use_sim_cs': True,
        'use_sim_j': False,
        'use_sim_d': False,
        'use_file': True,
        'use_social': True
    }
    # features_string = \
    #     ('cosine cosine_tt cosine_tc cosine_ct cosine_cc ' if config['use_sim_cs'] else '') + \
    #     ('jaccard jaccard_tt jaccard_tc jaccard_ct jaccard_cc ' if config['use_sim_j'] else '') + \
    #     ('dice dice_tt dice_tc dice_ct dice_cc ' if config['use_sim_d'] else '') + \
    #     ('files_shared ' if config['use_file'] else '') + \
    #     ('is_reporter is_assignee engagement in_top_2 in_comments ' if config['use_social'] else '') + \
    #     ('developer_normalised_lag lag_from_issue_open_to_pr_submission lag_from_last_issue_update_to_pr_submission '
    #      if config['use_temporal'] else '') + \
    #     ('no_pr_desc branch_size files_touched_by_pr ' if config['use_pr_only'] else '') + \
    #     ('report_size participants bounces existing_links ' if config['use_issue_only'] else '')
    # features_string = features_string.strip().split(' ')
    # index_feature_map = {i: features_string[i] for i in range(len(features_string))}
    features = ['cosine_tc', 'report_size', 'branch_size', 'files_touched_by_pr', 'developer_normalised_lag']
    stopwords = utils_.GitMineUtils.STOPWORDS \
                + list(set(java_reserved + c_reserved + cpp_reserved + javascript_reserved + python_reserved))
    for project in projects:
        n_batches = 5
        project_dir = location_format % project
        with open(project_dir) as f:
            repo = jsonpickle.decode(f.read())

        with open(project_dir[:-len('.json')] + '_truth.json') as f:
            truth = jsonpickle.decode(f.read())

        batches = generate_batches(repo, n_batches)
        for i in [n_batches - 1]:
            linker = Linker(net_size_in_days=14, min_tok_len=3, undersample_multiplicity=1000, stopwords=stopwords,
                            feature_config=config, predictions_between_updates=1000)
            training = list()
            for j in range(n_batches - 1):
                training += batches[j]
            linker.fit(inflate_events(training, repo.langs, repo.name), truth, features=features)
            # forest = linker.clf
            # importances = forest.feature_importances_
            # std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
            # pd.DataFrame(data={'Feature': features_string, 'Importance': importances, 'STD': std}) \
            #     .to_csv(project_dir[:-5] + ('_results_f%d_NullExplicit_UNKExplicit_FullFeatures_IMP.csv' % i))
            scores, unk_rate = linker.validate_over_suffix(batches[i])
            scores_dict = dict()
            for pr_id, predictions in scores:
                try:
                    scores_dict[pr_id].union(set(predictions))
                except KeyError:
                    scores_dict[pr_id] = set(predictions)
            for pr_id in scores_dict.keys():
                scores_dict[pr_id] = list(scores_dict[pr_id])
                scores_dict[pr_id] = sorted(scores_dict[pr_id], reverse=True, key=lambda p: (p[1], p[0]))

            with open(project_dir[:-5] + ('_results_f%d_selected_features_MF.txt' % i), 'w') as f:
                f.write(str(scores_dict))

            with open(project_dir[:-5] + ('_unk_rate_f%d_selected_features_MF.txt' % i), 'w') as f:
                f.write(str(unk_rate))
