import pickle
import os
import math

import jsonpickle
import numpy as np
from datetime import timedelta, datetime

from gensim.corpora import Dictionary
from gensim.models import TfidfModel

from Prediction.feature_generation import generate_features
from Prediction.gitScraper import clone_git_repo_to_tmp, get_all_commit_hashes, process_a_commit
from Prediction.training_utils import train_classifier, generate_training_data, generate_dev_fingerprint, \
    generate_tfidf, update, inflate_events, generate_batches, null_issue, flatten_events
from Util import utils_
from Util.ReservedKeywords import java_reserved, c_reserved, cpp_reserved, javascript_reserved, python_reserved
from gitMine.VCClasses import IssueStates, Commit, Issue

stopwords = utils_.GitMineUtils.STOPWORDS \
            + list(set(java_reserved + c_reserved + cpp_reserved + javascript_reserved + python_reserved))


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


class Linker(object):
    def __init__(self, net_size_in_days, min_tok_len, undersample_multiplicity):
        self.repository_obj = None
        self.truth = None
        self.clf = None
        self.fingerprint = None
        self.model = None
        self.dictionary = None
        self.prediction_threshold = 1e-5
        self.net_size_in_days = net_size_in_days
        self.min_tok_len = min_tok_len
        self.undersample_multiplicity = undersample_multiplicity

    def fit(self, repository_obj, truth):
        self.repository_obj = repository_obj
        self.truth = truth
        self.fingerprint = generate_dev_fingerprint(self.repository_obj)
        self.model, self.dictionary = generate_tfidf(self.repository_obj, stopwords, self.min_tok_len)
        self.clf = train_classifier(generate_training_data(self.repository_obj, stopwords, self.fingerprint,
                                                           self.dictionary, self.model, self.truth,
                                                           self.undersample_multiplicity, self.min_tok_len,
                                                           self.net_size_in_days))

    def predict(self, prediction_pr):
        predictions = list()
        # Predict
        prediction_pr.comments = prediction_pr.comments[:1]
        open_issues = [i for i in self.repository_obj.issues
                       if
                       # (len(i.states) == 0 or i.states[-1].to_ == IssueStates.open)
                       # and
                       (min([abs(entity.timestamp - prediction_pr.comments[0].timestamp)
                             if entity.timestamp and prediction_pr.comments
                             else timedelta(days=self.net_size_in_days, seconds=1)
                             for entity in
                             [i.original_post]
                             + i.states
                             + i.actions]) <= timedelta(days=self.net_size_in_days))]
        open_issues += [null_issue]
        prediction_data = list()
        for issue_ in open_issues:
            prediction_data.append(generate_features(issue_, prediction_pr, stopwords, self.fingerprint,
                                                     self.dictionary, self.model, dict(), self.min_tok_len,
                                                     self.net_size_in_days))

        for point in prediction_data:
            probabilities = self.clf.predict_proba(np.array((point.engagement,
                                                             point.cosine_tt,
                                                             point.cosine,
                                                             point.lag,
                                                             point.lag_close,
                                                             point.lag_open,
                                                             point.pr_commits,)).reshape(1, -1))
            prediction = (point.issue, float(probabilities[0][1]))
            predictions.append(prediction)
        predictions = sorted([p for p in predictions if p[1] >= self.prediction_threshold],
                             key=lambda p: (p[1], p[0]),
                             reverse=True)
        return prediction_pr.number, predictions

    def update_and_predict(self, event):
        if isinstance(event[1], Commit):
            if event[0] not in self.repository_obj.commits:
                self.repository_obj.commits.append(event[0])
        elif isinstance(event[1], Issue):
            update(event, self.repository_obj.issues)
        else:
            prediction = self.predict(event[1])
            update(event, self.repository_obj.prs)
            return prediction

    def validate_over_suffix(self, suffix):
        scores = list()
        for event in suffix:
            result = self.update_and_predict(event)
            if result:
                scores.append(result)
        return scores

    def update_from_github(self, since):
        # TODO: Get all new entities from <since> and push them through update
        pass

    def update_from_local_git(self, git_location, since_sha):
        repo_name = self.repository_obj.name
        hashes = get_all_commit_hashes(git_location)
        hashes = hashes[hashes.index(since_sha):]
        commits = list(map(lambda h: process_a_commit(h, repo_name, git_location), hashes))
        for commit in commits:
            if commit not in self.repository_obj.commits:
                self.repository_obj.commits.append(commit)

    def update_truth(self, link):
        """
        Update the inner representation with a new link
        :param link: a link tuple, issue id in first and pr number in second
        """
        try:
            self.truth[link[0]].append(link[1])
        except KeyError:
            self.truth[link[0]] = [link[0]]

    def trim_truth(self):
        issue_ids = [i.id_ for i in self.repository_obj.issues]
        temp = dict()
        for issue_id in self.truth.keys():
            if issue_id in issue_ids:
                temp[issue_id] = self.truth[issue_id]
        self.truth = temp

    def forget_older_than(self, max_age_in_days):
        """
        Trim model state to a set age and retrain classifier
        :param max_age_in_days: The age (in days) of the oldest artifacts to keep
        """
        self.fit(inflate_events(
            list(filter(lambda e: (e[0].timestamp - datetime.now()) <= timedelta(days=max_age_in_days),
                        flatten_events(self.repository_obj))),
            self.repository_obj.name, self.repository_obj.langs), self.truth)
        self.trim_truth()

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
            self.dictionary.save_as_text(os.path.join(path, 'tfidf', 'term2id.txt'))
            self.model.save(os.path.join(path, 'tfidf', 'model.tfidf'))
        # Persist CLF
        if self.clf:
            os.makedirs(os.path.dirname(os.path.join(path, 'clf_model', 'model.p')), exist_ok=True)
            pickle.dump(self.clf, open(os.path.join(path, 'clf_model', 'model.p'), 'wb'))

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
        except FileNotFoundError:
            pass
        try:
            self.clf = pickle.load(open(os.path.join(path, 'clf_model', 'model.p'), 'rb'))
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
        stub = Linker(None, None, None)
        stub.__load_from_disk(path)
        return stub


if __name__ == '__main__':
    location_format = '../data/dev_set/%s.json'
    projects = [
        # 'PhilJay_MPAndroidChart',
        # 'ReactiveX_RxJava',
        # 'palantir_plottable',
        'tensorflow_tensorflow',
    ]

    for project in projects:
        n_batches = 5 if project == 'PhilJay_MPAndroidChart' else 7
        project_dir = location_format % project
        with open(project_dir) as f:
            repo = jsonpickle.decode(f.read())

        with open(project_dir[:-len('.json')] + '_truth.json') as f:
            truth = jsonpickle.decode(f.read())

        batches = generate_batches(repo, n_batches)
        for i in range(n_batches - 2):
            linker = Linker(net_size_in_days=14, min_tok_len=2, undersample_multiplicity=100)
            training = batches[i] + batches[i + 1]
            linker.fit(inflate_events(training, repo.langs, repo.name), truth)
            scores = linker.validate_over_suffix(batches[i + 2])
            scores_dict = dict()
            for pr_id, predictions in scores:
                try:
                    scores_dict[pr_id].union(set(predictions))
                except KeyError:
                    scores_dict[pr_id] = set(predictions)
            for pr_id in scores_dict.keys():
                scores_dict[pr_id] = list(scores_dict[pr_id])
                scores_dict[pr_id] = sorted(scores_dict[pr_id], reverse=True, key=lambda p: (p[1], p[0]))

            with open(project_dir[:-5] + ('_results_f%d.txt' % i), 'w') as f:
                f.write(str(scores_dict))
