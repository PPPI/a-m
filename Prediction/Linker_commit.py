import sys

import jsonpickle
import numpy as np
import pandas as pd
import os
from datetime import timedelta
from multiprocessing.pool import Pool

from Prediction.Linker import Linker
from Prediction.feature_generation import FeatureGenerator
from Prediction.training_utils import train_classifier, \
    generate_training_data_commit, generate_tfidf_commit, null_issue, null_commit
from Util import utils_
from Util.CrossValidationSplits import cross_split_repo
from Util.ReservedKeywords import java_reserved, c_reserved, cpp_reserved, javascript_reserved, python_reserved
from gitMine.VCClasses import Commit, Issue, IssueStates


class Issue_Closure(object):
    def __init__(self, prediction_object, feature_generator):
        self.prediction_object = prediction_object
        self.feature_generator = feature_generator

    def __call__(self, issue):
        return self.feature_generator.generate_features_commit(issue, self.prediction_object, False)


class Commit_Closure(object):
    def __init__(self, prediction_object, feature_generator):
        self.prediction_object = prediction_object
        self.feature_generator = feature_generator

    def __call__(self, commit):
        return self.feature_generator.generate_features_commit(self.prediction_object, commit, False)


class CommitLinker(Linker):
    def __init__(self, net_size_in_days, undersample_multiplicity, feature_config, stopwords=None):
        super().__init__(net_size_in_days, undersample_multiplicity, feature_config, predictions_between_updates=10e15,
                         min_tok_len=2, stopwords=stopwords)

    def fit(self, repository_obj, truth=None, features=None):
        self.repository_obj = repository_obj
        self.features = features

        similarity_config = None
        temporal_config = None
        cache = self.feature_generator.text_cache if self.feature_generator else None
        if self.use_sim_cs or self.use_sim_j or self.use_sim_d or self.use_file:
            self.model, self.dictionary, cache = generate_tfidf_commit(self.repository_obj, self.stopwords,
                                                                       self.min_tok_len, cache=cache)
            similarity_config = {
                'dict': self.dictionary,
                'model': self.model,
                'min_len': self.min_tok_len,
                'stopwords': self.stopwords,
            }
        if self.use_temporal:
            temporal_config = {
                'fingerprint': None,
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
        self.clf = train_classifier(generate_training_data_commit(self.repository_obj,
                                                                  self.feature_generator,
                                                                  self.net_size_in_days,
                                                                  self.undersample_multiplicity))

    def validate_over_suffix(self, suffix):
        scores = list()
        unk_rate = list()
        for event in suffix:
            result = self.update_from_flat_repo_and_predict(event)
            if result:
                scores.append(result)
                id_, _ = result
                id_ = id_[len('issue_'):]
                UNKs = self.feature_generator.get_tf(self.feature_generator.via_text_cache(id_, event[1]))[-1][-1]
                unk_rate.append(UNKs)
        return scores, unk_rate

    def predict(self, prediction_object):
        threshold = self.prediction_threshold
        predictions = list()
        if isinstance(prediction_object, Commit):
            # Predict
            open_issues = [i for i in self.repository_obj.issues
                           if
                           # (len(i.states) == 0 or i.states[-1].to_ == IssueStates.open)
                           # or
                           (min([abs(entity.timestamp - prediction_object.timestamp)
                                 if hasattr(entity, 'timestamp') and entity.timestamp
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
                    prediction_data.append(self.feature_generator.generate_features_commit(issue, prediction_object,
                                                                                           False))

            for point in prediction_data:
                probabilities = self.clf.predict_proba(np.array(tuple([v for k, v in point.items()
                                                                       if k not in ['linked', 'issue', 'commit']]))
                                                       .reshape(1, -1))
                if point['issue'] == 'null_issue':
                    threshold = max(threshold, probabilities[0][1])
                else:
                    prediction = (point['issue'], float(probabilities[0][1]))
                    predictions.append(prediction)
            predictions = sorted([p for p in predictions if p[1] >= threshold],
                                 key=lambda p: (p[1], p[0]),
                                 reverse=True)
            response = prediction_object.c_hash, predictions
        elif isinstance(prediction_object, Issue):
            # Predict
            candidates = [c for c in self.repository_obj.commits
                          if
                          (min([abs(entity.timestamp - c.timestamp)
                                if hasattr(entity, 'timestamp') and entity.timestamp
                                else timedelta(days=self.net_size_in_days, seconds=1)
                                for entity in
                                [prediction_object.original_post]
                                + prediction_object.states
                                + prediction_object.actions]) <= timedelta(days=self.net_size_in_days))]
            candidates += [null_commit]
            prediction_data = list()

            if len(candidates) > 128:
                with Pool(processes=os.cpu_count() - 1) as wp:
                    for point in wp.map(func=Commit_Closure(prediction_object, self.feature_generator),
                                        iterable=candidates, chunksize=128):
                        prediction_data.append(point)
            else:
                for commit in candidates:
                    prediction_data.append(self.feature_generator.generate_features_commit(prediction_object,
                                                                                           commit, False))

            for point in prediction_data:
                probabilities = self.clf.predict_proba(np.array(tuple([v for k, v in point.items()
                                                                       if k not in ['linked', 'issue', 'commit']]))
                                                       .reshape(1, -1))
                if point['commit'] == 'null_commit':
                    threshold = max(threshold, probabilities[0][1])
                else:
                    prediction = (point['commit'], float(probabilities[0][1]))
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
                self.model, self.dictionary, new_cache = generate_tfidf_commit(self.repository_obj, self.stopwords,
                                                                               self.min_tok_len,
                                                                               cache=self.feature_generator.text_cache)
                similarity_config = {
                    'dict': self.dictionary,
                    'model': self.model,
                    'min_len': self.min_tok_len,
                    'stopwords': self.stopwords,
                }
                if self.use_temporal:
                    self.fingerprint = None
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


if __name__ == '__main__':
    n_folds = 10
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
    features = None
    stopwords = utils_.GitMineUtils.STOPWORDS \
                + list(set(java_reserved + c_reserved + cpp_reserved + javascript_reserved + python_reserved))
    threshold = .5
    with open(sys.argv[1]) as f:
        repo = jsonpickle.decode(f.read())
    train, test, links = cross_split_repo(repo, n_folds)
    clinker = CommitLinker(feature_config=config, net_size_in_days=7, undersample_multiplicity=1,
                           stopwords=stopwords)
    results = {'Fold': list(),
               'True Positives': list(),
               'False Positives': list(),
               'False Negatives': list(),
               'Precision': list(),
               'Recall': list(),
               'F1': list()}
    for i in range(n_folds):
        clinker.fit(train[i], features=features)
        truth = links[i]

        final_suggestions = dict()

        scores = list()
        for commit in test[i].commits:
            id_, pred = clinker.predict(commit)
            pred = [other for other, prob in pred if prob >= threshold]
            scores.append((id_, pred))

        for id_, pred in scores:
            for issue in pred:
                try:
                    final_suggestions[issue].append(id_)
                except KeyError:
                    final_suggestions[issue] = [id_]

        scores = list()
        for issue in test[i].issues:
            id_, pred = clinker.predict(issue)
            pred = [other for other, prob in pred if prob >= threshold]
            scores.append((id_, pred))

        for id_, pred in scores:
            for commit in pred:
                try:
                    final_suggestions[id_].append(commit)
                except KeyError:
                    final_suggestions[id_] = [commit]

        tp = 0
        fp = 0
        fn = 0
        for issue in final_suggestions.keys():
            try:
                expected = set(truth[issue])
            except KeyError:
                expected = set()
            predictions = set(final_suggestions[issue])
            tp += len(expected.intersection(predictions))
            fp += len(predictions.difference(expected))
            fn += len(expected.difference(predictions))

        for issue in truth.keys():
            if issue not in final_suggestions.keys():
                fn += len(truth[issue])

        p = tp / (tp + fp) if (tp + fp) > 0 else .0
        r = tp / (tp + fn) if (tp + fn) > 0 else .0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else .0
        results['Fold'].append(i)
        results['True Positives'].append(tp)
        results['False Positives'].append(fp)
        results['False Negatives'].append(fn)
        results['Precision'].append(p)
        results['Recall'].append(r)
        results['F1'].append(f1)
    pd.DataFrame(data=results).to_csv(sys.argv[1][:-5] + '_results.csv')
