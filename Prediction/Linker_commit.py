import numpy as np
import os
from datetime import timedelta
from multiprocessing.pool import Pool

from Prediction.Linker import Linker
from Prediction.feature_generation import FeatureGenerator
from Prediction.training_utils import generate_dev_fingerprint, train_classifier, \
    generate_training_data_commit, generate_tfidf_commit, null_issue, null_commit
from gitMine.VCClasses import Commit, Issue


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
    def __init__(self, net_size_in_days, undersample_multiplicity, feature_config):
        super().__init__(net_size_in_days, undersample_multiplicity, feature_config)

    def fit(self, repository_obj, truth, features=None):
        self.repository_obj = repository_obj
        self.truth = truth
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
                                 if entity.timestamp
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
                                if entity.timestamp
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
                    prediction_data.append(self.feature_generator.generate_features(prediction_object, commit, False))

            for point in prediction_data:
                probabilities = self.clf.predict_proba(np.array(tuple([v for k, v in point.items()
                                                                       if k not in ['linked', 'issue', 'commit']]))
                                                       .reshape(1, -1))
                if point['commit'] == 'null_commit':
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
