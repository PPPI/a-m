import jsonpickle

from Prediction.feature_generation import FeatureGenerator
from Prediction.training_utils import generate_training_data, generate_tfidf, generate_dev_fingerprint
from Util import utils_
from Util.ReservedKeywords import java_reserved, c_reserved, cpp_reserved, javascript_reserved, python_reserved

if __name__ == '__main__':
    stopwords = utils_.GitMineUtils.STOPWORDS \
                + list(set(java_reserved + c_reserved + cpp_reserved + javascript_reserved + python_reserved))
    min_tok_len = 3
    net_size_in_days = 14
    multiplicity = 1000
    repo_locations = ['../data/dev_set/%s.json' % s for s in [
        'PhilJay_MPAndroidChart',
        'ReactiveX_RxJava',
        'palantir_plottable',
        'tensorflow_tensorflow',
    ]]

    for repo_loc in repo_locations:
        with open(repo_loc) as f:
            repo = jsonpickle.decode(f.read())
        with open(repo_loc[:-5] + '_truth.json') as f:
            truth = jsonpickle.decode(f.read())
        model, dictionary = generate_tfidf(repo, stopwords, min_tok_len)
        similarity_config = {
            'dict': dictionary,
            'model': model,
            'min_len': min_tok_len,
            'stopwords': stopwords,
        }

        fingerprint = generate_dev_fingerprint(repo)
        temporal_config = {
            'fingerprint': fingerprint,
            'net_size_in_days': net_size_in_days,
        }
        feature_generator = FeatureGenerator(
            use_file=True,
            use_sim_cs=True,
            use_sim_j=True,
            use_social=True,
            use_temporal=True,
            use_pr_only=True,
            use_issue_only=True,
            similarity_config=similarity_config,
            temporal_config=temporal_config,
        )
        data = generate_training_data(feature_generator=feature_generator, mult_=multiplicity,
                                      net_size_in_days=net_size_in_days, training_repo_=repo, truth_=truth)

        with open(repo_loc[:-5] + '_preprocessed.json', 'w') as f:
            f.write(jsonpickle.encode(data))
