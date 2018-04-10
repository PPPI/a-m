import itertools
import jsonpickle
import sys
import numpy as np
import pandas as pd

from Prediction.Linker import Linker
from Prediction.training_utils import generate_batches, inflate_events
from Util import utils_
from Util.ReservedKeywords import java_reserved, c_reserved, cpp_reserved, javascript_reserved, python_reserved

if __name__ == '__main__':
    location_format = sys.argv[1]

    with open(sys.argv[2]) as f:
        projects = [line.strip() for line in f.readlines()]

    stopwords = utils_.GitMineUtils.STOPWORDS \
                + list(set(java_reserved + c_reserved + cpp_reserved + javascript_reserved + python_reserved))
    configs = itertools.product([True, False], repeat=7)
    for use_sim_cs, use_sim_j, use_file, use_social, use_temporal, use_pr_only, use_issue_only in configs:
        if not (use_sim_cs or use_sim_j or use_file or use_social or use_temporal or use_pr_only or use_issue_only):
            continue
        config = {
            'use_issue_only': use_issue_only,
            'use_pr_only': use_pr_only,
            'use_temporal': use_temporal,
            'use_sim_cs': use_sim_cs,
            'use_sim_j': use_sim_j,
            'use_file': use_file,
            'use_social': use_social
        }
        features_string = \
            ('cosine cosine_tt cosine_tc cosine_ct cosine_cc ' if config['use_sim_cs'] else '') + \
            ('jaccard jaccard_tt jaccard_tc jaccard_ct jaccard_cc ' if config['use_sim_j'] else '') + \
            ('files_shared ' if config['use_file'] else '') + \
            ('is_reporter is_assignee engagement in_top_2 in_comments ' if config['use_social'] else '') + \
            (
                'developer_normalised_lag lag_from_issue_open_to_pr_submission '
                'lag_from_last_issue_update_to_pr_submission '
                if config['use_temporal'] else '') + \
            ('no_pr_desc branch_size files_touched_by_pr ' if config['use_pr_only'] else '') + \
            ('report_size participants bounces existing_links ' if config['use_issue_only'] else '')
        features_string = features_string.strip().split(' ')
        index_feature_map = {i: features_string[i] for i in range(len(features_string))}
        for project in projects:
            n_batches = 5
            project_dir = location_format % project
            with open(project_dir) as f:
                repo = jsonpickle.decode(f.read())

            with open(project_dir[:-len('.json')] + '_truth.json') as f:
                truth = jsonpickle.decode(f.read())

            batches = generate_batches(repo, n_batches)
            for i in [n_batches - 1]:
                linker = Linker(net_size_in_days=14, min_tok_len=2, undersample_multiplicity=1000, stopwords=stopwords,
                                feature_config=config, predictions_between_updates=1000)
                training = list()
                for j in range(n_batches - 1):
                    training += batches[j]
                linker.fit(inflate_events(training, repo.langs, repo.name), truth)
                forest = linker.clf
                importances = forest.feature_importances_
                std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
                pd.DataFrame(data={'Feature': features_string, 'Importance': importances, 'STD': std}) \
                    .to_csv(project_dir[:-5] + ('_IMP_results_f%d_io%s_po%s_t%s_cs%s_j%s_f%s_s%s.csv' %
                                                (i, use_issue_only, use_pr_only, use_temporal, use_sim_cs,
                                                 use_sim_j, use_file, use_social)))
                scores = linker.validate_over_suffix(batches[i])
                scores_dict = dict()
                for pr_id, predictions in scores:
                    try:
                        scores_dict[pr_id].union(set(predictions))
                    except KeyError:
                        scores_dict[pr_id] = set(predictions)
                for pr_id in scores_dict.keys():
                    scores_dict[pr_id] = list(scores_dict[pr_id])
                    scores_dict[pr_id] = sorted(scores_dict[pr_id], reverse=True, key=lambda p: (p[1], p[0]))

                with open(project_dir[:-5] + ('_RAW_results_f%d_io%s_po%s_t%s_cs%s_j%s_f%s_s%s.txt' %
                                              (i, use_issue_only, use_pr_only, use_temporal, use_sim_cs,
                                               use_sim_j, use_file, use_social)), 'w') as f:
                    f.write(str(scores_dict))
