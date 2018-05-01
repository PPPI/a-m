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
    features = ['cosine_tc', 'report_size', 'branch_size', 'files_touched_by_pr', 'developer_normalised_lag']
    # configs = itertools.product([True, False], repeat=8)
    configs = [[True]*8]
    for use_sim_cs, use_sim_j, use_sim_d, use_file, use_social, use_temporal, use_pr_only, use_issue_only in configs:
        if not (use_sim_cs or use_sim_j or use_file or use_social or use_temporal or use_pr_only or use_issue_only):
            continue
        config = {
            'use_issue_only': use_issue_only,
            'use_pr_only': use_pr_only,
            'use_temporal': use_temporal,
            'use_sim_cs': use_sim_cs,
            'use_sim_j': use_sim_j,
            'use_sim_d': use_sim_d,
            'use_file': use_file,
            'use_social': use_social
        }

        for project in projects:
            n_batches = 5
            project_dir = location_format % project
            with open(project_dir) as f:
                repo = jsonpickle.decode(f.read())

            with open(project_dir[:-len('.json')] + '_truth.json') as f:
                truth = jsonpickle.decode(f.read())

            batches = generate_batches(repo, n_batches)
            for under in [1, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:
                for i in [n_batches - 1]:
                    linker = Linker(net_size_in_days=7, min_tok_len=2, undersample_multiplicity=under, stopwords=stopwords,
                                    feature_config=config, predictions_between_updates=1000)
                    training = list()
                    for j in range(n_batches - 1):
                        training += batches[j]
                    linker.fit(inflate_events(training, repo.langs, repo.name), truth, features=features)
                    scores, unk_rate = linker.validate_over_suffix(batches[i])
                    pr_numbers = [pr.number for pr in repo.prs]
                    issue_ids = [issue.id_ for issue in repo.issues]
                    scores_p = dict([(id_, pred) for id_, pred in scores if id_ in pr_numbers])
                    scores_i = dict([(id_, pred) for id_, pred in scores if id_ in issue_ids])

                    with open(project_dir[:-5] + ('_RAW_p_results_f%d_io%s_po%s_t%s_cs%s_j%s_f%s_s%s_u%d.txt' %
                                                  (i, use_issue_only, use_pr_only, use_temporal, use_sim_cs,
                                                   use_sim_j, use_file, use_social, under)), 'w') as f:
                        f.write(str(scores_p))

                    with open(project_dir[:-5] + ('_RAW_i_results_f%d_io%s_po%s_t%s_cs%s_j%s_f%s_s%s_u%d.txt' %
                                                  (i, use_issue_only, use_pr_only, use_temporal, use_sim_cs,
                                                   use_sim_j, use_file, use_social, under)), 'w') as f:
                        f.write(str(scores_i))

                    with open(project_dir[:-5] + ('_RAW_unk_rate_f%d_io%s_po%s_t%s_cs%s_j%s_f%s_s%s_u%d.txt' %
                                                  (i, use_issue_only, use_pr_only, use_temporal, use_sim_cs,
                                                   use_sim_j, use_file, use_social, under)), 'w') as f:
                        f.write(str(unk_rate))