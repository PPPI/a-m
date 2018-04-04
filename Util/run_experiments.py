import itertools
import jsonpickle
import sys

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
    configs = itertools.product([False, True], repeat=7)
    for use_sim_cs, use_sim_j, use_file, use_social, use_temporal, use_pr_only, use_issue_only in configs:
        config = {
            'use_issue_only': use_issue_only,
            'use_pr_only': use_pr_only,
            'use_temporal': use_temporal,
            'use_sim_cs': use_sim_cs,
            'use_sim_j': use_sim_j,
            'use_file': use_file,
            'use_social': use_social
        }
        for project in projects:
            n_batches = 5 if project == 'PhilJay_MPAndroidChart' else 7
            project_dir = location_format % project
            with open(project_dir) as f:
                repo = jsonpickle.decode(f.read())

            with open(project_dir[:-len('.json')] + '_truth.json') as f:
                truth = jsonpickle.decode(f.read())

            batches = generate_batches(repo, n_batches)
            for i in range(n_batches - 2):
                linker = Linker(net_size_in_days=14, min_tok_len=2, undersample_multiplicity=100, stopwords=stopwords,
                                feature_config=config, predictions_between_updates=100)
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

                with open(project_dir[:-5] + ('_RAW_results_f%d_io%s_po%s_t%s_cs%s_j%s_f%s_s%s.txt' %
                                              (i, use_issue_only, use_pr_only, use_temporal, use_sim_cs,
                                               use_sim_j, use_file, use_social)), 'w') as f:
                    f.write(str(scores_dict))