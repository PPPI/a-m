import os
import sys
from Prediction.Linker import Linker

import jsonpickle

from Util import utils_
from Util.ReservedKeywords import java_reserved, c_reserved, cpp_reserved, javascript_reserved, python_reserved


def main(args):
    stopwords = utils_.GitMineUtils.STOPWORDS \
                + list(set(java_reserved + c_reserved + cpp_reserved + javascript_reserved + python_reserved))
    if len(args) < 1:
        raise ValueError('Please provide the path to the repository object as a cmd argument. Make sure the truth '
                         'object is in the same folder.')
    filename = args[0]
    with open(filename) as f:
        repo = jsonpickle.decode(f.read())
    with open(filename[:-len('.json')] + '_truth.json') as f:
        truth = jsonpickle.decode(f.read())
    repo.name = '/' + repo.name if '/' != repo.name[0] else repo.name

    config = {
        'use_issue_only': True,
        'use_pr_only': True,
        'use_temporal': True,
        'use_sim_cs': False,
        'use_sim_j': False,
        'use_sim_d': True,
        'use_file': True,
        'use_social': True
    }
    # features = ['dice', 'report_size', 'branch_size', 'files_touched_by_pr', 'developer_normalised_lag']
    features = None
    print('Loaded repository %s' % repo.name)
    linker = Linker(net_size_in_days=31, min_tok_len=2, undersample_multiplicity=100, feature_config=config,
                    predictions_between_updates=1000, stopwords=stopwords)
    linker.fit(repo, truth, features=features)
    print('Trained Random Forest classifier')
    out_path = os.path.join(os.getcwd(), 'models', repo.name[1:].translate({ord(c): '_' for c in '\\/'}))
    os.makedirs(out_path, exist_ok=True)
    linker.persist_to_disk(out_path)
    print('Recorded model to disk')


if __name__ == '__main__':
    main(sys.argv[1:])
