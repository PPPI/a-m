import os
import sys
from Prediction.Linker import Linker

import jsonpickle

from Util import utils_
from Util.ReservedKeywords import java_reserved, c_reserved, cpp_reserved, javascript_reserved, python_reserved

if __name__ == '__main__':
    stopwords = utils_.GitMineUtils.STOPWORDS \
                + list(set(java_reserved + c_reserved + cpp_reserved + javascript_reserved + python_reserved))
    filename = sys.argv[1]
    with open(filename) as f:
        repo = jsonpickle.decode(f.read())
    with open(filename[:-len('.json')] + '_truth.json') as f:
        truth = jsonpickle.decode(f.read())

    config = {
        'use_issue_only': False,
        'use_pr_only': True,
        'use_temporal': True,
        'use_sim_cs': False,
        'use_sim_j': True,
        'use_file': True,
        'use_social': True
    }
    print('Loaded repository %s' % repo.name)
    Linker = Linker(net_size_in_days=14, min_tok_len=2, undersample_multiplicity=100, feature_config=config,
                    predictions_between_updates=100, stopwords=stopwords)
    Linker.fit(repo, truth)
    print('Trained Random Forest classifier')
    out_path = os.path.join(os.getcwd(), 'models', repo.name[1:].translate({ord(c): '_' for c in '\\/'}))
    os.makedirs(out_path, exist_ok=True)
    Linker.persist_to_disk(out_path)
    print('Recorded model to disk')
