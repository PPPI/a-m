import os
import sys
from Prediction.Linker import Linker

import jsonpickle

if __name__ == '__main__':
    filename = sys.argv[1]
    with open(filename) as f:
        repo = jsonpickle.decode(f.read())
    with open(filename[:-len('.json')] + '_truth.json') as f:
        truth = jsonpickle.decode(f.read())

    print('Loaded repository %s' % repo.name)
    Linker = Linker(net_size_in_days=14, min_tok_len=2, undersample_multiplicity=100)
    Linker.fit(repo, truth)
    print('Trained Random Forest classifier')
    out_path = os.path.join(os.getcwd(), 'models', repo.name[1:].translate({ord(c): '_' for c in '\\/'}))
    os.makedirs(out_path, exist_ok=True)
    Linker.persist_to_disk(out_path)
    print('Recorded model to disk')
