import random
import sys
import jsonpickle

import numpy as np

if __name__ == '__main__':
    location = sys.argv[1]

    with open(location[:-5] + '_truth.json') as f:
        truth = jsonpickle.decode(f.read())
    with open(location) as f:
        repo = jsonpickle.decode(f.read())

    pr_nr = [pr.number for pr in repo.prs]
    issue_ids = [i.id_ for i in repo.issues]

    for keep_rate in np.arange(0.05, 0.95, 0.05):
        keep_rate = float('%2.3f' % keep_rate)
        noisy_truth = dict()
        for issue_id, pr_nrs in truth.items():
            for pr_nr in pr_nrs:
                roll = random.uniform(0.0, 1.0)
                target = keep_rate
                if roll > target:
                    try:
                        noisy_truth[issue_id].append(pr_nr)
                    except KeyError:
                        noisy_truth[issue_id] = [pr_nr]

        with open(location[:-5] + '_r_%2.2f_truth.json' % keep_rate, 'w') as f:
            f.write(jsonpickle.encode(noisy_truth))
