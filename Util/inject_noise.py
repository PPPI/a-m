import random
import sys
import jsonpickle
import itertools

if __name__ == '__main__':
    location = sys.argv[1]
    keep_rate = float(sys.argv[2])
    fake_rate = float(sys.argv[3])

    with open(location[:-5] + '_truth.json') as f:
        truth = jsonpickle.decode(f.read())
    with open(location) as f:
        repo = jsonpickle.decode(f.read())

    pr_nr = [pr.number for pr in repo.prs]
    issue_ids = [i.id_ for i in repo.issues]

    noisy_truth = dict()
    for pr_nr, issue_id in itertools.product(pr_nr, issue_ids):
        roll = random.uniform(0.0, 1.0)
        try:
            target = keep_rate if pr_nr in truth[issue_id] else fake_rate
        except KeyError:
            target = fake_rate
        if roll > target:
            try:
                noisy_truth[issue_id].append(pr_nr)
            except KeyError:
                noisy_truth[issue_id] = [pr_nr]

    with open(location[:-5] + '_n_truth.json', 'w') as f:
        f.write(jsonpickle.encode(noisy_truth))
