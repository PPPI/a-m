import json
import random
import sys
import webbrowser as wb
from typing import Dict, Any, List, Tuple

import jsonpickle
import numpy as np


def prompt_user_bool(prompt_msg, default=None, options_map=None):
    options_map = options_map if options_map is not None else {'y': True, 'n': False}
    if default is not None and default not in options_map.keys():
        default = None

    prompt_msg += ' ['
    for option in options_map.keys():
        if option == default:
            prompt_msg += option.upper()
        else:
            prompt_msg += option.lower()
        prompt_msg += '/'
    prompt_msg = prompt_msg[:-1]
    prompt_msg += ']'

    while True:
        prompt = input(prompt_msg)
        if prompt == '' and default is not None:
            return options_map[default]

        for option in options_map.keys():
            if prompt.lower() == option:
                return options_map[option]

        print('%s not recognised as an option!' % prompt, file=sys.stderr)


def flatten_dict_of_lists(dict_of_lists: Dict[Any, List[Any]]) -> List[Tuple[Any, Any]]:
    return [(this, that) for this in dict_of_lists.keys() for that in dict_of_lists[this]]


if __name__ == '__main__':
    random.seed(46513)  # Randomly generated on google.com for the range 42-65553
    url_format = 'https://www.github.com/%s/issues/%d'
    location_format = '../data/dev_set/%s.json'
    n_fold = 5
    projects = [
        'PhilJay_MPAndroidChart',
        'ReactiveX_RxJava',
        'google_guava',
        'facebook_react',
        'palantir_plottable',  # Dev set end
    ]
    first_n = input('How many links should we sample?')
    hits = dict()
    for project in projects:
        hits[project] = list()
        with open((location_format[:-5] + '_truth.json') % project) as f:
            truth = flatten_dict_of_lists(jsonpickle.decode(f.read()))

        if len(truth) > 100:
            issues = random.sample(truth, 100)
        else:
            issues = truth

        if first_n == 'all':
            first_n_actual = len(issues)
        else:
            first_n_actual = min(int(first_n), len(issues))

        for issue, other in issues[:first_n_actual]:
            issue_id = int(issue[1:])
            wb.open(url_format % (project.replace('_', '/'), int(other[1:])))
            wb.open(url_format % (project.replace('_', '/'), issue_id))
            print(f"You are now considering {project}'s {issue_id}.")
            hits[project].append(prompt_user_bool(f"Is {other} a true link?"))

        print(f"The GT accuracy over the sample is {np.mean(hits[project]):2.3f} for project {project}.")

        with open(f"./gt_accuracy_{project}.json", 'w') as f:
            f.write(json.dumps(hits[project]))
