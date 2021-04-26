import json
import random
import sys
import webbrowser as wb

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
        'palantir_plottable',
        'tensorflow_tensorflow',  # Dev set end
    ]
    hits = dict()
    for project in projects:
        hits[project] = list()
        with open((location_format[:-5] + '_truth.json') % project) as f:
            truth = jsonpickle.decode(f.read())
        issues = random.sample(list(truth.keys()), 100)

        for issue in issues:
            issue_id = int(issue[1:])
            wb.open(url_format % (project.replace('_', '/'), issue_id))
            for other in truth[issue]:
                wb.open(url_format % (project.replace('_', '/'), other[1:]))
                hits[project].append(prompt_user_bool(f"Is {other} a true link?"))

    for project in projects:
        print('The GT accuracy over the sample is %2.3f' % np.mean(hits[project]))

    with open('./gt_accuracy.json', 'w') as f:
        f.write(json.dumps(hits))
