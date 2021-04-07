import ast
import random
import webbrowser as wb

import numpy as np

from Util.sample_links import prompt_user_bool

if __name__ == '__main__':
    random.seed(23823)  # Randomly generated on google.com for the range 42-65553
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
        results_i = list()
        for fold in [n_fold - 1]:
            with open((location_format[:-5] + '_results_i_f%d_selected_features_MF_restricted_n.txt') % (project, fold)) as f:
                result_str = f.read()
            result = ast.literal_eval(result_str)
            results_i.append((fold, result))
        results_i = dict(results_i)

        results_p = list()
        for fold in [n_fold - 1]:
            with open((location_format[:-5] + '_results_p_f%d_selected_features_MF_restricted_n.txt') % (project, fold)) as f:
                result_str = f.read()
            result = ast.literal_eval(result_str)
            results_p.append((fold, result))
        results_p = dict(results_p)

        results = {**results_i, **results_p}

        results_sampled = random.sample(list(results.keys()), 100)

        for result in results_sampled:
            issue_id = int(result[1:])
            wb.open(url_format % (project.replace('_', '/'), issue_id))
            for other, probability in results[result]:
                if probability >= 0.5:
                    hits[project].append(prompt_user_bool(f"Is {other} a true link?"))

    for project in projects:
        print('The GT accuracy over the sample is %2.3f' % np.mean(hits[project]))
