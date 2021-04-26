import ast
import json
import random
import webbrowser as wb

import numpy as np

from Util.sample_links import prompt_user_bool

if __name__ == '__main__':
    random.seed(23823)  # Randomly generated on google.com for the range 42-65553
    url_format = 'https://www.github.com/%s/issues/%d'
    location_format = '../data/dev_set/%s.json'
    n_fold = 10
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
        for fold in range(n_fold):
            with open((location_format[
                       :-5] + '_RAW_i_results_f%d_ioTrue_poTrue_tTrue_csTrue_jFalse_fFalse_sFalse_u1_r0.05.txt') % (
                              project, fold)) as f:
                result_str = f.read()
            result = ast.literal_eval(result_str)
            results_i.append((fold, result))
        results_i = dict(results_i)

        results_p = list()
        for fold in range(n_fold):
            with open((location_format[
                       :-5] + '_RAW_p_results_f%d_ioTrue_poTrue_tTrue_csTrue_jFalse_fFalse_sFalse_u1_r0.05.txt') % (
                              project, fold)) as f:
                result_str = f.read()
            result = ast.literal_eval(result_str)
            results_p.append((fold, result))
        results_p = dict(results_p)

        results = dict()
        for fold in range(n_fold):
            results[fold] = {**results_i[fold], **results_p[fold]}
            results[fold] = {k: [ref for ref, prop in v if prop >= 0.5] for k, v in results[fold].items()}
            results[fold] = [(k, other_k) for k, v in results[fold].items() if len(v) > 0 for other_k in v]

        results = [inner for outer in results.values() for inner in outer]

        if len(results) > 100:
            results_sampled = random.sample(results, 100)
        else:
            results_sampled = results

        for result, other in results_sampled:
            issue_id = int(result[6:])
            wb.open(url_format % (project.replace('_', '/'), issue_id))
            wb.open(url_format % (project.replace('_', '/'), other[6:]))
            hits[project].append(prompt_user_bool(f"Is {other[6:]} a true link?"))

    for project in projects:
        print('The GT accuracy over the sample is %2.3f' % np.mean(hits[project]))

    with open('./gt_accuracy.json', 'w') as f:
        f.write(json.dumps(hits))
