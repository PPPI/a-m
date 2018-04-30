import ast
import math

import jsonpickle
import numpy as np
import pandas as pd


def evaluate_at_threshold(result, th, top_k, truth):
    if top_k == 'inf':
        result_ = [(pr, sorted([(issue, prob) for issue, prob in pred if prob >= th],
                               reverse=True, key=lambda p: (p[1], p[0]))) for pr, pred in result.items()]
    else:
        result_ = [(pr, sorted([(issue, prob) for issue, prob in pred if prob >= th],
                               reverse=True, key=lambda p: (p[1], p[0]))[:top_k]) for pr, pred in result.items()]
    hit = 0
    ap = 0
    mrr = .0
    dcg = 0
    fpr = 0
    fnr = 0
    renorm = 0

    for pr_nr, predictions in result_:
        predictions = [p[0] for p in predictions]
        if 'null_issue' in predictions:
            predictions = predictions[:predictions.index('null_issue')]

        try:
            expected_truth = truth['#' + pr_nr[len('issue_'):]]
        except KeyError:
            expected_truth = set()

        predictions = [('#' + val[len('issue_'):] in expected_truth) for val in predictions]

        if (len(expected_truth) > 0) and (len(predictions) > 0):
            renorm += 1
            hit += any(predictions)
            curr_ap = np.mean([len([val for val in predictions[:(i + 1)] if val is True]) / (i + 1)
                               for i, j in enumerate(predictions) if j is True])
            if not math.isnan(curr_ap):
                ap += curr_ap
            try:
                mrr += 1 / (predictions.index(True) + 1)
            except ValueError:
                mrr += 0
            dcg += sum([1 / (math.log(i + 1, 2) if i > 1 else 1) if j is True else 0
                        for i, j in enumerate(predictions)])
        elif (len(expected_truth) == 0) and (len(predictions) > 0):
            # renorm += 1
            fpr += 1
        elif (len(expected_truth) > 0) and (len(predictions) == 0):
            fnr += 1
        elif (len(expected_truth) == 0) and (len(predictions) == 0):
            hit += 1
            # pass

    mrr = mrr / renorm if renorm > 0 else mrr
    ap = ap / renorm if renorm > 0 else ap
    dcg = dcg / renorm if renorm > 0 else dcg
    # hit = hit / renorm if renorm > 0 else hit
    hit /= len(result_)
    fpr /= len(result_)
    fnr /= len(result_)

    return hit, ap, mrr, dcg, fpr, fnr


if __name__ == '__main__':
    location_format = '../data/dev_set/%s.json'
    n_fold = 5
    projects = [
        # 'PhilJay_MPAndroidChart',
        'ReactiveX_RxJava',
        # 'google_guava',
        # 'facebook_react',
        # 'palantir_plottable',
        # 'tensorflow_tensorflow', # Dev set end
    ]
    for project in projects:
        results = list()
        for fold in [n_fold - 1]:
            with open((location_format[:-5] + '_results_f%d_selected_features_MF_p.txt') % (project, fold)) as f:
                result_str = f.read()
            result = ast.literal_eval(result_str)
            results.append((fold, result))
        results = dict(results)
        with open((location_format[:-5] + '_truth.json') % project) as f:
            truth = jsonpickle.decode(f.read())

        truth_pr = dict()
        for issue in truth.keys():
            for pr in truth[issue]:
                try:
                    truth_pr[pr].add(issue)
                except KeyError:
                    truth_pr[pr] = {issue}

        data = {'Threshold': list(), 'K': list(), 'Hit-rate': list(), 'Average Precision': list(),
                'Mean Reciprocal Rank': list(), 'Discounted Cumulative Gain': list(),
                'False Positive Rate': list(), 'False Negative Rate': list()}
        for result in results.values():
            for th in np.arange(.0, 1., step=.01):
                for top_k in [1, 3, 5, 7, 10, 15, 20, 'inf']:
                    hit, ap, mrr, dcg, fpr, fnr = evaluate_at_threshold(result, th, top_k, truth_pr)

                    data['Threshold'].append(float('%.5f' % th))
                    data['K'].append(top_k)
                    data['Hit-rate'].append(hit)
                    data['Average Precision'].append(ap)
                    data['Mean Reciprocal Rank'].append(mrr)
                    data['Discounted Cumulative Gain'].append(dcg)
                    data['False Positive Rate'].append(fpr)
                    data['False Negative Rate'].append(fnr)

        pd.DataFrame(data=data).to_csv((location_format[:-len('.json')] + '_results_interpreted_MF_p.csv') % project)
