import ast
import math
import os
import sys

import jsonpickle
import numpy as np
import pandas as pd


def evaluate_at_threshold(result_p, result_i, th, top_k, repo):
    if top_k == 'inf':
        result_p_ = [(pr, sorted([(issue, prob) for issue, prob in pred if prob >= th],
                                 reverse=True, key=lambda p: (p[1], p[0]))) for pr, pred in result_p.items()]
    else:
        result_p_ = [(pr, sorted([(issue, prob) for issue, prob in pred if prob >= th],
                                 reverse=True, key=lambda p: (p[1], p[0]))[:top_k]) for pr, pred in result_p.items()]
    if top_k == 'inf':
        result_i_ = [(issue, sorted([(pr, prob) for pr, prob in pred if prob >= th],
                                    reverse=True, key=lambda p: (p[1], p[0]))) for issue, pred in result_i.items()]
    else:
        result_i_ = [(issue, sorted([(pr, prob) for pr, prob in pred if prob >= th],
                                    reverse=True, key=lambda p: (p[1], p[0]))[:top_k]) for issue, pred in
                     result_i.items()]
    hit = 0
    ap = 0
    mrr = .0
    dcg = 0
    fpr = 0
    fnr = 0
    renorm = 0
    neg_renorm = 0

    for pr_nr, predictions in result_p_:
        predictions = [p[0] for p in predictions]
        if 'null_issue' in predictions:
            predictions = predictions[:predictions.index('null_issue')]

        pr_sha = [{c.c_hash for c in p.commits} for p in repo.prs if p.number == pr_nr][0]

        expected_truth = [True for i in repo.issues for issue_side in i.commits
                          if any([pr_side.startswith(issue_side) for pr_side in pr_sha])]

        predictions = [any([pr_side.startswith(issue_side) for pr_side in pr_sha
                            for issue_side in [i.commits for i in repo.issues if i.id_ == val][0]])
                       for val in predictions]

        if (len(expected_truth) > 0) and (len(predictions) > 0):
            renorm += 1
            neg_renorm += 1
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
            renorm += 1
            fpr += 1
        elif (len(expected_truth) > 0) and (len(predictions) == 0):
            neg_renorm += 1
            fnr += 1
        elif (len(expected_truth) == 0) and (len(predictions) == 0):
            hit += 1
            # pass
    for issue, predictions in result_i_:
        predictions = [p[0] for p in predictions]
        if 'null_pr' in predictions:
            predictions = predictions[:predictions.index('null_pr')]

        issue_sha = [i.commits for i in repo.issues if i.id_ == issue][0]

        expected_truth = [True for p in repo.prs for pr_side in {c.c_hash for c in p.commits}
                          if any([pr_side.startswith(issue_side) for issue_side in issue_sha])]

        predictions = [any([pr_side.startswith(issue_side) for issue_side in issue_sha
                            for pr_side in [{c.c_hash for c in p.commits} for p in repo.prs if p.number == val][0]])
                       for val in predictions]

        if (len(expected_truth) > 0) and (len(predictions) > 0):
            renorm += 1
            neg_renorm += 1
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
            renorm += 1
            fpr += 1
        elif (len(expected_truth) > 0) and (len(predictions) == 0):
            neg_renorm += 1
            fnr += 1
        elif (len(expected_truth) == 0) and (len(predictions) == 0):
            hit += 1
            # pass

    mrr = mrr / renorm if renorm > 0 else mrr
    ap = ap / renorm if renorm > 0 else ap
    dcg = dcg / renorm if renorm > 0 else dcg
    fpr = fpr / renorm if renorm > 0 else fpr
    fnr = fnr / neg_renorm if neg_renorm > 0 else fnr
    # hit = hit / renorm if renorm > 0 else hit
    hit /= (len(result_i_) + len(result_p_))

    return hit, ap, mrr, dcg, fpr, fnr


if __name__ == '__main__':
    location_format = sys.argv[1]

    with open(sys.argv[2]) as f:
        projects = [line.strip() for line in f.readlines()]

    for project in projects:
        try:
            results_i = list()
            for fold in [4]:
                with open((location_format[
                           :-5] + '_RAW_i_results_f4_ioTrue_poTrue_tTrue_csTrue_jFalse_fFalse_sFalse_u1.txt') % project) as f:
                    result_str = f.read()
                result = ast.literal_eval(result_str)
                results_i.append((fold, result))
            results_i = dict(results_i)

            results_p = list()
            for fold in [4]:
                with open((location_format[
                           :-5] + '_RAW_p_results_f4_ioTrue_poTrue_tTrue_csTrue_jFalse_fFalse_sFalse_u1.txt') % project) as f:
                    result_str = f.read()
                result = ast.literal_eval(result_str)
                results_p.append((fold, result))
            results_p = dict(results_p)

            with open(location_format % project) as f:
                repo = jsonpickle.decode(f.read())

            data = {'Threshold': list(), 'K': list(), 'Hit-rate': list(), 'Average Precision': list(),
                    'Mean Reciprocal Rank': list(), 'Discounted Cumulative Gain': list(),
                    'False Positive Rate': list(), 'False Negative Rate': list()}
            for fold in [4]:
                result_p = results_p[fold]
                result_i = results_i[fold]
                for th in np.arange(.0, 1., step=.01):
                    for top_k in [1, 3, 5, 7, 10, 15, 20, 'inf']:
                        hit, ap, mrr, dcg, fpr, fnr = evaluate_at_threshold(result_p, result_i, th, top_k, repo)

                        data['Threshold'].append(float('%.5f' % th))
                        data['K'].append(top_k)
                        data['Hit-rate'].append(hit)
                        data['Average Precision'].append(ap)
                        data['Mean Reciprocal Rank'].append(mrr)
                        data['Discounted Cumulative Gain'].append(dcg)
                        data['False Positive Rate'].append(fpr)
                        data['False Negative Rate'].append(fnr)

            pd.DataFrame(data=data).to_csv(
                (location_format[:-len('.json')] + '_results_interpreted_commits_MF_restricted.csv') % project)
        except Exception as e:
            print('%s' % project)
            print(str(e))
