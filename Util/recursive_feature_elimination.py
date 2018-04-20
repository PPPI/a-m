import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

if __name__ == '__main__':
    jsonpickle_numpy.register_handlers()
    feature_names = 'cosine cosine_tt cosine_tc cosine_ct cosine_cc jaccard jaccard_tt jaccard_tc jaccard_ct ' \
                    'jaccard_cc dice dice_tt dice_tc dice_ct dice_cc files_shared is_reporter ' \
                    'is_assignee engagement in_top_2 in_comments ' \
                    'developer_normalised_lag lag_from_issue_open_to_pr_submission ' \
                    'lag_from_last_issue_update_to_pr_submission no_pr_desc branch_size files_touched_by_pr ' \
                    'report_size participants bounces existing_links'.split(' ')
    X = list()
    y = list()
    data_locations = ['../data/dev_set/%s_preprocessed.json' % s for s in [
        'PhilJay_MPAndroidChart',
        'ReactiveX_RxJava',
        'palantir_plottable',
        # 'tensorflow_tensorflow',
    ]]

    for data_loc in data_locations:
        with open(data_loc) as f:
            data = jsonpickle.decode(f.read())
        for point in data:
            X.append(tuple([v for k, v in point.items() if k not in ['linked', 'issue', 'pr']]))
            y.append(1 if point['linked'] else -1)

    estimator = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample')
    selector = RFE(estimator, 5, step=1)
    selector = selector.fit(X, y)
    print(list(zip(feature_names, selector.ranking_)))
