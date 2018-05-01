import ast
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    location_format = '../data/dev_set/%s.json'
    n_fold = 5
    projects = [
        'PhilJay_MPAndroidChart',
        'ReactiveX_RxJava',
        # 'google_guava',
        # 'facebook_react',
        'palantir_plottable',
        # 'tensorflow_tensorflow', # Dev set end
    ]
    for project in projects:
        results_unk = list()
        for fold in [n_fold - 1]:
            with open((location_format[:-5] + '_unk_rate_f%d_selected_features_MF_restricted_p.txt') % (project, fold)) as f:
                result_str = f.read()
            result = ast.literal_eval(result_str)
            results_unk.append((fold, result))
        results_unk = dict(results_unk)

        for fold in [n_fold - 1]:
            plt.figure()
            unks = results_unk[fold]
            unks = pd.Series(data=unks)
            unks_smooth = pd.ewma(unks, span=30)
            unks.plot(kind='line')
            unks_smooth.plot(kind='line')
            # unks_smooth.hist(cumulative=True, density=1, bins=len(unks_smooth))
            plt.savefig((location_format[:-5] + '.png') % project)
