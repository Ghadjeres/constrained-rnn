import os

import pandas as pd
import seaborn as sns
from constraints.data_utils import PACKAGE_DIR
from constraints.divergences import get_group_div, all_divs
from matplotlib import pyplot as plt

ordered_notes = ['__',
                 'B#3', 'C4', 'C#4', 'D-4', 'C##4', 'D4', 'E--4', 'D#4', 'E-4',
                 'D##4', 'E4', 'F-4', 'E#4', 'F4', 'F#4', 'G-4', 'F##4', 'G4',
                 'A--4',
                 'G#4', 'A-4', 'G##4', 'A4', 'B--4', 'A#4', 'B-4', 'B4', 'C-5',
                 'B#4', 'C5', 'C#5', 'D-5', 'C##5', 'D5', 'E--5', 'D#5', 'E-5',
                 'D##5', 'E5', 'F-5', 'E#5', 'F5', 'F#5', 'G-5', 'F##5', 'G5',
                 'A--5',
                 'G#5', 'A-5', 'A5', 'B--5',
                 'rest',
                 'START', 'END'
                 ]
ordered_notes.reverse()

# WARNING HARD CODED
index2note = {0: 'G#4', 1: 'E-4', 2: 'F4', 3: '__', 4: 'G-5', 5: 'C4',
              6: 'A#4', 7: 'D#4', 8: 'C#4', 9: 'D#5', 10: 'E5', 11: 'G5',
              12: 'E-5', 13: 'END', 14: 'B-4', 15: 'E#4', 16: 'D-5', 17: 'F#4',
              18: 'E#5', 19: 'START', 20: 'A4', 21: 'G4', 22: 'E4', 23: 'A5',
              24: 'F#5', 25: 'A-4', 26: 'B4', 27: 'A-5', 28: 'G-4', 29: 'D4',
              30: 'C#5', 31: 'C5', 32: 'D5', 33: 'rest', 34: 'F5', 35: 'C-5',
              36: 'F-5', 37: 'B#4', 38: 'G#5', 39: 'F-4', 40: 'D-4',
              41: 'B--4', 42: 'B--5', 43: 'E--5', 44: 'B#3', 45: 'F##4',
              46: 'C##4', 47: 'G##4', 48: 'C##5', 49: 'D##4', 50: 'F##5',
              51: 'D##5', 52: 'A--5', 53: 'E--4', 54: 'A--4'}


def plot_generation_results_directory():
    results_dir = os.path.join(PACKAGE_DIR,
                               'results',
                               'generations')
    experiments_dir = os.listdir(results_dir)
    for experiment_dir in experiments_dir:
        experiment_dir_abs = os.path.join(results_dir,
                                          experiment_dir)
        if not os.path.isdir(experiment_dir_abs):
            continue

        csv_files = filter(lambda s: s.endswith('.csv'),
                           os.listdir(experiment_dir_abs))
        for csv in csv_files:
            csv_filename = os.path.join(experiment_dir_abs,
                                        csv)

            base_filename = os.path.splitext(csv)[0]
            fig_filepath = os.path.join(experiment_dir_abs,
                                        f'{base_filename}.svg'
                                        )
            plot_csv(csv_filename,
                     fig_filepath=fig_filepath)

            fig_filepath = os.path.join(experiment_dir_abs,
                                        f'dists.svg'
                                        )
            plot_dists(csv_filename,
                       fig_filepath=fig_filepath)


def plot_ratios_csv(csv_filename, fig_filepath):
    df = pd.read_csv(csv_filename,
                     header=0)
    # pal = sns.cubehelix_palette(1 + max(df[
    #                                         'num_enforced_constraints']),
    #                             as_cmap=True)
    pal = sns.dark_palette('seagreen',
                           as_cmap=True,
                           reverse=True)
    fig, ax = plt.subplots()

    xymin = min(min(df['no_constraint']), min(df['constraint'])) - 2
    xymax = max(max(df['no_constraint']), max(df['constraint'])) + 2
    plt.axis([xymin, xymax, xymin, xymax])

    sns.regplot(x='no_constraint',
                y='no_constraint',
                data=df,
                fit_reg=True,
                marker='',
                ax=ax,
                )

    sns.regplot(x='no_constraint',
                y='constraint',
                data=df,
                marker='',
                fit_reg=True,
                ax=ax
                )

    ax = ax.scatter(x=df['no_constraint'],
                    y=df['constraint'],
                    c=df['num_enforced_constraints'],
                    s=4,
                    cmap=pal)
    sns.plt.savefig(os.path.join(fig_filepath),
                    format='svg')
    plt.clf()


def plot_ratios_stats_directory():
    results_dir = os.path.join(PACKAGE_DIR,
                               'results',
                               'ratios')

    csv_files = filter(lambda s: s.endswith('.csv'),
                       os.listdir(results_dir))
    for csv in csv_files:
        csv_filename = os.path.join(results_dir,
                                    csv)
        base_filename = os.path.splitext(csv)[0]
        fig_filepath = os.path.join(results_dir,
                                    f'{base_filename}.svg'
                                    )
        if os.path.exists(fig_filepath):
            continue
        plot_ratios_csv(csv_filename,
                        fig_filepath=fig_filepath)


def plot_csv(csv_filename, fig_filepath):
    print(f'Plotting {fig_filepath}')
    df = pd.read_csv(csv_filename,
                     header=None,
                     names=['note_index', 'time_index', 'value',
                            'model_index'])
    df['note'] = (df['note_index'].apply((lambda x: index2note[x])))
    df['time_index'] = (df['time_index'] - 15) / 4

    matrices = []
    basename, ext = os.path.splitext(fig_filepath)
    for model_index, model_name in enumerate(['constrained', 'unconstrained']):
        full_fig_filepath = basename + f'_{model_name}' + ext

        dfx = df[df['model_index'] == model_index]
        matrix = dfx.pivot(index='note', columns='time_index', values='value')
        matrix = matrix.reindex(ordered_notes)
        matrices.append(matrix)

        if os.path.exists(full_fig_filepath):
            continue

        # figure = plt.gcf()
        # ax1 = figure.add_axes([0., 0., 1.2, 1.0])
        sns.set(font_scale=0.6)
        # ax = sns.heatmap(matrix, vmin=0, vmax=0.4, ax=ax1, xticklabels=4)

        ax = sns.heatmap(matrix, vmin=0, vmax=1.,  # xticklabels=4,
                         cmap="YlGnBu"
                         )
        plt.yticks(rotation=15)
        num_xticks = matrix.shape[1]
        tick_locations = [4 * i for i in range(num_xticks // 4)]
        tick_values = [i + 1 for i in range(num_xticks // 4)]
        plt.xticks(tick_locations, tick_values)

        sns.plt.savefig(os.path.join(full_fig_filepath),
                        format='svg')
        plt.clf()

    # plot difference
    full_fig_filepath = basename + f'_diff' + ext
    if not os.path.exists(full_fig_filepath):
        diff_matrix = matrices[0] - matrices[1]
        sns.set(font_scale=0.6)
        ax = sns.heatmap(diff_matrix, vmin=-1, center=0., vmax=1.,  # xticklabels=4,
                         cmap="RdBu_r"
                         )
        plt.yticks(rotation=15)
        num_xticks = diff_matrix.shape[1]
        tick_locations = [4 * i for i in range(num_xticks // 4)]
        tick_values = [i + 1 for i in range(num_xticks // 4)]
        plt.xticks(tick_locations, tick_values)

        sns.plt.savefig(os.path.join(full_fig_filepath),
                        format='svg')
        plt.clf()


def plot_dists(csv_filename, fig_filepath):
    basename, ext = os.path.splitext(fig_filepath)
    df = pd.read_csv(csv_filename,
                     header=None,
                     names=['note_index', 'time_index', 'value',
                            'model_index'])
    df['note'] = (df['note_index'].apply((lambda x: index2note[x])))
    df['time_index'] = (df['time_index'] - 15) / 4

    grouped = df.groupby('time_index')

    for sqrt in [True, False]:
        kl_series = [
            grouped.apply(get_group_div(div, sqrt=sqrt)).rename(div.__name__)
            for div in all_divs
        ]
        kl_series = pd.concat(kl_series, axis=1)

        # save plots
        filepath = basename + f'{"_sqrt" if sqrt else ""}' + ext
        if not os.path.exists(filepath):
            kl_series.plot(subplots=True, layout=(4, 1))
            plt.savefig(filepath, format='svg')

        filepath = basename + f'_grouped' + f'{"_sqrt" if sqrt else ""}' + ext
        if not os.path.exists(filepath):
            kl_series.plot()
            plt.savefig(filepath, format='svg')
        plt.close('all')


if __name__ == '__main__':
    plot_generation_results_directory()
    # plot_ratios_stats_directory()
