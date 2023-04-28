from itertools import groupby
from typing import Tuple, List
from xmlrpc.client import boolean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

ROOT_PATH = '/Users/hyewonhan/thisishywn/OSAS-severity-prediction'

LABEL_DICT = {
    'sex': 'Sex',
    'age': 'Age',
    'ESS1': 'ESS1',
    'ESS2': 'ESS2',
    'ESS3': 'ESS3',
    'ESS4': 'ESS4',
    'ESS5': 'ESS5',
    'ESS6': 'ESS6',
    'ESS7': 'ESS7',
    'ESS8': 'ESS8',
    'ISI1a': 'ISI1a',
    'ISI1b': 'ISI1b',
    'ISI1c': 'ISI1c',
    'ISI2': 'ISI2',
    'ISI3': 'ISI3',
    'ISI4': 'ISI4',
    'ISI5': 'ISI5',
    'height': 'Height',
    'weight': 'Weight',
    'BMI': 'BMI',
    'PSQI': 'PSQI',
    'ISI': 'ISI total score',
    'SSS': 'SSS total score',
    'ESS': 'ESS total score',
    'K-BDI2': 'K-BDI-II total score',
    'abdominal_circumference': 'Abdominal circumference',
    'hip_circumference': 'Hip Circumference',
    'neck_circumference_sit': 'Neck circumference (siting position)',
    'neck_circumference_lie_down': 'Neck circumference (lying position)',
    'head_circumference': 'Head circumference',
    'last_sleep_time': 'Hours of sleep',
    'took_sleeping_pill_yn': 'Consumption of hypnotics',
    'AHI': 'AHI',
    'AHI_group': 'OSAS severity',
}


class ClassificationResult:
    def __init__(self, num_divide: int) -> None:
        self.ml_algorithms = ['LGBMClassifier', 'XGBClassifier', 'RandomForestClassifier', 'CatBoostClassifier']
        self.ahi_points = ['p5', 'p15', 'p30']
        self.metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'auc']

        self.df_dict_default = {}
        for p in self.ahi_points:
            self.df_dict_default[p] = pd.read_csv(f'{ROOT_PATH}/classification_results/{p}_best_result.csv')

        self.df_dict_hp = {}
        for p in self.ahi_points:
            for i in range(0, num_divide):
                if i == 0:
                    if p == 'p30':
                        self.df_dict_hp[p] = pd.concat([pd.read_csv(f'classification_results/{p}/{p}_hpo_best_result_{i}a.csv'), pd.read_csv(f'classification_results/{p}/{p}_hpo_best_result_{i}b.csv')])
                    else:
                        self.df_dict_hp[p] = pd.read_csv(f'classification_results/{p}/{p}_hpo_best_result_{i}.csv')
                else:
                    temp_df = pd.read_csv(f'{ROOT_PATH}/classification_results/{p}/{p}_hpo_best_result_{i}.csv')
                    temp_df['divide_num'] = i
                    self.df_dict_hp[p] = pd.concat([self.df_dict_hp[p], temp_df])
        
        self.fpr_tpr_dict = {}

    def report_ml(self) -> pd.DataFrame:
        best_dict = {}
        for p in self.ahi_points:
            for i, metric in enumerate(self.metrics):
                for algo in self.ml_algorithms:
                    if i == 0:
                        best_dict[p] = pd.concat([
                            self.df_dict_hp[p].loc[(self.df_dict_hp[p]['cluster'] == 0) & (self.df_dict_hp[p]['classification_algorithm'] == algo) & (self.df_dict_hp[p]['feature_engineering'] == 'None')].sort_values(by=metric, ascending=False).head(1).copy(),
                            self.df_dict_hp[p].loc[(self.df_dict_hp[p]['cluster'] == 1) & (self.df_dict_hp[p]['classification_algorithm'] == algo) & (self.df_dict_hp[p]['feature_engineering'] == 'None')].sort_values(by=metric, ascending=False).head(1).copy(),
                        ])
                        best_dict[p]['best_result_metric'] = metric
                        best_dict[p]['AHI_cut_off'] = p
                    else:
                        cl_0 = self.df_dict_hp[p].loc[(self.df_dict_hp[p]['cluster'] == 0) & (self.df_dict_hp[p]['classification_algorithm'] == algo) & (self.df_dict_hp[p]['feature_engineering'] == 'None')].sort_values(by=metric, ascending=False).head(1).copy()
                        cl_1 = self.df_dict_hp[p].loc[(self.df_dict_hp[p]['cluster'] == 1) & (self.df_dict_hp[p]['classification_algorithm'] == algo) & (self.df_dict_hp[p]['feature_engineering'] == 'None')].sort_values(by=metric, ascending=False).head(1).copy()
                        
                        cl_0['best_result_metric'] = metric
                        cl_1['best_result_metric'] = metric
                        
                        cl_0['AHI_cut_off'] = p
                        cl_1['AHI_cut_off'] = p
                        
                        best_dict[p] = pd.concat([
                            best_dict[p],
                            cl_0,
                            cl_1,
                        ])

        best_df = pd.concat([best_dict[p] for p in self.ahi_points])
        metric_result_df = self._metric_results(best_df, ml_algorithm=True)

        metric_result_df = metric_result_df.droplevel('best_result_metric')
        self.metric_result_df_hp = metric_result_df

        final_metric_dict = {
            'classification_algorithm': [],
            'AHI_cut_off': [],
            'accuracy_mean': [],
            'accuracy_std': [],
            'accuracy_min': [],
            'accuracy_max': [],
            'accuracy_list': [],
            'f1_score_mean': [],
            'f1_score_std': [],
            'f1_score_min': [],
            'f1_score_max': [],
            'f1_score_list': [],
            'precision_mean': [],
            'precision_std': [],
            'precision_min': [],
            'precision_max': [],
            'precision_list': [],
            'recall_mean': [],
            'recall_std': [],
            'recall_min': [],
            'recall_max': [],
            'recall_list': [],
            'auc_mean': [],
            'auc_std': [],
            'auc_min': [],
            'auc_max': [],
            'auc_list': [],
        }

        for p in self.ahi_points:
            for algo in self.ml_algorithms:
                final_metric_df = metric_result_df.iloc[(metric_result_df.index.get_level_values('AHI_cut_off') == p) & (metric_result_df.index.get_level_values('classification_algorithm') == algo)].copy()
                final_metric_dict['AHI_cut_off'].append(final_metric_df.index.get_level_values('AHI_cut_off').values[0])
                final_metric_dict['classification_algorithm'].append(algo)
                for metric in self.metrics:
                    metric_arr = final_metric_df.loc[:, [(metric, 'mean')]].values
                    final_metric_dict[f'{metric}_mean'].append(np.mean(metric_arr))
                    final_metric_dict[f'{metric}_std'].append(np.std(metric_arr))
                    final_metric_dict[f'{metric}_min'].append(np.min(metric_arr))
                    final_metric_dict[f'{metric}_max'].append(np.max(metric_arr))
                    final_metric_dict[f'{metric}_list'].append(','.join(str(x) for x in np.unique(metric_arr)))

        final_metric_df = pd.DataFrame(final_metric_dict)
        self.final_metric_df_hp = final_metric_df
        
        for metric in self.metrics:
            final_metric_df[metric] = [f"{row[f'{metric}_mean']*100:.2f} ({row[f'{metric}_std']*100:.2f})" for _, row in final_metric_df.iterrows()]

        mean_std_df = final_metric_df.loc[:, ['classification_algorithm', 'AHI_cut_off']+self.metrics].copy()
        mean_std_df['sort_index'] = [2*n for n in range(0, 12)]
        
        for metric in self.metrics:
            final_metric_df[metric] = [f"({row[f'{metric}_min']*100:.2f}–{row[f'{metric}_max']*100:.2f})" for _, row in final_metric_df.iterrows()]

        range_df = final_metric_df.loc[:, ['classification_algorithm', 'AHI_cut_off']+self.metrics].copy()
        range_df['sort_index'] = [2*n+1 for n in range(0, 12)]
        
        convert_df = pd.concat([mean_std_df, range_df])
        convert_df.sort_values(by=['classification_algorithm', 'sort_index'], inplace=True)
        convert_df.drop(['sort_index'], axis=1, inplace=True)

        convert_df = convert_df.loc[:, ['classification_algorithm', 'AHI_cut_off', 'accuracy', 'auc', 'f1_score', 'precision', 'recall']]

        return convert_df

    def report(self) -> pd.DataFrame:
        report_default = self._default_report()
        report_hp_tuned = self._hp_report()

        return pd.concat([report_default, report_hp_tuned])
    
    def _default_report(self) -> pd.DataFrame:
        best_dict = {}
        for p in self.ahi_points:
            for i, metric in enumerate(self.metrics):
                if i == 0:
                    best_dict[p] = pd.concat([
                        self.df_dict_default[p].loc[self.df_dict_default[p]['cluster'] == 0].sort_values(by=metric, ascending=False).head(1).copy(),
                        self.df_dict_default[p].loc[self.df_dict_default[p]['cluster'] == 1].sort_values(by=metric, ascending=False).head(1).copy(),
                    ])
                    best_dict[p]['best_result_metric'] = metric
                    best_dict[p]['AHI_cut_off'] = p
                else:
                    cl_0 = self.df_dict_default[p].loc[self.df_dict_default[p]['cluster'] == 0].sort_values(by=metric, ascending=False).head(1).copy()
                    cl_1 = self.df_dict_default[p].loc[self.df_dict_default[p]['cluster'] == 1].sort_values(by=metric, ascending=False).head(1).copy()
                    
                    cl_0['best_result_metric'] = metric
                    cl_1['best_result_metric'] = metric
                    
                    cl_0['AHI_cut_off'] = p
                    cl_1['AHI_cut_off'] = p
                    
                    best_dict[p] = pd.concat([
                        best_dict[p],
                        cl_0,
                        cl_1,
                    ])

        best_df = pd.concat([best_dict[p] for p in self.ahi_points])

        best_none_dict = {}
        for p in self.ahi_points:
            for i, metric in enumerate(self.metrics):
                if i == 0:
                    best_none_dict[p] = pd.concat([
                        self.df_dict_default[p].loc[(self.df_dict_default[p]['feature_engineering'] == 'None') & (self.df_dict_default[p]['cluster'] == 0)].sort_values(by=metric, ascending=False).head(1).copy(),
                        self.df_dict_default[p].loc[(self.df_dict_default[p]['feature_engineering'] == 'None') & (self.df_dict_default[p]['cluster'] == 1)].sort_values(by=metric, ascending=False).head(1).copy(),
                    ])
                    best_none_dict[p]['best_result_metric'] = metric
                    best_none_dict[p]['AHI_cut_off'] = p
                else:
                    cl_0 = self.df_dict_default[p].loc[(self.df_dict_default[p]['feature_engineering'] == 'None') & (self.df_dict_default[p]['cluster'] == 0)].sort_values(by=metric, ascending=False).head(1).copy()
                    cl_1 = self.df_dict_default[p].loc[(self.df_dict_default[p]['feature_engineering'] == 'None') & (self.df_dict_default[p]['cluster'] == 1)].sort_values(by=metric, ascending=False).head(1).copy()
                    
                    cl_0['best_result_metric'] = metric
                    cl_1['best_result_metric'] = metric
                    
                    cl_0['AHI_cut_off'] = p
                    cl_1['AHI_cut_off'] = p
                    
                    best_none_dict[p] = pd.concat([
                        best_none_dict[p],
                        cl_0,
                        cl_1,
                    ])

        best_none_df = pd.concat([best_none_dict[p] for p in self.ahi_points])

        best_df['feature_engineered'] = 'Clustering with feature engineering'
        best_none_df['feature_engineered'] = 'Clustering only'
        default_df = pd.concat([best_df, best_none_df])

        metric_result_df = self._metric_results(default_df)
        
        metric_result_df = metric_result_df.droplevel('best_result_metric')
        self.metric_result_df_default = metric_result_df

        final_metric_dict = {
            'feature_engineered': [],
            'AHI_cut_off': [],
            'accuracy_mean': [],
            'accuracy_std': [],
            'accuracy_min': [],
            'accuracy_max': [],
            'accuracy_list': [],
            'f1_score_mean': [],
            'f1_score_std': [],
            'f1_score_min': [],
            'f1_score_max': [],
            'f1_score_list': [],
            'precision_mean': [],
            'precision_std': [],
            'precision_min': [],
            'precision_max': [],
            'precision_list': [],
            'recall_mean': [],
            'recall_std': [],
            'recall_min': [],
            'recall_max': [],
            'recall_list': [],
            'auc_mean': [],
            'auc_std': [],
            'auc_min': [],
            'auc_max': [],
            'auc_list': [],
        }

        feature_engineered_list = ['Clustering with feature engineering', 'Clustering only']
        for yn in feature_engineered_list:
            for p in self.ahi_points:
                final_metric_df = metric_result_df.iloc[(metric_result_df.index.get_level_values('feature_engineered') == yn) & (metric_result_df.index.get_level_values('AHI_cut_off') == p)].copy()
                final_metric_dict['feature_engineered'].append(final_metric_df.index.get_level_values('feature_engineered').values[0])
                final_metric_dict['AHI_cut_off'].append(final_metric_df.index.get_level_values('AHI_cut_off').values[0])
                for metric in self.metrics:
                    metric_arr = final_metric_df.loc[:, [(metric, 'mean')]].values
                    if metric == 'auc':
                        self.fpr_tpr_dict[yn][p]['auc_mean'] = np.mean(metric_arr)
                    final_metric_dict[f'{metric}_mean'].append(np.mean(metric_arr))
                    final_metric_dict[f'{metric}_std'].append(np.std(metric_arr))
                    final_metric_dict[f'{metric}_min'].append(np.min(metric_arr))
                    final_metric_dict[f'{metric}_max'].append(np.max(metric_arr))
                    final_metric_dict[f'{metric}_list'].append(','.join(str(x) for x in np.unique(metric_arr)))

        final_metric_df = pd.DataFrame(final_metric_dict)
        self.final_metric_df_default = final_metric_df
        
        for metric in self.metrics:
            final_metric_df[metric] = [f"{row[f'{metric}_mean']*100:.2f} ({row[f'{metric}_std']*100:.2f})" for _, row in final_metric_df.iterrows()]

        mean_std_df = final_metric_df.loc[:, ['feature_engineered', 'AHI_cut_off']+self.metrics].copy()
        mean_std_df['sort_index'] = [2*n for n in range(0, 6)]
        
        for metric in self.metrics:
            final_metric_df[metric] = [f"({row[f'{metric}_min']*100:.2f}–{row[f'{metric}_max']*100:.2f})" for _, row in final_metric_df.iterrows()]

        range_df = final_metric_df.loc[:, ['feature_engineered', 'AHI_cut_off']+self.metrics].copy()
        range_df['sort_index'] = [2*n+1 for n in range(0, 6)]
        
        convert_df = pd.concat([mean_std_df, range_df])
        convert_df.sort_values(by='sort_index', inplace=True)
        convert_df.drop(['sort_index'], axis=1, inplace=True)

        convert_df.sort_values(by='feature_engineered', inplace=True)
        convert_df = convert_df.loc[:, ['feature_engineered', 'AHI_cut_off', 'accuracy', 'auc', 'f1_score', 'precision', 'recall']]

        return convert_df

    def _hp_report(self) -> pd.DataFrame:
        best_dict = {}
        for p in self.ahi_points:
            for i, metric in enumerate(self.metrics):
                if i == 0:
                    best_dict[p] = pd.concat([
                        self.df_dict_hp[p].loc[self.df_dict_hp[p]['cluster'] == 0].sort_values(by=metric, ascending=False).head(1).copy(),
                        self.df_dict_hp[p].loc[self.df_dict_hp[p]['cluster'] == 1].sort_values(by=metric, ascending=False).head(1).copy(),
                    ])
                    best_dict[p]['best_result_metric'] = metric
                    best_dict[p]['AHI_cut_off'] = p
                else:
                    cl_0 = self.df_dict_hp[p].loc[self.df_dict_hp[p]['cluster'] == 0].sort_values(by=metric, ascending=False).head(1).copy()
                    cl_1 = self.df_dict_hp[p].loc[self.df_dict_hp[p]['cluster'] == 1].sort_values(by=metric, ascending=False).head(1).copy()
                    
                    cl_0['best_result_metric'] = metric
                    cl_1['best_result_metric'] = metric
                    
                    cl_0['AHI_cut_off'] = p
                    cl_1['AHI_cut_off'] = p
                    
                    best_dict[p] = pd.concat([
                        best_dict[p],
                        cl_0,
                        cl_1,
                    ])

        best_df = pd.concat([best_dict[p] for p in self.ahi_points])
        best_df['feature_engineered'] = 'Clustering with feature engineering and hyperparameter tuning'
        metric_result_df = self._metric_results(best_df, default=False)

        metric_result_df = metric_result_df.droplevel('best_result_metric')
        self.metric_result_df_hp = metric_result_df

        final_metric_dict = {
            'AHI_cut_off': [],
            'accuracy_mean': [],
            'accuracy_std': [],
            'accuracy_min': [],
            'accuracy_max': [],
            'accuracy_list': [],
            'f1_score_mean': [],
            'f1_score_std': [],
            'f1_score_min': [],
            'f1_score_max': [],
            'f1_score_list': [],
            'precision_mean': [],
            'precision_std': [],
            'precision_min': [],
            'precision_max': [],
            'precision_list': [],
            'recall_mean': [],
            'recall_std': [],
            'recall_min': [],
            'recall_max': [],
            'recall_list': [],
            'auc_mean': [],
            'auc_std': [],
            'auc_min': [],
            'auc_max': [],
            'auc_list': [],
        }

        for p in self.ahi_points:
            final_metric_df = metric_result_df.iloc[(metric_result_df.index.get_level_values('AHI_cut_off') == p)].copy()
            final_metric_dict['AHI_cut_off'].append(final_metric_df.index.get_level_values('AHI_cut_off').values[0])
            for metric in self.metrics:
                metric_arr = final_metric_df.loc[:, [(metric, 'mean')]].values
                if metric == 'auc':
                    self.fpr_tpr_dict['Clustering with feature engineering and hyperparameter tuning'][p]['auc_mean'] = np.mean(metric_arr)
                final_metric_dict[f'{metric}_mean'].append(np.mean(metric_arr))
                final_metric_dict[f'{metric}_std'].append(np.std(metric_arr))
                final_metric_dict[f'{metric}_min'].append(np.min(metric_arr))
                final_metric_dict[f'{metric}_max'].append(np.max(metric_arr))
                final_metric_dict[f'{metric}_list'].append(','.join(str(x) for x in np.unique(metric_arr)))

        final_metric_df = pd.DataFrame(final_metric_dict)
        self.final_metric_df_hp = final_metric_df
        
        for metric in self.metrics:
            final_metric_df[metric] = [f"{row[f'{metric}_mean']*100:.2f} ({row[f'{metric}_std']*100:.2f})" for _, row in final_metric_df.iterrows()]

        mean_std_df = final_metric_df.loc[:, ['AHI_cut_off']+self.metrics].copy()
        mean_std_df['sort_index'] = [2*n for n in range(0, 3)]
        
        for metric in self.metrics:
            final_metric_df[metric] = [f"({row[f'{metric}_min']*100:.2f}–{row[f'{metric}_max']*100:.2f})" for _, row in final_metric_df.iterrows()]

        range_df = final_metric_df.loc[:, ['AHI_cut_off']+self.metrics].copy()
        range_df['sort_index'] = [2*n+1 for n in range(0, 3)]
        
        convert_df = pd.concat([mean_std_df, range_df])
        convert_df.sort_values(by='sort_index', inplace=True)
        convert_df.drop(['sort_index'], axis=1, inplace=True)

        convert_df['feature_engineered'] = 'Clustering with feature engineering and hyperparameter tuning'
        convert_df = convert_df.loc[:, ['feature_engineered', 'AHI_cut_off', 'accuracy', 'auc', 'f1_score', 'precision', 'recall']]

        return convert_df

    def _metric_results(self, df: pd.DataFrame, default: boolean = True, ml_algorithm: boolean = False) -> pd.DataFrame:
        groupby = ['feature_engineered', 'AHI_cut_off', 'best_result_metric']
        
        if ml_algorithm == True:
            groupby = ['classification_algorithm', 'AHI_cut_off', 'best_result_metric']

        return_df = df.loc[:, groupby+['accuracy', 'f1_score', 'precision', 'recall', 'auc', 'fpr', 'tpr']].copy()
        metrics_loc = [(metric, 'mean') for metric in self.metrics]

        if default == False:
            groupby = ['AHI_cut_off', 'best_result_metric']
        grouped_return_df = return_df.groupby(groupby).describe().loc[:, metrics_loc].copy()
        
        for fe in np.unique(return_df['feature_engineered'].values):
            temp_dict = {
                'p5': {'fpr': [], 'tpr': [],},
                'p15': {'fpr': [], 'tpr': [],},
                'p30': {'fpr': [], 'tpr': [],},
            }
            for p in self.ahi_points:
                for _, row in return_df.loc[return_df['feature_engineered'] == fe].iterrows():
                    temp_dict[p]['fpr'].append(list(map(float, row['fpr'].split(','))))
                    temp_dict[p]['tpr'].append(list(map(float, row['tpr'].split(','))))

                temp_dict[p]['fpr'] = np.array(temp_dict[p]['fpr']).mean(axis=0)
                temp_dict[p]['tpr'] = np.array(temp_dict[p]['tpr']).mean(axis=0)
            self.fpr_tpr_dict[fe] = temp_dict.copy()

        return grouped_return_df


def find_best(ahi_point: str) -> pd.DataFrame:
    df = pd.read_csv(f'{ROOT_PATH}/cluster_results/cluster_result_{ahi_point}.csv')
    _duplicated_accuracy_test(df)

    test_best = df.sort_values(by='test_average_accuracy', ascending=False).head(5).copy()
    test_best['best_from'] = 'test'
    
    return test_best

def find_rfe_best(ahi_point: str) -> pd.DataFrame:
    df = pd.read_csv(f'{ROOT_PATH}/cluster_results/cluster_result_{ahi_point}.csv')
    _duplicated_accuracy_test(df)

    test_best = df.loc[df['selector'] == 'RFE'].sort_values(by='test_average_accuracy', ascending=False).head(5).copy()
    test_best['best_from'] = 'test'
    
    return test_best

def display_best_plots(ahi_point: str, rfe: bool = False) -> None:
    if rfe == True:
        best_df = find_rfe_best(ahi_point).head(1)
    else:
        best_df = find_best(ahi_point).head(1)
    plot_dirs = ['elbow', 'pca']

    for _, row in best_df.iterrows():
        print(f"========== BEST FROM {row['best_from'].upper()} ==========")
        print(f"- Clustering Algorithm: {row['clustering_algorithm']}")
        print(f"- Scaler: {row['scaler']}")
        print(f"- Selector: {row['selector']}")
        print(f"- Selected Features: {row['selected_features'].replace(',', ', ')}")
        print(f"- Test Dataset Cluster Average Accuracy: {row['test_average_accuracy']}")
        print(f"- Valid Dataset Cluster Average Accuracy: {row['valid_oof_accuracy']}")
        
        filename = f"{row['clustering_algorithm']}+{row['scaler']}+{row['selector']}+{row['selected_features']}"
        for plot_dir in plot_dirs:
            display(
                Image(
                    filename=f'{ROOT_PATH}/images/{ahi_point}/{plot_dir}/{filename}.png'
                )
            )

def get_best_df(ahi_point: str, best_from: str, rfe: bool = False) -> pd.DataFrame:
    if rfe == True:
        best_df = find_rfe_best(ahi_point).head(1)
    else:
        best_df = find_best(ahi_point).head(1)

    for _, row in best_df.iterrows():
        if row['best_from'] == best_from:
            return_df = pd.read_csv(f"{ROOT_PATH}/data/clustered/{ahi_point}/{row['clustering_algorithm']}+{row['scaler']}+{row['selector']}+{row['selected_features']}.csv")
            return_df['clustering_algorithm'] = row['clustering_algorithm']
            return return_df

def _duplicated_accuracy_test(df: pd.DataFrame) -> None:
    test_acc = df.sort_values(by='test_average_accuracy', ascending=False).head(2)['test_average_accuracy'].values
    valid_acc = df.sort_values(by='valid_oof_accuracy', ascending=False).head(2)['valid_oof_accuracy'].values

    if test_acc[0] == test_acc[1]:
        print("Warning: Test top 2 results have the same accuracy.")
    if valid_acc[0] == valid_acc[1]:
        print("Warning: Valid top 2 results have the same accuracy.")
    