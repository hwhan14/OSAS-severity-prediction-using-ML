from turtle import title
from typing import List, Dict
import gc

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.feature_selection import *
from sklearn.metrics import *
from sklearn.cluster import *
from sklearn.preprocessing import *
from yellowbrick.cluster import KElbowVisualizer
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.base import ClusterMixin

import warnings 
warnings.filterwarnings('ignore')
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

from src.utils import ROOT_PATH, LABEL_DICT


class GaussianMixtureCluster(GaussianMixture, ClusterMixin):
    """
    Another GaussianMixture class for elbow method
    """

    def fit(self, X):
        super().fit(X)
        self.labels_ = self.predict(X)
        return self

    def get_params(self, **kwargs):
        output = super().get_params(**kwargs)
        output["n_clusters"] = output.get("n_components", None)
        return output

    def set_params(self, **kwargs):
        kwargs["n_components"] = kwargs.pop("n_clusters", None)
        return super().set_params(**kwargs)

class BayesianGaussianMixtureCluster(BayesianGaussianMixture, ClusterMixin):
    """
    Another BayesianGaussianMixture class for elbow method
    """

    def fit(self, X):
        super().fit(X)
        self.labels_ = self.predict(X)
        return self

    def get_params(self, **kwargs):
        output = super().get_params(**kwargs)
        output["n_clusters"] = output.get("n_components", None)
        return output

    def set_params(self, **kwargs):
        kwargs["n_components"] = kwargs.pop("n_clusters", None)
        return super().set_params(**kwargs)


class Cluster:
    def __init__(self, df: pd.DataFrame, categorical_features: List[str], ahi_point: str) -> None:
        self.df = df
        self.ahi_point = ahi_point

        self.clusters = [AgglomerativeClustering, KMeans, BisectingKMeans, GaussianMixture, BayesianGaussianMixture]
        self.clusters_str = ['AgglomerativeClustering', 'KMeans', 'BisectingKMeans', 'GaussianMixture', 'BayesianGaussianMixture']

        self.scalers = ['None', MinMaxScaler, Normalizer, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler]
        self.scalers_str = ['None', 'MinMaxScaler', 'Normalizer', 'PowerTransformer', 'QuantileTransformer', 'RobustScaler', 'StandardScaler']

        self.selectors = [
            mutual_info_classif,
            RFE,
        ]
        self.selectors_str = [
            'mutual_info_classif',
            'RFE',
        ]

        self.clf = LGBMClassifier(random_state=42)
        self.k_fold = StratifiedKFold(n_splits=5)

        self.categorical_features = categorical_features
        self.numerical_features = list(set(self.df.columns) - set(self.categorical_features) - {'set', 'AHI_group'})
        self.features = list(set(self.df.columns) - {'AHI', 'AHI_group', 'set'})

        self.k_features = [3, 5, 7, 10]
        self.return_df = self.df
        self.feature_select_result = []
    
    def feature_select(self) -> List:
        for i, sc in enumerate(self.scalers):
            print(f'Scaler: {self.scalers_str[i]}')
            scaled_df = self.df.copy()
            if sc == 'None':
                pass
            else:
                scaler = sc()
                scaled_df[self.numerical_features] = scaler.fit_transform(self.df[self.numerical_features])
            for j, sel in enumerate(self.selectors):
                print(f'Selector: {self.selectors_str[j]}')
                try:
                    for k in self.k_features:
                        print(f'k: {k}')
                        if self.selectors_str[j] == 'RFE':
                            selector = sel(self.clf, n_features_to_select=k)
                            selector.fit_transform(X=scaled_df[self.features], y=scaled_df['AHI_group'])
                            selected_features = list(selector.get_feature_names_out())
                        elif self.selectors_str[j] == 'mutual_info_classif':
                            mi_scores = sel(X=scaled_df[self.features], y=scaled_df['AHI_group'])
                            selected_features = list(selector.get_feature_names_out())
                        
                        self.feature_select_result.append((i, j, k, selected_features))
                except Exception as e:
                    print(e)
                    continue
            print('\n\n')

            del scaled_df
            gc.collect()

        return self.feature_select_result
        
    def feature_select_combined(self) -> List:
        for i, sc in enumerate(self.scalers):
            print(f'Scaler: {self.scalers_str[i]}')
            scaled_df = self.df.copy()
            if sc == 'None':
                pass
            else:
                scaler = sc()
                scaled_df[self.numerical_features] = scaler.fit_transform(scaled_df[self.numerical_features])
                print('scaling done')
            try:
                mis_orig = mutual_info_classif(scaled_df[self.features], scaled_df['AHI_group'])
                mi_features = [(f, mi) for f, mi in zip(self.features, mis_orig)]
                mi_features.sort(key=lambda tup: tup[1])
                threshold = np.mean(mis_orig)
                filtered_features = [f for f, mi in mi_features if mi > threshold]

                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111)

                ax.set_xlabel("Mutual information score")
                ax.set_ylabel("Feature")

                mis = [mi for _, mi in mi_features]
                pos = np.arange(len(mis))
                plt.barh(pos, mis, color=sns.color_palette('crest', len(mis)))
                plt.yticks(pos, [LABEL_DICT[f] for f, _ in mi_features])
                for idx, tick in enumerate(ax.yaxis.get_major_ticks()) :
                    if np.squeeze(mis)[idx] > threshold :
                        tick.label.set_color("red")
                        
                ax.axvline(x=threshold, ymin=0, color='black', ls='--', lw=1, label=f"mean = {np.round(threshold, 3)}")
                ax.grid(color='gray', alpha=0.3)
                legend = ax.legend(loc="best", frameon=True, framealpha=0.6)
                ax.add_artist(legend)
                plt.savefig(f'{ROOT_PATH}/images/{self.ahi_point}/mi/{self.scalers_str[i]}.png', bbox_inches='tight')
                plt.clf()

                min_features_to_select = 1
                rfecv = RFECV(
                    estimator=self.clf,
                    step=1,
                    cv=self.k_fold,
                    scoring="accuracy",
                    min_features_to_select=min_features_to_select,
                )
                cv_df = scaled_df[scaled_df['set'] != 'test'].copy()
                rfecv.fit(cv_df[filtered_features], cv_df['AHI_group'])

                print("Optimal number of features : %d" % rfecv.n_features_)
                print(rfecv.grid_scores_.shape)

                fig = plt.figure()
                ax = fig.add_subplot(111)

                ax.set_xlabel("Number of features selected (n)")
                ax.set_ylabel("Accuracy (%)")

                ax.plot(
                    range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select),
                    rfecv.cv_results_['mean_test_score']*100,
                    marker='o',
                )
                ax.fill_between(
                    range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select),
                    np.min(rfecv.grid_scores_, axis=1)*100,
                    np.max(rfecv.grid_scores_, axis=1)*100,
                    alpha=0.2,
                )
                best_idx = np.argmax(rfecv.cv_results_['mean_test_score'])+1
                ax.axvline(x=best_idx, ymin=0, color='black', ls='--', lw=1, label=f"best at n = {best_idx}, accuracy = {np.round(np.max(rfecv.cv_results_['mean_test_score'])*100, 2)}")
                legend = ax.legend(loc="best", frameon=True, framealpha=0.6)
                ax.add_artist(legend)
                ax.grid(color='gray', alpha=0.3)

                selected_features = rfecv.get_feature_names_out()
                filename = f"{self.scalers_str[i]}+MIRFECV+{','.join(selected_features)}"

                plt.savefig(f'{ROOT_PATH}/images/{self.ahi_point}/num_features/{filename}.png')
                plt.clf()

                self.feature_select_result.append((i, 11, len(selected_features), selected_features))
            except Exception as e:
                print(e)
                continue
            print('\n\n')

            del scaled_df
            gc.collect()

        return self.feature_select_result

    def cluster(self, feature_select_result=[], combined=False) -> None:
        if len(feature_select_result) == 0:
            feature_select_result = self.feature_select_result
        for i, res in enumerate(feature_select_result):
            print(f'{i+1}/{len(feature_select_result)}')
            scaler_idx = res[0]
            selector_idx = res[1]
            num_features = res[2]
            selected_features = res[3]
            for j, cls in enumerate(self.clusters):
                print(f'Cluster Algorithm: {self.clusters_str[j]}')
                try:
                    scaled_df = self.df.copy()
                    sc = self.scalers[scaler_idx]

                    if combined == True:
                        selector_str = 'MIRFECV'
                    else:
                        selector_str = self.selectors_str[selector_idx]

                    if sc == 'None':
                        pass
                    else:
                        scaler = sc()
                        scaled_df[self.numerical_features] = scaler.fit_transform(self.df[self.numerical_features])

                    X = scaled_df[selected_features]

                    if 'Mixture' in self.clusters_str[j]:
                        if 'Bayesian' in self.clusters_str[j]:
                            mixture = BayesianGaussianMixtureCluster()
                        else:
                            mixture = GaussianMixtureCluster()
                        visualizer = KElbowVisualizer(mixture, k=(2, 8), metric='silhouette', force_model=True)
                    else:
                        visualizer = KElbowVisualizer(cls(), k=(2, 8), metric='silhouette')
                    visualizer.fit(X)
                    plot_filename = f"{self.clusters_str[j]}+{self.scalers_str[scaler_idx]}+{selector_str}+{','.join(selected_features)}"
                    visualizer.show(outpath=f"{ROOT_PATH}/images/{self.ahi_point}/elbow/{plot_filename}.png")
                    plt.clf()

                    if j in [3, 4]:
                        cluster = cls(n_components=visualizer.elbow_value_)
                    else:
                        cluster = cls(n_clusters=visualizer.elbow_value_)

                    clustered_result = cluster.fit_predict(X)
                    scaled_df['cluster'] = clustered_result
                    
                    self._pca_visualize(scaled_df, list(selected_features)+['cluster'], plot_filename)

                    scaled_df['scaler'] = self.scalers_str[scaler_idx]
                    scaled_df['selector'] = selector_str
                    scaled_df['num_features'] = num_features
                    scaled_df['selected_features'] = ','.join(selected_features)
                    scaled_df['elbow_k'] = visualizer.elbow_value_

                    scaled_df.to_csv(f'{ROOT_PATH}/data/clustered/{self.ahi_point}/{plot_filename}.csv', index=False)

                    return_df = self._predict(scaled_df, self.clusters_str[j])

                    del scaled_df
                    gc.collect()

                    if len(return_df.values) == 0:
                        print('Encountered error during predicting. Skipping...')
                        continue
                    else:
                        if i == 0 and j == 0:
                            self.return_df = return_df
                        else:
                            self.return_df = pd.concat([self.return_df, return_df], axis=0)

                        print(f'{self.clusters_str[j]} Done.')
                except Exception as e:
                    print(e)
                    print('Encountered error during clustering. Skipping...')
                    continue
            print('\n\n')
    
    def _predict(self, scaled_df: pd.DataFrame, cluster_algorithm: str) -> pd.DataFrame:
        df_dict = {
            'clustering_algorithm': [],
            'test_average_accuracy': [],
            'test_clusters_accuracy': [],
            'valid_oof_accuracy': [],
            'clusters_accuracy': [],
            'scaler': [],
            'selector': [],
            'selected_features': [],
            'n_cluster': [],
        }
        
        cluster_score_list = []
        test_score_list = []
        n_cluster = scaled_df['elbow_k'].values[0]
        df_dict['n_cluster'].append(n_cluster)

        df_dict['scaler'].append(scaled_df['scaler'].values[0])
        df_dict['selector'].append(scaled_df['selector'].values[0])
        df_dict['selected_features'].append(scaled_df['selected_features'].values[0])

        df_dict['clustering_algorithm'] = cluster_algorithm

        try:
            for cluster in range(0, n_cluster):
                df_c = scaled_df[scaled_df['cluster'] == cluster].copy()
                
                X = df_c[(df_c['set'] == 'train') | (df_c['set'] == 'valid')].copy()
                y = X['AHI_group'].copy()
                X.drop(['AHI_group', 'AHI', 'set', 'cluster', 'selector', 'scaler', 'num_features', 'selected_features', 'elbow_k'], axis=1, inplace=True)

                X_test = df_c[(df_c['set'] == 'test')].copy()
                y_test = X_test['AHI_group'].copy()
                X_test.drop(['AHI_group', 'AHI', 'set', 'cluster', 'selector', 'scaler', 'num_features', 'selected_features', 'elbow_k'], axis=1, inplace=True)

                score_list = []
                test_preds = np.zeros((X_test.shape[0], len(y_test.unique())))
                for i, (train_idx, valid_idx) in enumerate(self.k_fold.split(X, y)):
                    X_train, X_valid = X.iloc[train_idx, :], X.iloc[valid_idx, :]
                    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

                    self.clf.fit(
                        X_train, y_train,
                        eval_set=[(X_train, y_train), (X_valid, y_valid)],
                        eval_names=['train', 'valid'],
                        callbacks=[early_stopping(100, verbose=0), log_evaluation(period=0)],
                    )
                    valid_pred = self.clf.predict(X_valid)
                    valid_score = accuracy_score(y_valid, valid_pred)

                    test_preds += self.clf.predict_proba(X_test)

                    score_list.append(valid_score)

                avg_accuracy = np.mean(score_list)
                cluster_score_list.append(avg_accuracy)

                cluster_test_accuracy = accuracy_score(y_test, np.argmax(test_preds, axis=1))
                test_score_list.append(cluster_test_accuracy)

            df_dict['test_average_accuracy'].append(np.mean(test_score_list))
            df_dict['test_clusters_accuracy'].append(str(test_score_list))
            df_dict['clusters_accuracy'].append(str(cluster_score_list))
            df_dict['valid_oof_accuracy'].append(sum(cluster_score_list)/len(cluster_score_list))
        except Exception as e:
            print(e)
            df_dict = {
                'clustering_algorithm': [],
                'test_average_accuracy': [],
                'valid_oof_accuracy': [],
                'clusters_accuracy': [],
                'scaler_idx': [],
                'selector_idx': [],
                'selected_features': [],
                'n_cluster': [],
            }

        return_df = pd.DataFrame(df_dict)
        return return_df

    def _pca_visualize(self, scaled_df: pd.DataFrame, selected_features: List[str], filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8,6))
        viz_df = scaled_df[selected_features].copy()

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(viz_df.drop(['cluster'], axis=1))

        pca_dict = {
            'x': pca_result[:, 0],
            'y': pca_result[:, 1],
            'cluster': viz_df['cluster'],
        }
        pca_df = pd.DataFrame(pca_dict)

        scatter = ax.scatter(
            x=pca_df['x'],
            y=pca_df['y'],
            c=pca_df['cluster'],
            alpha=0.5,
            cmap='Dark2',
        )
        
        legend = ax.legend(*scatter.legend_elements(), loc="best", title='Clusters', frameon=True, framealpha=0.6)
        ax.add_artist(legend)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        
        plt.savefig(f'{ROOT_PATH}/images/{self.ahi_point}/pca/{filename}.png')
        plt.clf()

        del viz_df
        gc.collect()
