from typing import List, Tuple
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier, log_evaluation, early_stopping
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from imblearn.over_sampling import SMOTE
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from tqdm import tqdm

import warnings 
warnings.filterwarnings('ignore')

from src.utils import ROOT_PATH


class Classifier:
    def __init__(self, df: pd.DataFrame, categorical_features: List[str], ahi_point: str, root_path: str = ROOT_PATH) -> None:
        self.df = df
        self.categorical_features = categorical_features
        self.ahi_point = ahi_point
        self.k_fold = StratifiedKFold(n_splits=5)

        self.clustering_algorithm = df['clustering_algorithm'].values[0]
        self.scaler = df['scaler'].values[0]
        self.selector = df['selector'].values[0]
        self.selected_features = df['selected_features'].values[0]
        self.n_clusters = df['elbow_k'].values[0]

        self.classifiers = [LGBMClassifier, XGBClassifier, RandomForestClassifier, CatBoostClassifier]
        self.classifiers_str = ['LGBMClassifier', 'XGBClassifier', 'RandomForestClassifier', 'CatBoostClassifier']

        self.fe_df = df.copy()
        self.fe_df['feature_engineering'] = 'None'
        self.fe_df['num_of_applied_eng'] = 0

        self.fe_methods = {
            'A_l1': ['AHI prediction with NC(lying position)', [self.with_pred_l, 0], 1],
            'A_l2': ['AHI prediction with NC(lying position)', [self.with_pred_l, 1], 1],
            'A_l': ['AHI prediction with NC(lying position)', [self.with_pred_l, 2], 1],
            'A_s1': ['AHI prediction with NC(sitting position)', [self.with_pred_s, 0], 1],
            'A_s2': ['AHI prediction with NC(sitting position)', [self.with_pred_s, 1], 1],
            'A_s': ['AHI prediction with NC(sitting position)', [self.with_pred_s, 2], 1],
            'B': ['Body mesurement ratios', [self.additional_features], 2],
            'C': ['Weighted ESS', [self.weighted_ess], 0],
        }

        self.features = list(set(self.df.columns) - {'AHI', 'AHI_group', 'set', 'cluster', 'scaler', 'selector', 'num_features', 'selected_features', 'elbow_k', 'clustering_algorithm'})
        self.fe_list = [
            (0, []),
            (1, ['A_l1']), (1, ['A_l2']), (1, ['A_s1']), (1, ['A_s2']), (1, ['B']), (1, ['C']),
            (2, ['A_l1', 'B']), (2, ['A_l2', 'B']), (2, ['A_l', 'C']),
            (2, ['A_s1', 'B']), (2, ['A_s2', 'B']), (2, ['A_s', 'C']), (2, ['B', 'C']), 
            (3, ['A_l', 'B', 'C']), (3, ['A_s', 'B', 'C']),
        ]

        header_str = 'clustering_algorithm,scaler,selector,selected_features,n_clusters,cluster,feature_engineering,classification_algorithm,accuracy,f1_score,precision,recall,auc,fpr,tpr,threshold,hp_tuned'
        self.result_df = pd.DataFrame(columns=header_str.split(','))

        self.root_path = root_path

    def _load_data(self, df: pd.DataFrame, k_fold: bool) -> Tuple:
        train = df[df['set'] == 'train'].copy()
        valid = df[df['set'] == 'valid'].copy()
        test = df[df['set'] == 'test'].copy()

        X_train = train[self.features].copy()
        y_train = train['AHI_group'].copy()
        X_valid = valid[self.features].copy()
        y_valid = valid['AHI_group'].copy()
        X_test = test[self.features].copy()
        y_test = test['AHI_group'].copy()

        del train, valid, test
        gc.collect()

        if k_fold == True:
            X = pd.concat([X_train, X_valid])
            y = pd.concat([y_train, y_valid])
            return X, X_test, y, y_test
        else:
            return X_train, X_valid, X_test, y_train, y_valid, y_test

    def classify(self, no_feature_engineering: bool = False, hp_tuning: bool = False, test: bool = False, n_trials: int = 100, use_hyperband: bool = True, fe_split: List[int] = []) -> pd.DataFrame:
        """
        if no_feature_engineering is True, all the other parameters will be ignored.
        n_trials, use_hyperband will be ignored if hp_tuning is False.
        n_trials will be 5 if test is True.
        """

        df = self.fe_df.copy()
        if no_feature_engineering == True:
            self._classify_no_fe(df)
            return self.result_df
        else:
            clf_fe_list = list(set(df['feature_engineering'].values))
            clf_fe_list.sort()

            if len(fe_split) != 0:
                clf_fe_list = clf_fe_list[fe_split[0]:fe_split[1]]
            if test == True:
                clf_fe_list = clf_fe_list[0:2]
                if hp_tuning == True:
                    n_trials = 3

            for fe in clf_fe_list:
                clf_df = df[df['feature_engineering'] == fe]
                if hp_tuning == True:
                    print(f'Starting classification with hp tuning: {fe}')
                    self._classify_hp_tuning(clf_df, fe, n_trials, use_hyperband)
                else:
                    print(f'Starting classification: {fe}')
                    self._classify_default(clf_df, fe)
                print(f'{fe} classification done.\n\n')
            
            return self.result_df
    
    def _classify_no_fe(self, df: pd.DataFrame):
        clf_df = df.copy()
        clf_df.drop(['cluster'], axis=1, inplace=True)
        X, X_test, y, y_test = self._load_data(clf_df, k_fold=True)

        cm_dict = {
            'LGBMClassifier': [],
            'XGBClassifier': [],
            'RandomForestClassifier': [],
            'CatBoostClassifier': [],
        }

        test_preds = {
            'LGBMClassifier': np.zeros((X_test.shape[0], len(y_test.unique()))),
            'XGBClassifier': np.zeros((X_test.shape[0], len(y_test.unique()))),
            'RandomForestClassifier': np.zeros((X_test.shape[0], len(y_test.unique()))),
            'CatBoostClassifier': np.zeros((X_test.shape[0], len(y_test.unique()))),
        }

        for train_idx, valid_idx in tqdm(self.k_fold.split(X, y), total=self.k_fold.get_n_splits(), desc=f'k-fold'):
            X_train, X_valid = X.iloc[train_idx, :], X.iloc[valid_idx, :]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            for i, clf in enumerate(self.classifiers):
                clf_name = self.classifiers_str[i]
                classifier = clf(random_state=42)

                if clf_name == 'LGBMClassifier':
                    classifier.fit(
                        X_train, y_train,
                        eval_set=[(X_train, y_train), (X_valid, y_valid)],
                        eval_names=['train', 'valid'],
                        callbacks=[log_evaluation(period=0)],
                    )
                elif clf_name == 'XGBClassifier':
                    classifier.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_valid, y_valid)],
                        verbose=False
                    )
                elif clf_name == 'RandomForestClassifier':
                    classifier.fit(
                        X_train,
                        y_train
                    )
                elif clf_name == 'CatBoostClassifier':
                    train_data = Pool(data=X_train, label=y_train, cat_features=self.categorical_features)
                    valid_data = Pool(data=X_valid, label=y_valid, cat_features=self.categorical_features)
                    classifier.fit(
                        train_data,
                        eval_set=valid_data,
                        verbose=False
                    )
                
                test_preds[clf_name] += classifier.predict_proba(X_test)

        result_dict = {
            'clustering_algorithm': [],
            'scaler': [],
            'selector': [],
            'selected_features': [],
            'n_clusters': [],
            'cluster': [],
            'feature_engineering': [],
            'classification_algorithm': [],
            'TN': [],
            'FN': [],
            'TP': [],
            'FP': [],
            'npv': [],
            'sp': [],
            'accuracy': [],
            'f1_score': [],
            'precision': [],
            'recall': [],
            'auc': [],
            'fpr': [],
            'tpr': [],
            'threshold': [],
            'hp_tuned': [],
        }
        for clf in self.classifiers_str:
            result_dict['clustering_algorithm'].append(self.clustering_algorithm)
            result_dict['scaler'].append(self.scaler)
            result_dict['selector'].append(self.selector)
            result_dict['selected_features'].append(self.selected_features)
            result_dict['n_clusters'].append(0)
            result_dict['cluster'].append(i)
            result_dict['feature_engineering'].append('None(no fe cl)')
            result_dict['classification_algorithm'].append(clf)
            CM = confusion_matrix(y_test, np.argmax(test_preds[clf], axis=1))
            TN = CM[0][0]
            FN = CM[1][0]
            TP = CM[1][1]
            FP = CM[0][1]
            result_dict['TN'].append(TN)
            result_dict['FN'].append(FN)
            result_dict['TP'].append(TP)
            result_dict['FP'].append(FP)
            result_dict['npv'].append(TN/(TN+FN))
            result_dict['sp'].append(TN/(FP+TN))
            result_dict['accuracy'].append(accuracy_score(y_test, np.argmax(test_preds[clf], axis=1)))
            result_dict['f1_score'].append(f1_score(y_test, np.argmax(test_preds[clf], axis=1)))
            result_dict['precision'].append(precision_score(y_test, np.argmax(test_preds[clf], axis=1)))
            result_dict['recall'].append(recall_score(y_test, np.argmax(test_preds[clf], axis=1)))
            result_dict['auc'].append(roc_auc_score(y_test, test_preds[clf][:, 1]))
            fpr, tpr, threshold = roc_curve(y_test, test_preds[clf][:, 1])
            result_dict['fpr'].append(','.join(str(x) for x in fpr))
            result_dict['tpr'].append(','.join(str(x) for x in tpr))
            result_dict['threshold'].append(','.join(str(x) for x in threshold))
            result_dict['hp_tuned'].append(0)

            cm_dict[clf].append(confusion_matrix(y_test, np.argmax(test_preds[clf], axis=1)))

        result_df = pd.DataFrame(result_dict)
        self.result_df = pd.concat([self.result_df, result_df], ignore_index=True)

        for clf in self.classifiers_str:
            cm = np.sum(np.asarray(cm_dict[clf]), axis=0)
            plot_filename = f"{self.clustering_algorithm}+{self.scaler}+{self.selector}+{self.selected_features}+{clf}+no_fe_cl"
            self._plot_confusion_matrix(cm, n=2, plt_name=plot_filename)

    def _classify_default(self, df: pd.DataFrame, feature_engineering: str):
        cm_dict = {
            'LGBMClassifier': [],
            'XGBClassifier': [],
            'RandomForestClassifier': [],
            'CatBoostClassifier': [],
        }
        for i in range(0, self.n_clusters):
            clf_df = df[df['cluster'] == i]
            X, X_test, y, y_test = self._load_data(clf_df, k_fold=True)

            test_preds = {
                'LGBMClassifier': np.zeros((X_test.shape[0], len(y_test.unique()))),
                'XGBClassifier': np.zeros((X_test.shape[0], len(y_test.unique()))),
                'RandomForestClassifier': np.zeros((X_test.shape[0], len(y_test.unique()))),
                'CatBoostClassifier': np.zeros((X_test.shape[0], len(y_test.unique()))),
            }

            for train_idx, valid_idx in tqdm(self.k_fold.split(X, y), total=self.k_fold.get_n_splits(), desc=f'k-fold cluster {i}'):
                X_train, X_valid = X.iloc[train_idx, :], X.iloc[valid_idx, :]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

                for j, clf in enumerate(self.classifiers):
                    clf_name = self.classifiers_str[j]
                    classifier = clf(random_state=42)

                    if clf_name == 'LGBMClassifier':
                        classifier.fit(
                            X_train, y_train,
                            eval_set=[(X_train, y_train), (X_valid, y_valid)],
                            eval_names=['train', 'valid'],
                            callbacks=[log_evaluation(period=0)],
                        )
                    elif clf_name == 'XGBClassifier':
                        classifier.fit(
                            X_train,
                            y_train,
                            eval_set=[(X_valid, y_valid)],
                            verbose=False
                        )
                    elif clf_name == 'RandomForestClassifier':
                        classifier.fit(
                            X_train,
                            y_train
                        )
                    elif clf_name == 'CatBoostClassifier':
                        train_data = Pool(data=X_train, label=y_train, cat_features=self.categorical_features)
                        valid_data = Pool(data=X_valid, label=y_valid, cat_features=self.categorical_features)
                        classifier.fit(
                            train_data,
                            eval_set=valid_data,
                            verbose=False
                        )
                    
                    test_preds[clf_name] += classifier.predict_proba(X_test)

            result_dict = {
                'clustering_algorithm': [],
                'scaler': [],
                'selector': [],
                'selected_features': [],
                'n_clusters': [],
                'cluster': [],
                'feature_engineering': [],
                'classification_algorithm': [],
                'TN': [],
                'FN': [],
                'TP': [],
                'FP': [],
                'npv': [],
                'sp': [],
                'accuracy': [],
                'f1_score': [],
                'precision': [],
                'recall': [],
                'auc': [],
                'fpr': [],
                'tpr': [],
                'threshold': [],
                'hp_tuned': [],
            }
            for clf in self.classifiers_str:
                result_dict['clustering_algorithm'].append(self.clustering_algorithm)
                result_dict['scaler'].append(self.scaler)
                result_dict['selector'].append(self.selector)
                result_dict['selected_features'].append(self.selected_features)
                result_dict['n_clusters'].append(self.n_clusters)
                result_dict['cluster'].append(i)
                result_dict['feature_engineering'].append(feature_engineering)
                result_dict['classification_algorithm'].append(clf)
                CM = confusion_matrix(y_test, np.argmax(test_preds[clf], axis=1))
                TN = CM[0][0]
                FN = CM[1][0]
                TP = CM[1][1]
                FP = CM[0][1]
                result_dict['TN'].append(TN)
                result_dict['FN'].append(FN)
                result_dict['TP'].append(TP)
                result_dict['FP'].append(FP)
                result_dict['npv'].append(TN/(TN+FN))
                result_dict['sp'].append(TN/(FP+TN))
                result_dict['accuracy'].append(accuracy_score(y_test, np.argmax(test_preds[clf], axis=1)))
                result_dict['f1_score'].append(f1_score(y_test, np.argmax(test_preds[clf], axis=1)))
                result_dict['precision'].append(precision_score(y_test, np.argmax(test_preds[clf], axis=1)))
                result_dict['recall'].append(recall_score(y_test, np.argmax(test_preds[clf], axis=1)))
                result_dict['auc'].append(roc_auc_score(y_test, test_preds[clf][:, 1]))
                fpr, tpr, threshold = roc_curve(y_test, test_preds[clf][:, 1])
                result_dict['fpr'].append(','.join(str(x) for x in fpr))
                result_dict['tpr'].append(','.join(str(x) for x in tpr))
                result_dict['threshold'].append(','.join(str(x) for x in threshold))
                result_dict['hp_tuned'].append(0)

                cm_dict[clf].append(CM)

            result_df = pd.DataFrame(result_dict)
            self.result_df = pd.concat([self.result_df, result_df], ignore_index=True)

        for clf in self.classifiers_str:
            cm = np.sum(np.asarray(cm_dict[clf]), axis=0)
            plot_filename = f"{self.clustering_algorithm}+{self.scaler}+{self.selector}+{self.selected_features}+{clf}+{feature_engineering}"
            self._plot_confusion_matrix(cm, n=2, plt_name=plot_filename)
    
    def _classify_hp_tuning(self, df: pd.DataFrame, feature_engineering: str, n_trials: int, use_hyperband: bool):
        """
        Optuna's study uses MedianPruner as the default.
        If use_hyperband is True, HyperbandPruner will be used instead of MedianPruner.

        References:
            - https://github.com/optuna/optuna/wiki/Benchmarks-with-Kurobako#2-optunasamplerstpesampler
            - https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
        """
        sampler = TPESampler(seed=42)

        cm_dict = {
            'LGBMClassifier': [],
            'XGBClassifier': [],
            'RandomForestClassifier': [],
            'CatBoostClassifier': [],
        }
        test_preds = {}
        for i in range(0, self.n_clusters):
            clf_df = df[df['cluster'] == i]

            for j, clf_name in enumerate(self.classifiers_str):
                clf_name = self.classifiers_str[j]

                if clf_name == 'LGBMClassifier':
                    y_test, test_scores, cm = self.lgb_tuning(sampler, clf_df, i, n_trials, use_hyperband)
                elif clf_name == 'XGBClassifier':
                    y_test, test_scores, cm = self.xgb_tuning(sampler, clf_df, i, n_trials, use_hyperband)
                elif clf_name == 'RandomForestClassifier':
                    y_test, test_scores, cm = self.rf_tuning(sampler, clf_df, i, n_trials, use_hyperband)
                elif clf_name == 'CatBoostClassifier':
                    y_test, test_scores, cm = self.cb_tuning(sampler, clf_df, i, n_trials, use_hyperband)
                
                test_preds[clf_name] = test_scores
                cm_dict[clf_name].append(cm)

            result_dict = {
                'clustering_algorithm': [],
                'scaler': [],
                'selector': [],
                'selected_features': [],
                'n_clusters': [],
                'cluster': [],
                'feature_engineering': [],
                'classification_algorithm': [],
                'TN': [],
                'FN': [],
                'TP': [],
                'FP': [],
                'npv': [],
                'sp': [],
                'accuracy': [],
                'f1_score': [],
                'precision': [],
                'recall': [],
                'auc': [],
                'fpr': [],
                'tpr': [],
                'threshold': [],
                'hp_tuned': [],
            }
            for clf in self.classifiers_str:
                result_dict['clustering_algorithm'].append(self.clustering_algorithm)
                result_dict['scaler'].append(self.scaler)
                result_dict['selector'].append(self.selector)
                result_dict['selected_features'].append(self.selected_features)
                result_dict['n_clusters'].append(self.n_clusters)
                result_dict['cluster'].append(i)
                result_dict['feature_engineering'].append(feature_engineering)
                result_dict['classification_algorithm'].append(clf)
                CM = confusion_matrix(y_test, np.argmax(test_preds[clf], axis=1))
                TN = CM[0][0]
                FN = CM[1][0]
                TP = CM[1][1]
                FP = CM[0][1]
                result_dict['TN'].append(TN)
                result_dict['FN'].append(FN)
                result_dict['TP'].append(TP)
                result_dict['FP'].append(FP)
                result_dict['npv'].append(TN/(TN+FN))
                result_dict['sp'].append(TN/(FP+TN))
                result_dict['accuracy'].append(accuracy_score(y_test, np.argmax(test_preds[clf], axis=1)))
                result_dict['f1_score'].append(f1_score(y_test, np.argmax(test_preds[clf], axis=1)))
                result_dict['precision'].append(precision_score(y_test, np.argmax(test_preds[clf], axis=1)))
                result_dict['recall'].append(recall_score(y_test, np.argmax(test_preds[clf], axis=1)))
                result_dict['auc'].append(roc_auc_score(y_test, test_preds[clf][:, 1]))
                fpr, tpr, threshold = roc_curve(y_test, test_preds[clf][:, 1])
                result_dict['fpr'].append(','.join(str(x) for x in fpr))
                result_dict['tpr'].append(','.join(str(x) for x in tpr))
                result_dict['threshold'].append(','.join(str(x) for x in threshold))
                result_dict['hp_tuned'].append(1)

            result_df = pd.DataFrame(result_dict)
            self.result_df = pd.concat([self.result_df, result_df], ignore_index=True)

        for clf in self.classifiers_str:
            cm = np.sum(np.asarray(cm_dict[clf]), axis=0)
            plot_filename = f"{self.clustering_algorithm}+{self.scaler}+{self.selector}+{self.selected_features}+{clf}+{feature_engineering}+hp_tuned"
            self._plot_confusion_matrix(cm, n=2, plt_name=plot_filename)
    
    def _cm_labels(self, cm: np.ndarray, n: int) -> np.ndarray:
        counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
        percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
        labels = [f"{c}\n{p}" for c, p in zip(counts, percentages)]
        labels = np.asarray(labels).reshape(n, n)
        return labels
    
    def _plot_confusion_matrix(self, cm: np.ndarray, n: int, plt_name: str) -> None:
        labels = self._cm_labels(cm, n=n)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=labels, fmt="", cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(f"{self.root_path}/images/{self.ahi_point}/cm/{plt_name}.png")
        plt.clf()

    def _lgb_objective(self, trial, X, y):
        """
        https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
        """

        params_lgb = {
            "boosting_type": trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1.0),
            "max_depth": trial.suggest_int("max_depth", -1, 20),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "max_bin": trial.suggest_int("max_bin", 63, 255),
            "num_iterations": trial.suggest_int("num_iterations", 100, 1000),
        }
        fit_params = {
            'callbacks': [log_evaluation(period=0)]
        }

        model = LGBMClassifier(**params_lgb, random_state=42)
        accuracy = np.mean(cross_val_score(model, X, y, cv=5, scoring='accuracy', fit_params=fit_params))

        return accuracy

    def lgb_tuning(self, sampler, df: pd.DataFrame, cluster: int, n_trials: int, use_hyperband: bool) -> Tuple:
        X, X_test, y, y_test = self._load_data(df, k_fold=True)

        if use_hyperband == True:
            lgb_study = optuna.create_study(
                study_name="lgb_parameter_opt",
                direction="maximize",
                sampler=sampler,
                pruner=HyperbandPruner()
            )
        else:
            lgb_study = optuna.create_study(
                study_name="lgb_parameter_opt",
                direction="maximize",
                sampler=sampler,
            )
        lgb_study.optimize(
            lambda trial: self._lgb_objective(trial, X, y),
            n_trials=n_trials,
        )
        lgb_params = lgb_study.best_params
        
        test_scores = np.zeros((X_test.shape[0], len(y_test.unique())))
        for train_idx, valid_idx in tqdm(self.k_fold.split(X, y), total=self.k_fold.get_n_splits(), desc=f'k-fold cluster {cluster}'):
            X_train, X_valid = X.iloc[train_idx, :], X.iloc[valid_idx, :]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            lgb = LGBMClassifier(**lgb_params, random_state=42)
            lgb.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                eval_names=['train', 'valid'],
                callbacks=[log_evaluation(period=0)],
            )

            test_scores += lgb.predict_proba(X_test)
        
        cluster_cm = confusion_matrix(y_test, np.argmax(test_scores, axis=1))

        return y_test, test_scores, cluster_cm

    def xgb_objective(self, trial, X, y):
        """
        https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
        """
        params_xgb = {
            "booster": trial.suggest_categorical('booster', ['gbtree', 'dart']),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-5, 9e-1),
            "learning_rate": trial.suggest_float("eta", 1e-5, 1.0),
            "max_depth": trial.suggest_int("max_depth", 1, 20),
            "max_leaves": trial.suggest_int("max_leaves", 2, 64),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "n_estimators": trial.suggest_int("num_round", 100, 1000),
        }
        fit_params = {
            'verbose': False
        }

        model = XGBClassifier(**params_xgb, random_state=42)
        accuracy = np.mean(cross_val_score(model, X, y, cv=5, scoring='accuracy', fit_params=fit_params))

        return accuracy
    
    def xgb_tuning(self, sampler, df: pd.DataFrame, cluster: int, n_trials: int, use_hyperband: bool) -> Tuple:
        X, X_test, y, y_test = self._load_data(df, k_fold=True)

        if use_hyperband == True:
            xgb_study = optuna.create_study(
                study_name="xgb_parameter_opt",
                direction="maximize",
                sampler=sampler,
                pruner=HyperbandPruner()
            )
        else:
            xgb_study = optuna.create_study(
                study_name="xgb_parameter_opt",
                direction="maximize",
                sampler=sampler,
            )
        xgb_study.optimize(
            lambda trial: self.xgb_objective(trial, X, y),
            n_trials=n_trials,
        )
        xgb_params = xgb_study.best_params

        test_scores = np.zeros((X_test.shape[0], len(y_test.unique())))
        for train_idx, valid_idx in tqdm(self.k_fold.split(X, y), total=self.k_fold.get_n_splits(), desc=f'k-fold cluster {cluster}'):
            X_train, X_valid = X.iloc[train_idx, :], X.iloc[valid_idx, :]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            xgb = XGBClassifier(**xgb_params, random_state=42)
            xgb.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
            )

            test_scores += xgb.predict_proba(X_test)
        
        cluster_cm = confusion_matrix(y_test, np.argmax(test_scores, axis=1))

        return y_test, test_scores, cluster_cm

    def _rf_objective(self, trial, X, y):
        params_rf = {
            "max_depth": trial.suggest_int("max_depth", 1, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 256),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        }

        model = RandomForestClassifier(**params_rf, random_state=42)
        accuracy = np.mean(cross_val_score(model, X, y, cv=5, scoring='accuracy'))

        return accuracy
    
    def rf_tuning(self, sampler, df: pd.DataFrame, cluster: int, n_trials: int, use_hyperband: bool) -> Tuple:
        X, X_test, y, y_test = self._load_data(df, k_fold=True)
        
        if use_hyperband == True:
            rf_study = optuna.create_study(
                study_name="rf_parameter_opt",
                direction="maximize",
                sampler=sampler,
                pruner=HyperbandPruner()
            )
        else:
            rf_study = optuna.create_study(
                study_name="rf_parameter_opt",
                direction="maximize",
                sampler=sampler,
            )
        rf_study.optimize(
            lambda trial: self._rf_objective(trial, X, y),
            n_trials=n_trials,
        )
        rf_params = rf_study.best_params

        test_scores = np.zeros((X_test.shape[0], len(y_test.unique())))
        for train_idx, valid_idx in tqdm(self.k_fold.split(X, y), total=self.k_fold.get_n_splits(), desc=f'k-fold cluster {cluster}'):
            X_train, X_valid = X.iloc[train_idx, :], X.iloc[valid_idx, :]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            rf = RandomForestClassifier(**rf_params, random_state=42)
            rf.fit(X_train, y_train)

            test_scores += rf.predict_proba(X_test)
        
        cluster_cm = confusion_matrix(y_test, np.argmax(test_scores, axis=1))

        return y_test, test_scores, cluster_cm

    def _cb_objective(self, trial, X, y):
        """
        https://github.com/optuna/optuna-examples/blob/main/catboost/catboost_pruning.py
        """

        params_cb = {
            "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 1, 12),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
            "used_ram_limit": "6gb",
            "eval_metric": "Accuracy",
        }

        if params_cb["bootstrap_type"] == "Bayesian":
            params_cb["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif params_cb["bootstrap_type"] == "Bernoulli":
            params_cb["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

        fit_params = {
            'cat_features' : self.categorical_features,
            'verbose': False,
        }

        model = CatBoostClassifier(**params_cb, random_state=42, cat_features=self.categorical_features)
        accuracy = np.mean(cross_val_score(model, X, y, cv=5, scoring='accuracy', fit_params=fit_params))

        return accuracy

    def cb_tuning(self, sampler, df: pd.DataFrame, cluster: int, n_trials: int, use_hyperband: bool) -> Tuple:
        X, X_test, y, y_test = self._load_data(df, k_fold=True)

        if use_hyperband == True:
            cb_study = optuna.create_study(
                study_name="cb_parameter_opt",
                direction="maximize",
                sampler=sampler,
                pruner=HyperbandPruner()
            )
        else:
            cb_study = optuna.create_study(
                study_name="cb_parameter_opt",
                direction="maximize",
                sampler=sampler,
            )
        cb_study.optimize(
            lambda trial: self._cb_objective(trial, X, y),
            n_trials=n_trials,
        )
        cb_params = cb_study.best_params
        cb_params['used_ram_limit'] = '6gb'
        cb_params['eval_metric'] = 'Accuracy'

        test_scores = np.zeros((X_test.shape[0], len(y_test.unique())))
        for train_idx, valid_idx in tqdm(self.k_fold.split(X, y), total=self.k_fold.get_n_splits(), desc=f'k-fold cluster {cluster}'):
            X_train, X_valid = X.iloc[train_idx, :], X.iloc[valid_idx, :]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            cb = CatBoostClassifier(**cb_params, random_state=42, cat_features=self.categorical_features)
            cb.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                cat_features=self.categorical_features,
                verbose=0,
            )

            test_scores += cb.predict_proba(X_test)
        
        cluster_cm = confusion_matrix(y_test, np.argmax(test_scores, axis=1))

        return y_test, test_scores, cluster_cm

    def feature_engineer(self) -> pd.DataFrame:
        clf_df_orig = self.fe_df[self.features + ['AHI_group', 'set', 'AHI', 'cluster', 'elbow_k']].copy()

        for fe in self.fe_list:
            if len(fe[1]) == 0:
                print('Nothing applied')
                continue
            
            print(f"Applying {', '.join(fe[1])}...")

            clf_df = clf_df_orig.copy()

            apply_list = [self.fe_methods[e] for e in fe[1]]
            apply_list.sort(key = lambda apply_list: apply_list[2])
            for e in apply_list:
                if len(e[1]) == 2:
                    fe_method = e[1][0]
                    clf_df = fe_method(clf_df, e[1][1])
                else:
                    clf_df = e[1][0](clf_df)

            clf_df['feature_engineering'] = ','.join(fe[1])
            clf_df['num_of_applied_eng'] = fe[0]
            self.fe_df = pd.concat([self.fe_df, clf_df], axis=0)

            print('Done.')
        
        del clf_df, clf_df_orig
        gc.collect()
        
        print(f'Feature engineering done. fe_df shape: {self.fe_df.shape}')
        return self.fe_df
    
    # def smote(self, df: pd.DataFrame) -> pd.DataFrame:
    #     num_cluster = self.n_clusters

    #     for i in range(0, num_cluster):
    #         valid = df[(df['set'] == 'valid') & (df['cluster'] == i)].copy()
    #         test = df[(df['set'] == 'test') & (df['cluster'] == i)].copy()

    #         X_train = df[(df['set'] == 'train') & (df['cluster'] == i)].copy()
    #         y_train = X_train['AHI_group'].copy()
    #         X_train.drop(['AHI_group', 'set', 'cluster'], axis=1, inplace=True)

    #         oversampler = SMOTE(random_state=42)
    #         X_train, y_train = oversampler.fit_resample(X_train, y_train)

    #         train = pd.concat([X_train, y_train], axis=1)
    #         train['set'] = 'train'
    #         train['cluster'] = i
    #         train['feature_engineering'] = 'smote'
    #         train['num_of_applied_eng'] = np.nan
    #         train['clustering_algorithm'] = self.clustering_algorithm

    #         if i == 0:
    #             smote_df = pd.concat([train, valid, test])
    #         else:
    #             smote_df = pd.concat([smote_df, train, valid, test])

    #     return smote_df
    
    def additional_features(self, df: pd.DataFrame) -> pd.DataFrame:        
        df['AHR'] = df.abdominal_circumference / df.hip_circumference
        df['ACH'] = df.abdominal_circumference / df.height
        df['NCH_s'] = df.neck_circumference_sit / df.height
        df['NCH_ld'] = df.neck_circumference_lie_down / df.height
        
        return df

    def weighted_ess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        @article{guo2020weighted,
            title={Weighted Epworth sleepiness scale predicted the apnea-hypopnea index better},
            author={Guo, Qi and Song, Wei-dong and Li, Wei and Zeng, Chao and Li, Yan-hong and Mo, Jian-ming and L{\"u}, Zhong-dong and Jiang, Mei},
            journal={Respiratory Research},
            volume={21},
            number={1},
            pages={1--10},
            year={2020},
            publisher={Springer}
        }
        """
        df["ESS1"] = df["ESS1"].replace([0, 1, 2, 3], [0, 3, 4, 5])
        df["ESS2"] = df["ESS2"].replace([0, 1, 2, 3], [0, 3, 4, 5])
        df["ESS3"] = df["ESS3"].replace([0, 1, 2, 3], [0, 4, 5, 6])
        df["ESS4"] = df["ESS4"].replace([0, 1, 2, 3], [0, 2, 3, 4])
        df["ESS6"] = df["ESS6"].replace([0, 1, 2, 3], [0, 5, 6, 7])
        df["ESS8"] = df["ESS8"].replace([0, 1, 2, 3], [0, 5, 6, 7])

        df["ESS"] = (
            df.ESS1 + df.ESS2 + df.ESS3 + df.ESS4 + df.ESS5 + df.ESS6 + df.ESS7 + df.ESS8
        )

        return df

    def _predict_function(self, X, nc, eds, bmi, gc, c):
        """
        AHIpred = NCx0.84 + EDSx7.78 + BMIx0.91 - [8.2xgender constant (1 or 2)+37]
        @article{bouloukaki2011prediction,
            title={Prediction of obstructive sleep apnea syndrome in a large Greek population},
            author={Bouloukaki, Izolde and Kapsimalis, Fotis and Mermigkis, Charalampos and Kryger, Meir and Tzanakis, Nikos and Panagou, Panagiotis and Moniaki, Violeta and Vlachaki, Eleni M and Varouchakis, Georgios and Siafakas, Nikolaos M and others},
            journal={Sleep and Breathing},
            volume={15},
            number={4},
            pages={657--664},
            year={2011},
            publisher={Springer}
        }
        """
        return nc * X[:, 0] + eds * X[:, 1] + bmi * X[:, 2] - (gc * X[:, 3] + c)

    def with_pred_s(self, df: pd.DataFrame, method: int) -> pd.DataFrame:
        """
        method: 0
            greek
        method: 1
            not weighted
        method: 2
            weighted
        """
        
        if method == 0:
            bins = [0, 4, 10, 17, float("inf")]
            df["EDS"] = np.digitize(df["ESS"], bins)
            df["EDS"] = df["EDS"].replace([1, 2, 3, 4], [0, 1, 2, 3])
        elif method == 1:
            bins = [0, 11, 13, 16, float("inf")]
            df["EDS"] = np.digitize(df["ESS"], bins)
            df["EDS"] = df["EDS"].replace([1, 2, 3, 4], [0, 1, 2, 3])
        elif method == 2:
            bins = [0, 15, 20, 30, float("inf")]
            df["EDS"] = np.digitize(df["ESS"], bins)
            df["EDS"] = df["EDS"].replace([1, 2, 3, 4], [0, 1, 2, 3])

        df["sex"] = df["sex"].replace([0, 1], [1, 2])

        x_data = df.loc[:, ["neck_circumference_sit", "EDS", "BMI", "sex"]].values
        y_data = df["AHI"].values

        popt, pcov = curve_fit(self._predict_function, x_data, y_data)
        df["AHI_pred"] = (
            round(popt[0], 2) * df["neck_circumference_sit"]
            + round(popt[1], 2) * df["EDS"]
            + round(popt[2], 2) * df["BMI"]
            - (round(popt[3], 2) * df["sex"] + round(popt[4], 2))
        )

        df["sex"] = df["sex"].replace([1, 2], [0, 1])

        return df

    def with_pred_l(self, df: pd.DataFrame, method: int) -> pd.DataFrame:
        """
        method: 0
            greek
        method: 1
            not weighted
        method: 2
            weighted
        """
        
        if method == 0:
            bins = [0, 4, 10, 17, float("inf")]
            df["EDS"] = np.digitize(df["ESS"], bins)
            df["EDS"] = df["EDS"].replace([1, 2, 3, 4], [0, 1, 2, 3])
        elif method == 1:
            bins = [0, 11, 13, 16, float("inf")]
            df["EDS"] = np.digitize(df["ESS"], bins)
            df["EDS"] = df["EDS"].replace([1, 2, 3, 4], [0, 1, 2, 3])
        elif method == 2:
            bins = [0, 15, 20, 30, float("inf")]
            df["EDS"] = np.digitize(df["ESS"], bins)
            df["EDS"] = df["EDS"].replace([1, 2, 3, 4], [0, 1, 2, 3])

        df["sex"] = df["sex"].replace([0, 1], [1, 2])

        x_data = df.loc[:, ["neck_circumference_sit", "EDS", "BMI", "sex"]].values
        y_data = df["AHI"].values

        popt, pcov = curve_fit(self._predict_function, x_data, y_data)
        df["AHI_pred"] = (
            round(popt[0], 2) * df["neck_circumference_sit"]
            + round(popt[1], 2) * df["EDS"]
            + round(popt[2], 2) * df["BMI"]
            - (round(popt[3], 2) * df["sex"] + round(popt[4], 2))
        )

        df["sex"] = df["sex"].replace([1, 2], [0, 1])

        return df
