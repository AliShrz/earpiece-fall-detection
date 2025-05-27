import copy
import timeit
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import matplotlib.colors as mcolors
sns.set_style("ticks")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scripts import  sisfall, utils
from scripts.utils import get_freq

import matplotlib.ticker as mticker

from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay




def get_models(type=None, models_subset=None):
    all_models = {
        'tabular': {
            'LogisticCV': LogisticRegressionCV(cv=5, n_jobs=-1, solver='newton-cg'),
            'RandomForest': RandomForestClassifier(n_estimators=150, random_state=0),
            'KNN': KNeighborsClassifier(),
            'RidgeCV': RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
            'ExtraTrees': ExtraTreesClassifier(n_estimators=150, max_features=0.1,
                                            criterion="entropy", n_jobs=-1,
                                            random_state =0)
        },
    }

    if type is None: # run all models
        models = {**all_models['tabular'], **all_models['ts']}
    else: # the saner choice :-)
        models = all_models[type]
    if models_subset is not None: # select model subset
        models = {m: models[m] for m in models_subset}
    
    return models


def run_models(X_train, y_train, X_test, y_test, freq, s=7, type=None, subset=None, verbose=1, cm_grid=(1,5), confmat_name='confmat'):
    trained_models = {}
    models = get_models(type=type, models_subset=subset)
    metrics_df = pd.DataFrame(columns=['model', 'window_size', 'runtime', 'auc', 'precision', 'recall', 'specificity', 'f1-score'])
    if verbose > 2:
        fig, axs = plt.subplots(*cm_grid, figsize=(10,3), sharey=True)
        fig.supxlabel('Predicted label')
    for m, (model_name, clf) in enumerate(models.items()):
        clf = make_pipeline(
            StandardScaler(),
            SimpleImputer(missing_values=np.nan, strategy='mean'),
            clf
        )
        # preprocess = model_name in ['LogisticCV', 'RandomForest', 'KNN', 'RidgeCV', 'ExtraTrees']
        metrics, trained_model, cm = predict_eval(
            (model_name, clf), s, freq, X_in=(X_train, X_test),
            y_in=(y_train, y_test), verbose=verbose)
        if verbose > 2:
            ax = axs.flat[m]
            plot_cm(cm, ax=ax, model_name=model_name)
            ax.set_title(model_name)
            ax.set_xlabel('')
            if m>0:
                ax.set_ylabel('')
        these_metrics = pd.DataFrame(data=metrics)
        metrics_df = pd.concat([metrics_df, these_metrics], ignore_index=True)
        trained_models[model_name]=trained_model
    if verbose > 2:
        plt.savefig(f'figs/{confmat_name}.eps', format='eps', bbox_inches='tight')
        plt.show()

    return metrics_df, trained_models

def plot_metrics(df, x='model', pivot='f1-score', compare='metrics', **kwargs):
    default_kwargs = {'figsize': (6,2), 'rot': 0}
    kwargs = {**default_kwargs, **kwargs}
    if compare=='metrics':
        w = max(df['window_size'])
        window_df = df[df['window_size']==w].drop(columns=['window_size', 'runtime'])
        window_df.plot(kind='bar', x='model', **kwargs)
    elif compare=='window_size':
        crosstab = df.pivot_table(pivot, ['model'], 'window_size')
        crosstab.plot(kind='bar', rot=0, **kwargs)
        plt.grid()
        plt.xlabel('')
        plt.ylabel('')
        sns.despine()
    else:
        df.plot(kind='bar', x='model', y=compare)
    plt.legend(loc=9, ncol=3, bbox_to_anchor=(0.5,1.3), title=compare) 

def to_labels(pos_probs, threshold):
    # apply threshold to positive probabilities to create labels
    return (pos_probs >= threshold).astype('int')

def predict_eval(model, s, freq, X_in=None, y_in=None, starttime=None, adapt_threshold=False, verbose=1):
    win_size = int(s*freq)
    target_names = ['ADL', 'Fall']
    model_name, clf = model
    print(f'{model_name}', end='')
    # if preprocess:
    #     clf = make_pipeline(
    #         StandardScaler(),
    #         SimpleImputer(missing_values=np.nan, strategy='mean'),
    #         clf
    #     )
    has_proba = hasattr(clf, 'predict_proba')
    if X_in is not None:
        X_train, X_test = X_in
        y_train, y_test = y_in
    X_train = X_train[:, :win_size]
    X_test = X_test[:, :win_size]
    print('.', end='')
    clf.fit(X_train, y_train)
    print('.', end='')
    if starttime is None:
        starttime = timeit.default_timer()
    if has_proba:
        probs = clf.predict_proba(X_test)[:, 1]
        train_probs = clf.predict_proba(X_train)[:, 1]
        auc_score = np.round(roc_auc_score(y_test, probs), 2)
    else:
        probs = clf.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_score = np.round(auc(fpr, tpr), 2)
    y_pred = clf.predict(X_test)
    runtime = (timeit.default_timer() - starttime)/X_test.shape[0]
    runtime = np.round(runtime * 1000000) # microseconds (ms)
    print('.', end=' ')
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=1)
    if verbose > 1:
        print(f'{model_name} AUC: {auc_score}')
        print(classification_report(y_test, y_pred, target_names=target_names))
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn+fp)
    return dict({'model': [model_name],
                 'window_size': s,
                 'runtime':[np.round(runtime, 2)],
                 'auc': [np.round(auc_score*100, 2)],
                 'precision':[np.round(precision*100, 2)],
                 'recall':[np.round(recall*100, 2)],
                 'specificity': [np.round(specificity*100, 2)],
                 'f1-score':[np.round(f1*100, 2)]}
                ), clf, cm

def plot_cm(cm, model_name, ax, fontsize=20, colorbar=False):
    target_names = ['ADL', 'Fall']
    plt.rcParams.update({'font.size': fontsize})
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, colorbar=colorbar)
    # plt.grid(False)
    plt.rcParams.update({'font.size': 10})


def chunk_list(l, n):
    n_per_set = len(l)//n
    for i in range(1, n_per_set*n, n_per_set):
        chunk = l[i:i+n_per_set]
        if len(chunk) < n_per_set:
            break
        yield chunk

def get_dataset_name(dataset):
    names = {
        sisfall: 'SisFall'
    }
    return 'SisFall'

def cross_validate(dataset, model_type=None, models_subset=None,
                   s=7, cv=5, df=None, verbose=True, random_state=9):
    dataset_name = get_dataset_name(dataset)
    if df is None:
        df = dataset.load()
    rng = np.random.default_rng(random_state)
    subjects = list(df['SubjectID'].unique())
    rng.shuffle(subjects)
    # divide subjects into cv sets
    test_sets = list(chunk_list(subjects, cv))
    freq = get_freq(dataset)
    metrics_df = None
    for i, test_set in enumerate(test_sets):
        test_df = df[df['SubjectID']==test_set[0]]
        train_df = df.drop(df[df['SubjectID']==test_set[0]].index)
        for id in test_set[1:]:
            this_df = df[df['SubjectID']==id]
            test_df = pd.concat([test_df, this_df], ignore_index=True)
            train_df.drop(this_df.index, inplace=True)
            train_df.reset_index().drop(columns=['index'], inplace=True)
        X_train, y_train = dataset.get_X_y(train_df)
        X_test, y_test = dataset.get_X_y(test_df)
        if verbose:
            print(f'\n\n-- fold {i+1} ({len(test_set)} subjects) --')
            print(f"Train set: X: {X_train.shape}, y: {y_train.shape}\
            ([ADLs, Falls])", np.bincount(y_train))
            print(f"Test set: X: {X_test.shape}, y: {y_test.shape}\
            ([ADLs, Falls])", np.bincount(y_test))
        if metrics_df is None:
            metrics_df, _ = run_models(X_train, y_train, X_test, y_test, type=model_type,
                                       freq=freq, s=s, subset=models_subset)
            metrics_df['fold'] = i
        else:
            this_df, _ = run_models(X_train, y_train, X_test, y_test, type=model_type,
                                    freq=freq, s=s, subset=models_subset)
            this_df['fold'] = i
            metrics_df = pd.concat([metrics_df, this_df], ignore_index=True)
    mean_df = metrics_df.groupby(['model']).mean().round(2)
    std_df = metrics_df.groupby(['model']).std().round(2)
    cols = ['model', 'window_size', 'runtime', 'auc', 'precision', 'recall', 'specificity', 'f1-score']
    aggr = {c: [] for c in cols}
    for i in mean_df.index:
        aggr['model'].append(i)
        for col in cols[1:]:
            aggr[col].append(f'{mean_df.loc[i][col]} $\pm$ {std_df.loc[i][col]}')
    aggr_df = pd.DataFrame(data=aggr)
    aggr_df['Dataset'] = dataset_name
    metrics_df['Dataset'] = dataset_name
    aggr_df.to_csv(f'results/{dataset_name}_{model_type}_{s}.csv')
    return metrics_df, aggr_df

def boxplot(df, dataset, model_type, metric='f1-score', save=False, **kwargs):
    plt.figure(figsize=(5, 3), dpi=400)
    sns.boxplot(data=df.sort_values(by='model'),
                x='model', y='f1-score',
                width=0.3, **kwargs)
    plt.grid(axis='y')
    plt.xlabel('')
    sns.despine()
    plt.title(f'{model_type.capitalize()} models CV {metric}s on {dataset.upper()}')
    plt.xticks(rotation=15)
    if save:
        plt.savefig(f'figs/{dataset}_{model_type}_boxplot.eps',
                    format='eps', bbox_inches='tight')
    plt.show()


def cross_dataset_eval(simulated, real):
    new_freq = get_freq(real) # resample to the same frequency
    summary = []
    # Train on the real dataset first
    X_train_r, X_test_r, y_train_r, y_test_r = utils.train_test_subjects_split(real, show_test=True)
    print(f'\n<----- {get_dataset_name(real)} > {get_dataset_name(real)} ----->')
    train_on_real, _ = run_models(X_train_r, y_train_r, X_test_r,
                                  y_test_r, type='ts', freq=new_freq)
    train_on_real['trainset'] = get_dataset_name(real)
    summary.append(train_on_real)

    # Create lists to hold simulated training sets
    X_train_s = []
    y_train_s = []
    for d in simulated:
        print(f'\n\n<----- {get_dataset_name(d)} > {get_dataset_name(real)} ----->')
        X_train_d, _, y_train_d, _ = utils.train_test_subjects_split(
            d, clip=True, new_freq=new_freq)
        X_train_s.append(X_train_d)
        y_train_s.append(y_train_d)
        cross_df, _ = run_models(
            X_train_d, y_train_d, X_test_r, y_test_r, freq=new_freq, type='ts')
        cross_df['trainset'] = get_dataset_name(d)
        summary.append(cross_df)

        # Mix and train_test_split
        X_train_m = np.concatenate([X_train_d, X_train_r], axis=0)
        y_train_m = np.concatenate([y_train_d, y_train_r], axis=0)

        print(f'\n\n<----- {get_dataset_name(d)} + {get_dataset_name(real)} ----->')
        mixed_df, _ = run_models(
            X_train_m, y_train_m, X_test_r, y_test_r, freq=new_freq, type='ts')
        mixed_df['trainset'] = f'{get_dataset_name(d)}+'
        summary.append(mixed_df)
    
    # Join all and train on real
    X_train_s.append(X_train_r)
    y_train_s.append(y_train_r)
    X_train_all = np.concatenate(X_train_s, axis=0)
    y_train_all = np.concatenate(y_train_s, axis=0)

    print(f'\n\n<----- Combining all training sets from all datasets ----->')
    all_df, _ = run_models(
        X_train_all, y_train_all, X_test_r, y_test_r, freq=new_freq, type='ts')
    all_df['trainset'] = 'All'
    summary.append(all_df)

    return pd.concat(summary, ignore_index=True)


def get_sample_attributions(clf, X_test, y_test, c=28, normalise=True, n=2):
    y_pred = clf.predict(X_test)
    true_falls = np.logical_and(y_test==1, y_pred==1)
    false_falls = np.logical_and(y_test==0, y_pred==1)
    true_adls = np.logical_and(y_test==0, y_pred==0)
    false_adls = np.logical_and(y_test==1, y_pred==0)
    tp_exp = explain_model(clf, X_test[true_falls][:n],
                           y_test[true_falls][:n], chunks=c,
                           normalise=normalise)
    print(X_test[false_adls][:n].shape)
    fp_exp = explain_model(clf, X_test[false_falls][:n],
                           y_test[false_falls][:n], chunks=c,
                           normalise=normalise)
    tn_exp = explain_model(clf, X_test[true_adls][:n],
                            y_test[true_adls][:n], chunks=c,
                            normalise=normalise)
    fn_exp = explain_model(clf, X_test[false_adls][:n],
                            y_test[false_adls][:n], chunks=c,
                            normalise=normalise)
    tp = {'sample':X_test[true_falls][0], 'attr': tp_exp[0]}
    fp = {'sample':X_test[false_falls][0], 'attr': fp_exp[0]}
    tn = {'sample':X_test[true_adls][0], 'attr': tn_exp[0]}
    fn = {'sample':X_test[false_adls][0], 'attr': fn_exp[0]}

    return [tp, fp, tn, fn]

def scale_arr(arr):
    scaler = MinMaxScaler(feature_range=(-1,1))
    return scaler.fit_transform(arr.reshape(-1,1)).flatten()

def plot_sample_with_attributions(attr_dict):
    titles = ['True Fall', 'False Alarm', 'True ADL', 'Missed Fall']
    plt.rcParams.update({'font.size': 10})
    cmap = plt.get_cmap('coolwarm')
    attributions = copy.deepcopy(attr_dict)
    for i, (model_name, exps) in enumerate(attributions.items()):
        fig, axs = plt.subplots(2, 2, figsize=(10, 5), dpi=400,
                            sharey='row', sharex='col', layout='constrained')
        fig.suptitle(model_name)
        for e, exp in enumerate(exps):
            ax = axs.flat[e]
            ax.set_title(titles[e])
            y = scale_arr(exp['sample'])
            x = np.arange(len(y))
            c = exp['attr'].flatten()
            ax.plot(c, linestyle=':', label='attribution profile', alpha=0.35)
            # Normalize the color values
            norm = mcolors.Normalize(vmin=-1, vmax=1)
            for j in range(len(x)-1):
                ax.plot(x[j:j+2], y[j:j+2], color=cmap(norm(c[j])), linewidth=1.5, label='normalised sample' if j==0 else None)
            ticks_loc = ax.get_xticks().tolist()
            ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
            ax.set_xticklabels([i//100 for i in ticks_loc])
            # ax.grid(which='both', axis='x')     
        axs[1,1].legend()
        fig.supylabel('Attribution score')
        # Adding color bar to show the color scale
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = plt.axes((1.01, 0.05, 0.015, 0.92))
        plt.colorbar(sm, cax=cax)
        fig.supxlabel('Time in seconds')
        # sns.despine()
        plt.savefig(f'figs/{model_name}_explanation.pdf', bbox_inches='tight')
        plt.show()



def plot_confusion_matrices(results, y_train, y_test):
    """Plot confusion matrices for train and test sets"""
    n_models = len(results)
    fig, axes = plt.subplots(n_models, 2, figsize=(10, 5*n_models))
    
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, result) in enumerate(results.items()):
        # Training confusion matrix
        cm_train = confusion_matrix(y_train, result['train_pred'])
        sns.heatmap(cm_train, annot=True, fmt='d', ax=axes[idx, 0],
                   xticklabels=['ADL', 'Fall'], yticklabels=['ADL', 'Fall'])
        axes[idx, 0].set_title(f'{name} - Training Set')
        axes[idx, 0].set_xlabel('Predicted')
        axes[idx, 0].set_ylabel('Actual')
        
        # Test confusion matrix
        cm_test = confusion_matrix(y_test, result['test_pred'])
        sns.heatmap(cm_test, annot=True, fmt='d', ax=axes[idx, 1],
                   xticklabels=['ADL', 'Fall'], yticklabels=['ADL', 'Fall'])
        axes[idx, 1].set_title(f'{name} - Test Set')
        axes[idx, 1].set_xlabel('Predicted')
        axes[idx, 1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()

def plot_learning_curves(models, X_train, y_train, cv=5):
    """Plot learning curves to visualize overfitting"""
    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5))
    
    if len(models) == 1:
        axes = [axes]
    
    for idx, (name, model) in enumerate(models.items()):
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        axes[idx].plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        axes[idx].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        axes[idx].plot(train_sizes, val_mean, 'o-', color='red', label='Cross-validation score')
        axes[idx].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        axes[idx].set_title(f'{name} Learning Curve')
        axes[idx].set_xlabel('Training Set Size')
        axes[idx].set_ylabel('Accuracy Score')
        axes[idx].legend(loc='best')
        axes[idx].grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_feature_importance(results):
    """Analyze and plot feature importance"""
    fig, axes = plt.subplots(1, len(results), figsize=(8*len(results), 6))
    
    if len(results) == 1:
        axes = [axes]
    
    for idx, (name, result) in enumerate(results.items()):
        model = result['model']
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20 features
            
            axes[idx].bar(range(len(indices)), importances[indices])
            axes[idx].set_title(f'{name} - Top 20 Feature Importances')
            axes[idx].set_xlabel('Feature Index')
            axes[idx].set_ylabel('Importance')
            axes[idx].set_xticks(range(len(indices)))
            axes[idx].set_xticklabels(indices, rotation=45)
    
    plt.tight_layout()
    plt.show()

# Main execution
def full_model_analysis(dataset, df=None):
    """Complete analysis pipeline"""
    
    # 1. Train models and check overfitting
    print("=== TRAINING MODELS ===")
    results, (X_train, y_train, X_test, y_test) = train_and_analyze_models(dataset, df)
    
    # 2. Plot confusion matrices
    print("\n=== CONFUSION MATRICES ===")
    plot_confusion_matrices(results, y_train, y_test)
    
    # 3. Plot learning curves for overfitting analysis
    print("\n=== LEARNING CURVES ===")
    models_dict = {name: result['model'] for name, result in results.items()}
    plot_learning_curves(models_dict, X_train, y_train)
    
    # 4. Feature importance analysis
    print("\n=== FEATURE IMPORTANCE ===")
    analyze_feature_importance(results)
    
    # 5. Print detailed classification reports
    print("\n=== DETAILED CLASSIFICATION REPORTS ===")
    for name, result in results.items():
        print(f"\n{name} - Test Set Classification Report:")
        print(classification_report(y_test, result['test_pred'], target_names=['ADL', 'Fall']))
    
    return results
