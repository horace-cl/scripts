import ks_test
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score





def roc_curves_nominal(test, train, xgb_cut=0):
    """test and train must be dataframes with colums label and XGB. `label` is the true class of an event and XGB is the predict_proba value 
    """    


    test_roc  = roc_curve(test['label'],  test['XGB'])
    train_roc = roc_curve(train['label'], train['XGB'])
    test_auc = roc_auc_score(test['label'],  test['XGB'])
    train_auc= roc_auc_score(train['label'], train['XGB'])
    
    plt.figure(figsize=(12, 9))
    l = plt.plot(test_roc[0], test_roc[1], 
             ls='-', 
             label=f'Test {round(test_auc,4)}',
             linewidth=3)

    if xgb_cut:
        mask_ = test_roc[2]>=xgb_cut
        print(test_roc[0][mask_][-1], test_roc[1][mask_][-1])
        plt.scatter(test_roc[0][mask_][-1], 
                test_roc[1][mask_][-1], 
                s=250, marker='s')


    plt.plot(train_roc[0], train_roc[1], 
             ls='-', 
             label=f'Train {round(train_auc,4)}',
             linewidth=3)

    if xgb_cut:
        mask_ = train_roc[2]>=xgb_cut
        print(train_roc[0][mask_][-1], train_roc[1][mask_][-1])
        plt.scatter(train_roc[0][mask_][-1], 
                    train_roc[1][mask_][-1], 
                    s=200)

    x = np.linspace(0,1,1000)
    plt.plot(x, x, color='grey', ls='--', 
             label='Random choice')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(title='auc', frameon=True)
    plt.xscale('log')
    plt.grid(True)







def __params_to_string(model):
    model.get_params()
    string = ''
    lens = [len(k) for k in model.get_params()]
    max_len = max(lens)
    for k,v in model.get_params().items():
        string += f'{k} = {v}\n'

    return string

def plot_classifier_distributions(model, test, train, print_params=False, bins=20, figsize=[10,7], text_opts = dict()):
    """test and train must be dataframes with colums label and XGB. `label` is the true class of an event and XGB is the predict_proba value 
    """    

    test_background = test.query('label==0')['XGB']
    test_signal     = test.query('label==1')['XGB']
    train_background= train.query('label==0')['XGB']
    train_signal    = train.query('label==1')['XGB']

    test_pred = test['XGB']
    train_pred= train['XGB']

    density = True

    fig, ax = plt.subplots(figsize=figsize)

    background_color = 'red'

    opts = dict(
        range=[0,1],
        bins = bins,
        density = density 
    )
    histtype1 = dict(
        histtype='stepfilled',
        linewidth=3,
        alpha=0.45,
    )

    ax.hist(train_background, **opts, **histtype1, 
             facecolor=background_color, 
             edgecolor=background_color,
             zorder=0)
    ax.hist(train_signal, **opts, **histtype1, 
             facecolor='blue', 
             edgecolor='blue',
             zorder=1000)






    hist_test_0 = np.histogram(test_background, **opts)
    hist_test_1 = np.histogram(test_signal, **opts)
    bins_mean = (hist_test_0[1][1:]+hist_test_0[1][:-1])/2
    bin_width = bins_mean[1]-bins_mean[0]
    area0 = bin_width*np.sum(test.label==0)
    area1 = bin_width*np.sum(test.label==1)

    opts2 = dict(
          capsize=3,
          ls='none',
          marker='o'
    )


    ax.errorbar(bins_mean, hist_test_0[0],  yerr = np.sqrt(hist_test_0[0]/area0), xerr=bin_width/2,
                 color=background_color, **opts2, zorder=100)
    ax.errorbar(bins_mean, hist_test_1[0],  yerr = np.sqrt(hist_test_1[0]/area1), xerr=bin_width/2,
                 color='blue', **opts2, zorder=10000)




    _ks_back = ks_test.ks_2samp_sci(train_background, test_background)[1]
    _ks_sign = ks_test.ks_2samp_sci(train_signal, test_signal)[1]

    print('Own ks test\n',
          ks_test.ks_2samp_weighted(train_background, test_background)[1],
          ks_test.ks_2samp_weighted(train_signal, test_signal)[1], sep='\n\t')
    
    auc_test  = roc_auc_score(test.label,test_pred )
    auc_train = roc_auc_score(train.label,train_pred)
    legend_elements = [Patch(facecolor='black', edgecolor='black', alpha=0.4,
                             label=f'Train (auc) : {round(auc_train,4)}'),
                      Line2D([0], [0], marker='|', color='black', 
                             label=f'Test (auc) : {round(auc_test,4)}',
                              markersize=25, linewidth=1),
                       Circle((0.5, 0.5), radius=2, color='red',
                              label=f'Background (ks-pval) : {round(_ks_back,4)}',),
                       Circle((0.5, 0.5), 0.01, color='blue',
                              label=f'Signal (ks-pval) : {round(_ks_sign,4)}',),
                       ]

    ax.legend(
              #title='KS test',
              handles=legend_elements,
              #bbox_to_anchor=(0., 1.02, 1., .102),
              loc='upper center',
              ncol=2, 
              #mode="expand", 
              #borderaxespad=0.,
              frameon=True,
              fontsize=15)
    
    if print_params:
        x = text_opts.get('x', 1.02); text_opts.pop('x', None)
        y = text_opts.get('y', 1.02); text_opts.pop('y', None)
        fontsize=  text_opts.get('fontsize', 12); text_opts.pop('fontsize', None)
        
        ax.text(x, y, __params_to_string(model), 
            transform=ax.transAxes, 
            fontsize=fontsize, ha='left', va='top', **text_opts)

    ax.set_yscale('log')
    ax.set_xlabel('XGB output')
    #ax.set_ylim(0.005, 100)

    #plt.savefig(os.path.join(dir_, 'LR_overtrain.pdf'), bbox_inches='tight')

    del test_background, test_signal
    del train_background, train_signal
    del test_pred, train_pred

    return fig, ax