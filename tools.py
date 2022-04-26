import pandas as pd
import json
import os
import re
from pprint import pprint


path_to_latex = '/Users/horace/Documents/projects/CMS/LaTex/AN-21-020/Figures'
hesse_names   = ['minuit_hesse', 'hesse_np']


def read_json(json_path):
    with open(json_path, 'r') as js:
        return json.load(js)

def create_params_mass_fit(minimum, pdf):
    out_dict = dict()
    for param in pdf.get_params():
        out_dict[param.name] = dict(value=param.value().numpy())
        if param.name in minimum.params:
            result  = minimum.params[param]
            key_err = [k for k in hesse_names if k in result]
            if key_err:  
                out_dict[param.name]['err'] = result['minuit_hesse']['error']



def create_params_dict_u(minimum, pdf):
    out_dict = dict()
    for name, param in pdf.params.items():
        result = minimum.params[param]
        out_dict[name] = dict(value=result['value'], 
                              hesse=result['minuit_hesse']['error'])
    return out_dict


def crate_json(minimum, pdf, output_dir, name):
    out_dict = create_params_dict_u(minimum, pdf)
    if not name.endswith('.json'): name+='.json'
    with open(os.path.join(output_dir, name), 'w+') as jj:
        json.dump(out_dict, jj, indent=4)

        
        
        
        
        
# def create_params_dict_composed(minimum, pdf):
#     out_dict = dict()
    
#     for param,result in minimum.params.items():
#         key = 'minuit_hesse' if 'minuit_hesse' in result else 'hesse_np'
#         try:
#             out_dict[param.name] = dict(value=result['value'], 
#                               hesse=result[key]['error'])
#         except KeyError:
#             out_dict[param.name] = dict(value=result['value'])

#     for param in pdf.get_params():
#         if param.name in out_dict: continue
#         out_dict[param.name] = dict(value=param.value().numpy())        

#     try:
#         cov = minimum.covariance().tolist()
#         out_dict['covariance'] = cov
#         out_dict['covariance_params'] = [p.name for p in minimum.params]
#     except Exception as e:
#         print(e)
        
#     return out_dict

        
def create_params_dict_composed(minimum, pdf, substring_minimum='', substring_pdf=''):
    out_dict = dict()

    for param in pdf.get_params():
        fitted = False
        #Remove any unwanted string from parameter name for good matching
        param_name_clean = param.name.replace(substring_pdf, '')
        for param_min, result in minimum.params.items():
            key = 'minuit_hesse' if 'minuit_hesse' in result else 'hesse_np'
            if param_name_clean in param_min.name:
                out_dict[param_name_clean] = dict(value=result['value'], 
                                                  hesse=result[key]['error'])
                fitted = True
        if not fitted: out_dict[param_name_clean] = dict(value=param.value().numpy())

    try:
        cov = minimum.covariance().tolist()
        out_dict['covariance'] = cov
        out_dict['covariance_params'] = [p.name.replace(substring_minimum, '') for p in minimum.params]
    except Exception as e:
        print(e)
        
    return out_dict

def create_json_composed(minimum, pdf, output_dir, name, substring_minimum='', substring_pdf=''):
    return crate_json_composed(minimum, pdf, output_dir, name, substring_minimum, substring_pdf)

def crate_json_composed(minimum, pdf, output_dir, name, substring_minimum='', substring_pdf=''):
    out_dict = create_params_dict_composed(minimum, pdf, substring_minimum, substring_pdf)
    
    if not name.endswith('.json'): name+='.json'
    with open(os.path.join(output_dir, name), 'w+') as jj:
        json.dump(out_dict, jj, indent=4)
    return out_dict
        

        
        
def create_params_dict_polys(minimum, pdf):
    out_dict = dict()
    all_coefs = [c for c in pdf.params.values()]
    out_dict['coefs'] = list()
    out_dict['hesse'] = list()
    for coef in all_coefs:
        #for param, result in minimum.params.items():
        if coef in minimum.params:
            result = minimum.params[coef]
            out_dict['coefs'].append(result['value'])
            out_dict['hesse'].append(result['minuit_hesse']['error'])
        else:
            out_dict['coefs'].append(0)
            out_dict['hesse'].append(-1)
    if all([c==0 for c in out_dict['coefs']]): 
        out_dict = create_params_dict_polys_from_min(minimum, pdf)
    out_dict['covariance'] = list(minimum.covariance().tolist())
    return out_dict


def create_params_dict_polys_from_min(minimum, pdf):
    out_dict = dict()
    degree = pdf.degree
    
    params_dict_values = dict()
    for k in minimum.params.keys():
        params_dict_values[k.name] = k
    
    out_dict['coefs'] = list()
    out_dict['hesse'] = list()

    for indx in range(degree+1):
        param = params_dict_values.get(f'c^{indx}_{degree}', 'none')
        if param=='none':
            out_dict['coefs'].append(0)
            out_dict['hesse'].append(-1)
        else:
            result = minimum.params[param]
            out_dict['coefs'].append(result['value'])
            out_dict['hesse'].append(result['minuit_hesse']['error'])
    return out_dict
        
    
def crate_json_polys(minimum, pdf, output_dir, name):
    out_dict = create_params_dict_polys(minimum, pdf)
    if not name.endswith('.json'): name+='.json'
    with open(os.path.join(output_dir, name), 'w+') as jj:
        json.dump(out_dict, jj, indent=4)




def find_data_path(RD=True):
    
    root = 'DataSelection/DataSets' if RD else 'DataSelection/MonteCarlo'
    root = analysis_path(root)
    file_indx=1
    while file_indx>=0:
        print('Which dir?\n')
        files = os.listdir(root)
        for i,f in enumerate(files):
            print(f'{i} - {f}')
        print('-1 - Here')
        file_indx = int(input('Enter a number'))
        if file_indx>=0:
            root = os.path.join(root, files[file_indx])
    print(root)
    


def analysis_path(path):
    if 'HOMEANALYSIS' not in os.environ:
        try:
            HOME = os.environ['CERNBOX_HOME']
        except KeyError:
            print('Make sure to export `HOMEANALYSIS` in bash if not in SWAN')
            raise KeyError
    else:
        HOME = os.environ['HOMEANALYSIS']
    
    if HOME in path: 
        return path
    
    if path[0]=='/': path = path[1:]
    return os.path.join(HOME, path)



def read_data(path, RD = True):
    
    Bins = dict()
    temp = pd.read_pickle(os.path.join(path, 'Complete.pkl'))
    if RD:
        mass_mask = (temp.BMass>=5.0) & (temp.BMass<=5.7) & (temp.LumiMask)
    else:
        mass_mask = (temp.BMass>=5.0) & (temp.BMass<=5.7) & temp.GENCand
    Bins['Complete'] = temp[mass_mask]
    
    for i in range(-1,12):
        if i in [3,5]: continue
        temp = pd.read_pickle(os.path.join(path, f'Bin_{i}.pkl'))
        if RD:
            mass_mask = (temp.BMass>=5.0) & (temp.BMass<=5.7)  & (temp.LumiMask)
        else:
            mass_mask = (temp.BMass>=5.0) & (temp.BMass<=5.7)  & (temp.GENCand)
        Bins[i] = temp[mass_mask]
        
    return Bins



def read_params(path):
    params_dict = dict()
    #path = '../../Params/OneBackComp'
    for params in os.listdir(path):
        if not params.endswith('.json'): continue
        with open(os.path.join(path, params), 'r') as f:
            if 'Comp' in params:
                #Bin_ = params.replace('Bin', '').replace('.json', '')
                Bin_ = 'Complete'
            else:    
                number = re.findall(r'[-+]?\d+', params)
                if number:
                    Bin_ = int(number[0])
                else:
                    Bin_ = int(params.replace('Bin', '').replace('.json', ''))
            params_dict[Bin_] = json.load(f)
            
    return params_dict






#FOR ZFIT
def init_params_c(model, family, c=0.1):
    for param in model.get_params():
        if not param.floating:
            continue
        elif family=='chebyshev' and 'c_0' in param.name:
            params.set_value(1)
        else: 
            param.set_value(c)
        
        
def find_param(model, name):
    for par in model.get_params(): 
        if par.name==name: return par
        
        
def find_param_substring(model, substring):
    for par in model.get_params(): 
        if substring in par.name: return par
        
        
def find_params_substrinsg(model, substrings):
    params = list()
    for subs in substrings:
        p = find_param_substring(model, subs)
        params.append(p)
    return params




def create_params_dict(model):
    diction = dict()
    for key, val in model.params.items():
        diction[key] = val.value().numpy()
    return diction


def create_lumi_json_from_df(df_indexed):
    lumi_json = dict()
    df_index = df_indexed.sort_index().index.to_frame()
    
    runs = [int(r) for r in set(df_index.run)]
    runs.sort()
    
    for run in runs:
        df_run = df_index.query(f'run=={run}')
        lumis = [int(l) for l in set(df_run.luminosityBlock)]
        
        lumi_json[run]=consecutiveRanges(lumis)
    return lumi_json

def consecutiveRanges(numbers):
    #https://www.geeksforgeeks.org/find-all-ranges-of-consecutive-numbers-from-array/
    a = list(set(numbers))
    a.sort()
    length = 1
    ranges = []
    n = len(a)
    # If the array is empty,
    # return the list
    if (n == 0):
        return ranges
     
    # Traverse the array
    # from first position
    for i in range (1, n + 1):

        # Check the difference
        # between the current
        # and the previous elements
        # If the difference doesn't
        # equal to 1 just increment
        # the length variable.
        if (i == n or a[i] -
            a[i - 1] != 1):
        
            # If the range contains
            # only one element.
            # add it into the list.
            if (length == 1):
                ranges.append([a[i - length], a[i - length]])
            else:

                # Build the range between the first
                # element of the range and the
                # current previous element as the
                # last range.
                temp = (str(a[i - length]) +
                        " -> " + str(a[i - 1]))
                #ranges.append(temp)
                ranges.append([a[i - length], a[i - 1]])

            # After finding the 
            # first range initialize
            # the length by 1 to
            # build the next range.
            length = 1
        
        else:
            length += 1
    return ranges