#!/eos/home-h/hcrottel/new_env/bin/python
import pandas as pd
import numpy as np
import os
import json
import sys
import pdb
import tools

def read_json(bins):
    path = tools.analysis_path('scripts/bins')
    with open(f'{path}/BinsDefinition-{bins}.json', 'r') as file_:
        bins_ = json.load(file_)
    return bins_




###  LOAD LUMIMASK JSON 
def read_lumimask(name=None):
    path = tools.analysis_path('scripts/lumimask')
    if name: 
        jsonLumiMask=name
    else:
        jsonLumiMask = "Cert_314472-325175_13TeV_17SeptEarlyReReco2018ABC_PromptEraD_Collisions18_JSON.txt"
    
        if jsonLumiMask not in os.listdir(path):
            os.system(f'wget -O {os.path.join(path, jsonLumiMask)} https://cms-service-dqm.web.cern.ch/cms-service-dqm/CAF/certification/Collisions18/13TeV/ReReco/{jsonLumiMask}')

    with open(os.path.join(path, jsonLumiMask), 'r') as jl:
        mask = json.load(jl)
        LumiMask = dict() 
        for run in mask:
            LumiMask[int(run)] = mask[run]
    
    return LumiMask




def create_lumimask(df, name=None):
    LumiMask = read_lumimask(name)
    if LumiMask:
        good_runs = list(LumiMask.keys())
        print('procesed LumiMask')
        Good_Lumi = np.zeros(len(df), dtype=bool) #Define primero un array de Falses de la misma longitud que el dataframe
        index = df.index
        try:
            names = index.names 
        except Exception as e:
            print('Caught an Exception...')
            print(e)
            print('...Bye\n')
            names = ['False']
        if 'run' in names and 'luminosityBlock' in names:
            run_lumi = index.to_frame()[['run', 'luminosityBlock']]
        else:
            run_lumi = df[['run', 'luminosityBlock']]
        for i, (index, row) in enumerate(run_lumi.iterrows()): #Iterar por los eventos extrayend el `run` y `luminosityBlock`
            run =  row['run']
            lumi = row['luminosityBlock']
            if row['run'] in good_runs: # VERIFICAR QUE EL RUN ESTE EN EL LUMIMASK
                lumi_block = LumiMask[run] 
                good_event = np.zeros(len(lumi_block), dtype=bool) #ARRAY DEL MISMO TAMAÑO QUE EL LUMIMASK
                for j,block in enumerate(lumi_block):
                    if lumi>=block[0] and lumi<=block[1]: good_event[j]=True #SI EL LUMIBLOCK DEL EVENTO COINCIDE, ES UN BUEN EVENTO
                Good_Lumi[i] = np.any(good_event) #BASTA CON QUE EL LUMIBLOCK ESTE DENTRO DE UN RANGO DEL LUMIMASK 
            if i%int(len(df)/10)==0: print('\t',round(100*i/(int(len(df))),2), '%')
        print(len(df[Good_Lumi])/ len(df))
    df['LumiMask'] = Good_Lumi
    # - # - # CREATE LUMIMASK
    #pd.to_pickle(df, f'../joined-{Bins}/Complete.pkl')
    return df


def split_and_save(df, outputfile, bins):

    pd.to_pickle(df, os.path.join(outputfile, 'Complete.pkl'))
    for bin_ in bins:
        q2 = np.power(df.DiMuMass,2)
        if bin_!=str(11):
            mask = (q2>bins[bin_][0]) & (q2<=bins[bin_][1])
        else:
            mask0 = (q2>bins[bin_][0][0]) & (q2<=bins[bin_][0][1])
            mask1 = (q2>bins[bin_][1][0]) & (q2<=bins[bin_][1][1])
            mask2 = (q2>bins[bin_][2][0]) & (q2<=bins[bin_][2][1])
            mask = mask0 | mask1 | mask2

        bindf = df[mask]
        pd.to_pickle(bindf, os.path.join(outputfile, f'Bin_{bin_}.pkl')) 

           
        
def only_split(df, bins):
    #pdb.set_trace()
    if not type(bins)==dict:
        bins = read_json(bins) 
        
    data_dict=dict()
    data_dict['Complete'] = df
    
    if 'q2' in df:
        q2 = df['q2']
    else:
        q2 = np.power(df.DiMuMass,2)
        
    for bin_ in bins:
        range_ = bins[bin_]    
        
        if type(range_[0])!=list:
            mask = (q2>range_[0]) & (q2<=range_[1])
        else:
            mask0 = (q2>range_[0][0]) & (q2<=range_[0][1])
            mask1 = (q2>range_[1][0]) & (q2<=range_[1][1])
            mask2 = (q2>range_[2][0]) & (q2<=range_[2][1])
            mask = mask0 | mask1 | mask2

        bindf           = df[mask]
        data_dict[bin_] = bindf
        
    return data_dict

def get_q2_Bin(df, bins, q2Bin):
    if not type(bins)==dict:
        bins = read_json(bins) 
    if str(q2Bin) == 'Complete':
        return df
        
    #pdb.set_trace()
    if not type(bins)==dict:
        bins = read_json(bins) 

    
    if 'q2' in df:
        q2 = df['q2']
    else:
        q2 = np.power(df.DiMuMass,2)
     

    q2Bin_range = bins.get(str(q2Bin), None) 
    if not q2Bin_range:
        keys_list = "\n".join([k for k in bins.keys()])
        raise NotImplementedError(f'Bin: -> {q2Bin} <- not found in bins json. Json have these keys: {keys_list}')

    if len(q2Bin_range)==2:
        mask = (q2>bins[q2Bin][0]) & (q2<=bins[q2Bin][1])
    elif len(q2Bin_range)==3:
        mask0 = (q2>bins[q2Bin][0][0]) & (q2<=bins[q2Bin][0][1])
        mask1 = (q2>bins[q2Bin][1][0]) & (q2<=bins[q2Bin][1][1])
        mask2 = (q2>bins[q2Bin][2][0]) & (q2<=bins[q2Bin][2][1])
        mask = mask0 | mask1 | mask2
    else:
        raise NotImplementedError(f'Bin: {q2Bin} has the following range {q2Bin_range}. I only know to interpret one and three bins. Solution may be easy :)')

    return df[mask]
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
if __name__ == '__main__':

    import argparse
    parser    = argparse.ArgumentParser(description='Split files into different bins and apply lumi mask if Real Data')
    
    type_data = parser.add_mutually_exclusive_group(required=True)
    type_data.add_argument('--MC', 
                            action='store_true',
                            help  ='Set if MonteCarlo Dataset')
    type_data.add_argument('--RD',
                            action='store_true', 
                            help='Set if Real Data (Collision Data)')
	
    parser.add_argument('-b',
						action='store',
						default=None, 
                       help='ID for BinsDefinition.json')
    parser.add_argument('--inputfile',
                       action='store',
                       required=False, 
                       help='Path to read the pd.DataFrame')
    parser.add_argument('--inputdir',
                       action='store',
                       required=False, 
                       help='Path to read the pd.DataFrames')
    parser.add_argument('--outputfile',
                       action='store',
                       required=False, 
                       type=str,
                       default='-',
                       help='Parent directory to save the pkl, \
                        defaults to ./Skim{c}/')
    
    args = parser.parse_args() 
    
    
    
    
    
    
    log  = ''
    bins = read_json(args.b)
    
    if args.inputfile:
        data = pd.read_pickle(args.inputfile)
    elif args.inputdir:
        data = pd.DataFrame()
        for file in os.listdir(args.inputdir):
            if not file.endswith('.pkl'): continue
            temp = pd.read_pickle(os.path.join(args.inputdir,file))
            data = data.append(temp)
    
    
    if args.RD:
        data = create_lumimask(data)
        
    if args.outputfile == '-':
        outputfile = f'Skim{args.b}/'
        os.makedirs(outputfile, exist_ok=True)
    
    split_and_save(data, outputfile, bins)