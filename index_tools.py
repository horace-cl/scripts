import uproot3 as uproot
import pandas as pd
import numpy as np
from flatten_NanoAODs import expand_simple_branches, trg_branches
from termcolor import colored, cprint
#import pdb
#from IPython.core.debugger import set_trace

b_branches = [
    'BToKMuMu_fit_pt', 
    'BToKMuMu_fit_eta', 
    'BToKMuMu_fit_phi', 
    'BToKMuMu_fit_mass',
    'BToKMuMu_PDL',  
    'BToKMuMu_ePDL', 
    'BToKMuMu_cosThetaKMu',
    'BToKMuMu_svprob',
    'BToKMuMu_*significance',
    'BToKMuMu_cosAlpha', 
    'BToKMuMu_fit_cos2D',
    'BToKMuMu_pv_dz_trgmuon',
    'BToKMuMu_pv_index',

    #DiMuon's variables:
    'BToKMuMu_mll_llfit',
    'BToKMuMu_mllErr_llfit',
    'BToKMuMu_mll_pt',
    
    #K's variables:
    'BToKMuMu_fit_k_pt', 
    'BToKMuMu_fit_k_phi', 
    'BToKMuMu_fit_k_eta', 
    #'BToKMuMu_k_charge',
    #'BToKMuMu_k_nValidHits', 
    #'BToKMuMu_k_lostInnerHits', 
    #'BToKMuMu_k_matchLooseMuon',
    #'BToKMuMu_k_matchMediumMuon',
    #'BToKMuMu_k_matchMuon',
    #'BToKMuMu_k_matchSoftMuon',
    #'BToKMuMu_k_min_dr_trk_muon', 
    #'BToKMuMu_k_min_dr_trk_muon_',
    #'BToKMuMu_k_min_pt_trk_muon',
    #'BToKMuMu_k_HighPurity',
    #'BToKMuMu_k_numberOfPixelHits',
    #'BToKMuMu_k_numberOfHits',
    'BToKMuMu_k*',
    
    #Muons variables:, 
    'BToKMuMu_fit_l1_pt' , 'BToKMuMu_fit_l2_pt',
    'BToKMuMu_fit_l1_eta', 'BToKMuMu_fit_l2_eta',
    'BToKMuMu_fit_l1_phi', 'BToKMuMu_fit_l2_phi',
    'BToKMuMu_l1_charge' , 'BToKMuMu_l2_charge',
    'BToKMuMu_mu1_IP_sig', 'BToKMuMu_mu2_IP_sig',
    'BToKMuMu_mu1_isSoft', 'BToKMuMu_mu2_isSoft',
    'BToKMuMu_mu1_pt'    , 'BToKMuMu_mu2_pt',
    'BToKMuMu_mu1_eta'    , 'BToKMuMu_mu2_eta',
    'BToKMuMu_mu1_isTriggering', 'BToKMuMu_mu2_isTriggering',
             ]
#class tables():



old_naming_dict = {
    'BToKMuMu_fit_mass':'BMass',
    'fit_pt'   : 'Bpt',
    'svprob'   : 'prob',
    'fit_mass' : 'BMass',
    'mll_llfit': 'DiMuMass',
    'fit_k_pt' : 'kpt',
    'cosAlpha' : 'cosA',
    'lxy_significance' : 'signLxy',
    'lxy_pv_significance' : 'signLxy',
    'fit_l1_pt' : 'l1pt',
    'fit_l2_pt' : 'l2pt',
}
    
    
    
    

def cands_per_event(df):
    return df.reset_index('subentry').index.value_counts()

    
    
    
def create_all_tables(file, indexes, subentry=True):
    tables = dict()
    
    headers, counts = np.unique([k.decode("utf-8") .split('_')[0] for k in file.keys()], return_counts=True)
    
    for h,c in zip(headers, counts):
        if c<2: continue
        tables[h] = create_table(file, indexes, h+'*', subentry=True)
    
    return tables





def create_table(file, indexes, regex, subentry=True, remove_sufix=False, old_naming=False, mu_branches=[]):
    if regex.endswith('*'): regex = regex[:-1]
    table = file.arrays(indexes+[regex+'*'], outputtype=pd.DataFrame, flatten=True)
    if subentry and 'subentry' in table.index.names:
        table = table.reset_index(level='subentry')
        table = table.set_index(indexes+['subentry'])
    if remove_sufix:
        table.columns = [c.replace(regex+'_', '') for c in table.columns]
    if old_naming and remove_sufix:
        table.columns = [old_naming_dict.get(c, c) for c in table.columns]
    if mu_branches:
        mu1_indx = file.array('BToKMuMu_l1Idx')
        mu2_indx = file.array('BToKMuMu_l2Idx')
        mu_branches_clean = [b if b.startswith('Muon_') else 'Muon_'+b for b in mu_branches]
        muon_table = file.arrays(mu_branches_clean, namedecode='utf-8')
        reshaped_table = dict()
        for var, awk in muon_table.items():
            reshaped_table[var.replace('Muon', 'Muon1')] = awk[mu1_indx].flatten()
            reshaped_table[var.replace('Muon', 'Muon2')] = awk[mu2_indx].flatten()
        
        try:
            reshaped_table = pd.DataFrame.from_dict(reshaped_table)
            reshaped_table.index = table.index
            table = table.join(reshaped_table)
        except Exception as e:
            print(e)
            print('Muon Table not compatible with B Table???')
            print('Muon Table:\n')
            print(reshaped_table.head(), '\n\n\n')
            
            print('B Table:\n')
            print(table.head())
    return table





def create_skim(file,
    indexes=['run', 'luminosityBlock', 'event'],
    regex='BToKMuMu',
    isMC=False,
    subentry=True, 
    branches   = b_branches,
    mu_branches= [],
    old_naming=True, 
    trigTables=True, 
    softMuons=True, 
    verbose=True,
    run=None):
    """Make one single table from each NanoAOD file with information of B tables and possibly Trigger Tables.
    Any other matching must be implemented (e.g. Muons Tables)

    Parameters
    ----------
    file : uproot3.rootio.TTree
        The main branch where all Tables are stored
    indexes : list
        Name of the branches that will be used as index (default = ['run', 'luminosityBlock', 'event'])
    regex : str
        Regex of the tables that are wantes. Default = 'BToKMuMu'
    isMC : bool
        If data set is MonteCarlo (MC) a matching attempt will be done usign the information from the GenPart tables 
    subentry : bool
        To be set if in addition of the selected indexes, you want a counter for each candidate. The final index will be:
         [run, lumi, event, subentry]
    branches : list, iterable
        List of branches to be read from file.
    mu_branches : list, iterable
        List of branches to be read from the muon tables.
    old_naming : bool
        Map names from previous iterations of the analysis, the mapping is defined at the beginning of the file: `old_naming_dict`
    trigTables : bool
        If True, information from HLT, L1 and TrigObj tables is added
    softMuons : bool
        If True, only events where both muons are Soft, are considered.
    run : int
        If isMC, you can change the run number in order to maintain information regarding possible different MC generations
    verbose : bool
        If True, print additional inofrmation from ROOT files
    Returns
    -------
    pd.DataFrame
        A data frame with multiindex and all the columns defined with the arguments
    """

    #TODO: CHECK THAT BRANCHES CONTAIN REGEX AND ARE INSIDE FILE
    table = file.arrays(indexes+branches, outputtype=pd.DataFrame, flatten=True,)
    table.columns = [c.replace(regex+'_', '') for c in table.columns]
    
    if mu_branches:
        mu1_indx = file.array('BToKMuMu_l1Idx')
        mu2_indx = file.array('BToKMuMu_l2Idx')
        mu_branches_clean = [b if b.startswith('Muon_') else 'Muon_'+b for b in mu_branches]
        muon_table = file.arrays(mu_branches_clean, namedecode='utf-8')
        reshaped_table = dict()
        for var, awk in muon_table.items():
            reshaped_table[var.replace('Muon', 'Muon1')] = awk[mu1_indx].flatten()
            reshaped_table[var.replace('Muon', 'Muon2')] = awk[mu2_indx].flatten()
        
        try:
            reshaped_table = pd.DataFrame.from_dict(reshaped_table)
            reshaped_table.index = table.index
            table = table.join(reshaped_table)
        except Exception as e:
            print(e)
            print('Muon Table not compatible with B Table???')
            print('Muon Table:\n')
            print(reshaped_table.head(), '\n\n\n')
            
            print('B Table:\n')
            print(table.head())
    
    if old_naming: 
        #print(old_naming_dict)
        table.columns = [old_naming_dict.get(c, c) for c in table.columns]
        #prob_ = [b for b in branches if 'ass' in b]
        #print(f'---> {prob_}')
        
    subentry_index = subentry and 'subentry' in table.index.names
    if subentry_index:
        table = table.reset_index(level='subentry')
        table = table.set_index(indexes+['subentry'])
    else:
        table = table.set_index(indexes)
        
    if isMC:
        gen_mask = create_GEN_cand_mask(file, 
                            resonance=isMC if isMC in ['JPSI', 'PSI2S'] else None,
                            report=verbose)
        gen_mask = pd.Series(gen_mask.flatten(), index=table.index)
        table['GENCand'] = gen_mask
        if run:
            table = table.reset_index()    
            table.run = run
            if subentry_index: table = table.set_index(indexes+['subentry'])
            else : table = table.set_index(indexes)
        
    if trigTables:
        if type(trigTables)==str:
            trg_branches_ = [t for t in trg_branches if trigTables in t]
        else:
            trg_branches_ = trg_branches
        trg = pd.DataFrame(expand_simple_branches(file, trg_branches_, file.array('nBToKMuMu')), index=table.index)
        table = table.join(trg)

    if softMuons:
        ini = len(table) 
        if verbose: print('Before Soft Muons : ', ini)
        table = table[table.mu1_isSoft + table.mu2_isSoft == 2]
        if verbose: print(f'After Soft  Muons : {len(table)}  ({round(len(table)/ini, 3)})')

    return table



def create_GEN_cand_mask(file, resonance=None, report=True):
    
    ### INDICES DE KAONES Y MUONES INVOLUCRADOS
    ### EN EL CANDIDATO
    kIdx  = file.array('BToKMuMu_kIdx')
    m1Idx = file.array('BToKMuMu_l1Idx')
    m2Idx = file.array('BToKMuMu_l2Idx')
    
    
    ##Collection ::: GenPart	interesting gen particles 
    ## https://cms-nanoaod-integration.web.cern.ch/integration/master/mc94X_doc.html#LHE
    GenpdgID = file.array('GenPart_pdgId')
    IdxMother = file.array('GenPart_genPartIdxMother')
    Muon_genPartIdx = file.array('Muon_genPartIdx')
    Track_genPartIdx = file.array('ProbeTracks_genPartIdx')
    
    #MUONS AND TRACKS MUST BE A GEN PARTICLE (WHO HAS A ANCESTOR)
    onlyGENparticles = (Track_genPartIdx[kIdx]!=-1) & \
              (Muon_genPartIdx[m1Idx]!=-1) & \
              (Muon_genPartIdx[m2Idx]!=-1)
    
    #FIRST CHECK IF THE TWO MUONS HAVE GENPDGID = +-13
    isdimuonSystem = GenpdgID[Muon_genPartIdx[m1Idx]]*GenpdgID[Muon_genPartIdx[m2Idx]]==(-13*13)
    #TRACK MUST BE MATCHED TO A GEN  KAON ID = +-321
    trackisKaon = abs(GenpdgID[Track_genPartIdx[kIdx]])==321

    #MUONS MUST HAVE SAME MOTHER
    mu1MotherIdx = IdxMother[Muon_genPartIdx[m1Idx]]
    mu2MotherIdx = IdxMother[Muon_genPartIdx[m2Idx]]
    dimuonSystem_sameMother = mu1MotherIdx == mu2MotherIdx
    
    #THE MOTHER OF A MUON MUST BE A Psi(2S), JPsi, or the B+
    dimuon=0
    if resonance:
        if resonance.lower()=='jpsi':
            dimuon = 443
        elif resonance.lower() in ['psi2s', 'psiprime', 'pisp']:
            dimuon = 100443
        else:
            raise NotImplementedError('POSSIBLE DIMUON SYSTEMS: `jpsi`  `psi2s`')
    else:
        dimuon = 521
    dimuonSystem = abs(GenpdgID[mu1MotherIdx])==521
    
    
    #THE GREATMOTHER (IF THERE IS)OF A MUON MUST BE THE SAME AS THE MOTHER 
    #                                                          OF THE KAON
    if resonance:
        dimuon_kaon_same_Mother = IdxMother[mu1MotherIdx]==IdxMother[Track_genPartIdx[kIdx]]
    else:
        dimuon_kaon_same_Mother = mu1MotherIdx==IdxMother[Track_genPartIdx[kIdx]]
    
    #THE GREATMOTHER OF ANY FINAL STATE PARTICLE MUST BE A bu(B^+-)
    is_GEN_B = abs(GenpdgID[IdxMother[Track_genPartIdx[kIdx]]])==521
    
    #DOES THE B CANDIDATE HAVE ANCESTORS?
    main_B = IdxMother[IdxMother[Track_genPartIdx[kIdx]]]==-1
    
    #A TRUE CANDIDATE MUST SATISFY ALL OF THESE:
    # onlyGENparticles             ----->  PARTICLES MATCHED TO A GEN PARTICLE?
    # isdimuonSystem & trackisKaon ----->  PARTICLES MATCHED TO MUONS(+-) AND KAON?
    # dimuonSystem_sameMother      ----->  BOTH MUONS HAVE THE SAME MOTHER?
    # dimuon_kaon_same_Mother      ----->  DIMUON SYSTEM AND KAON, SAME MOTHER?
    # is_GEN_B & main_B            ----->  IS THE MOTHER OF DIMUON AND KAON A B+-?  THE B^+ HAS ANCESTORS?
    GENCandidate = (onlyGENparticles & isdimuonSystem & trackisKaon & \
                     dimuonSystem_sameMother &  dimuon_kaon_same_Mother & is_GEN_B & main_B )
    if report:
        #HOW MANY CANDIDATES PER EVENT PASSED THE GENCandidate Mask???
        BMass = file.array('BToKMuMu_fit_mass')
        cands_event = BMass[GENCandidate].count()
        print(f'\n\tNumber of Gen Candidates : {np.sum(cands_event)}')
        print(f'\n\tNumber of Events         : {len(BMass)}')
        print(f'\n\tNumber of Candidates     : {np.sum(BMass.count())}')
        
        if any(cands_event>1):     
            cprint(' ----- WARNING\nMore than one candidate per event  -----', 'red', file=sys.stderr)
    
    return GENCandidate




def change_repeated_index(df, verbose=False):
    
    #To check that indexes are not repeated, if so, there was a bad selection of 
    #Events/lumiblock in MC generation
    #This function moves the event number to the next one for each repeated candidate
    cands_ = df.index.value_counts()
    if np.all(cands_==1): return df
    
    clean, repeated = df.drop(cands_[cands_!=1].index), df.drop(cands_[cands_==1].index)
    repeated = repeated.reset_index()
    if verbose:
        print(repeated)
    for i in range(len(repeated)): 
        repeated.at[i, 'event'] += i
    repeated = repeated.set_index(['run','luminosityBlock','event','subentry'])
    clean = clean.append(repeated)
    if len(clean)!=len(df):
        raise ValueError('Cleaned and original df have changed sizes!\n ¿Is pandas v==1.3.2?')
    
    return change_repeated_index(clean, verbose=False)



def select_cand(df, var, LumiMask=True, verbose=False):
    
    #Check if there are no repeated indexes
    df = change_repeated_index(df, verbose=verbose)
    
    #Create the pd.Series of the number of candidates per event
    if LumiMask:    cands_event = cands_per_event(df[df.LumiMask])
    else:           cands_event = cands_per_event(df)
    
    #Sort DataFrame 
    if LumiMask:    df_sorted = df[df.LumiMask].sort_index()
    else:           df_sorted = df.sort_index()
    
    #Columns to print if verbose
    columns = ['mu1_pt', 'mu2_pt', 'kpt', 'cosA', 'prob', 'BMass']
    if 'L_XGB' in df: columns+=['L_XGB']
    
    #Dataframe to be filled with one candidate per event only
    one_cand_per_event = pd.DataFrame()
    
    #Lists to check if candidates "obviously" share muons
    same_muon1 = list()
    same_muon2 = list()
    
    #Counter to stop iterating if the events have only one candidate
    cc = 0
    
    #Iterate over all events 
    for i_ , (indx, cands) in enumerate(cands_event.iteritems()):
        # Break if the number of candidates is equal than one
        # The events are ordered from higher to lower number of candidates
        if cands<=1: break
        #We need 
        cc += cands
        # Get dataframe for all candidates in the event
        EVTS = df_sorted.loc[indx]
        
        # Check if the candidates share the same muon
        same_muon1.append(np.all(EVTS.mu1_pt==EVTS.mu1_pt.iloc[0]))
        same_muon2.append(np.all(EVTS.mu2_pt==EVTS.mu2_pt.iloc[0]))
        # If they do not have the same muon in leading and trailing:
        # print some info
        if not(same_muon1[-1] or same_muon2[-1]) and verbose:
            print(EVTS[columns].to_markdown())
            print('\n')
            
        #################################################
        #Select best canidadte based on the highest var #
        #################################################
        EVTS = EVTS[EVTS[var]==np.max(EVTS[var])]
        # Create new DF composed of only one candidate
        EVTS = EVTS.reset_index()
        EVTS['run'] = int(indx[0])
        EVTS['luminosityBlock'] = int(indx[1])
        EVTS['event'] = int(indx[2])
        # Set index information: Run, Lumi, Event
        EVTS = EVTS.set_index(['run', 'luminosityBlock', 'event', 'subentry'])
        #Append Information
        one_cand_per_event = one_cand_per_event.append(EVTS)

    if verbose:
        print('\n\n\n--------- x --------- x --------- x ---------')
        print(len(one_cand_per_event))
    
    # Create the final df following these steps:
    # Drop all events that have only one candidate per event
    # Use the dropped DF to append it to the one_cand_per_event DF
    df_sorted = one_cand_per_event.append(df_sorted.drop(cands_event.index[:i_]) )  
    
    if verbose:
        print(len(one_cand_per_event))

    return df_sorted




def dataset_binned(kind='RD', 
                   Bin ='ALL', 
                   cuts_json=12, 
                   bins_json=3, 
                   OneCand='prob',
                   path=None,
                   list_cuts = [
                            'resonance_rejection',
                            'anti_radiation_veto',
                            'Quality',
                            'XGB',
                            'missID',
                            'Bpt',
                            'l1pt', 
                            'l2pt', 
                            'k_min_dr_trk_muon'],
                   run=None,
                   verbose=False,
                   mu_branches=['HLT*'],
                   sample=None,
                   **kwargs
                  ):
    
    import tools
    import join_split
    import cuts
    from glob import glob
    
    cuts_json_ = cuts.read_cuts_json(cuts_json)
    bins_json_ = join_split.read_json(bins_json)
    
    paths = dict(RD=f'DataSelection/DataSets/CRAB_RE_AUG_2021/Skim9/Skim{bins_json}/Complete.pkl',
                 BSLL='DataSelection/NanoAOD/BTOSLL/Aug21/BParkNANO_mc_private_*.root',
                 PHSP='DataSelection/NanoAOD/PHSP/Aug21/BParkNANO_mc_private_*.root',
                 RDJPSI=f'DataSelection/DataSets/JPsi/A1.pkl',
                 RD_5per='DataSelection/DataSets/5percent/*.pkl'
                )
    
    if not path: path = tools.analysis_path(paths[kind])
    else: path = tools.analysis_path(path)
    
    #If Data is MC I read from NanoAOD
    #Else RealData is already in pd.DataFrames
    if 'RD' in kind and '*' not in path:    
        Data = pd.read_pickle(path)
        Data = cuts.apply_cuts(list_cuts, cuts_json_, Data)
    else:
        Data = pd.DataFrame() 
        for f in glob(path):
            if f.endswith('root'):
                f_ = uproot.open(f)['Events']
                skim_ = create_skim(f_, ['run', 'luminosityBlock', 'event'], 
                            'BToKMuMu', isMC=kind, verbose=verbose, run=run,
                            mu_branches=mu_branches)
            else:
                skim_ = pd.read_pickle(f)
            skim_ = cuts.apply_cuts(list_cuts, cuts_json_, skim_)
            if type(sample)==dict:
                print(len(skim_))
                skim_ = skim_.sample(**sample)
                print(sample)
                print(len(skim_))
            Data = Data.append(skim_)
            del skim_
            

    #Apply cuts
    Data = cuts.apply_cuts(list_cuts, cuts_json_, Data)
    
    #Select one candidate per event
    if OneCand:
        Data = select_cand(Data, OneCand, LumiMask=False, verbose=verbose)
    Binned_Data = join_split.only_split(Data, bins_json_)
    
    #Split data in q2 bins
    if Bin=='ALL':
        return Binned_Data
    else:
        return Binned_Data[Bin]
    
    

    
def create_gen_df(ntuple,
                  indexes=['run', 'luminosityBlock', 'event']):
    
    Bp4 = ntuple.array('B_p4')
    Bpt, Beta, Bphi, Bmass = Bp4.pt, Bp4.eta, Bp4.phi, Bp4.mass

    Mu1p4 = ntuple.array('Muon1_p4')
    Mu1pt, Mu1eta, Mu1phi = Mu1p4.pt, Mu1p4.eta, Mu1p4.phi

    Mu2p4 = ntuple.array('Muon2_p4')
    Mu2pt, Mu2eta, Mu2phi = Mu2p4.pt, Mu2p4.eta, Mu2p4.phi

    Kp4 = ntuple.array('K_p4')
    Kpt, Keta, Kphi= Kp4.pt, Kp4.eta, Kp4.phi
    
    Dimuonp4 = Mu1p4+Mu2p4
    
    run, lumi, event = ntuple.arrays(['run', 'luminosityBlock', 'event'], outputtype=tuple)
    cosThetaKMu = ntuple.array('costhetaKLJ')
    
    df = pd.DataFrame.from_dict(dict(
                    Bpt  = Bpt, Beta = Beta, Bphi = Bphi,
                    mu1pt= Mu1pt, mu1eta= Mu1eta, mu1phi= Mu1phi,
                    mu2pt= Mu2pt, mu2eta= Mu2eta, mu2phi= Mu2phi,
                    cosThetaKMu = cosThetaKMu, DiMuMass = Dimuonp4.mass,
                    run = run, luminositiBlock = lumi,  event = event), )
    df = df.set_index(['run', 'luminositiBlock', 'event'])
    
    return df

def gen_dataset_binned(
                   kind = 'PHSP',
                   Bin ='ALL', 
                   bins_json=3, 
                   path=None,
                  ):
    
    from glob import glob
    import tools
    import join_split
    
    bins_json_ = join_split.read_json(bins_json)
    paths = dict(PHSP = 'DataSelection/Gen_tuples/PHSP/*.root',
                 BSLLBALL = 'DataSelection/Gen_tuples/BSLLBALL/*.root',
                )
    if not path: path = tools.analysis_path(paths[kind])

    
    Data = pd.DataFrame() 
    for f in glob(path):
        f_ = uproot.open(f)['Analyzer']['ntuple']
        if len(f_)==0: print(f);continue
        Data = Data.append( create_gen_df(f_))

    if Bin=='none':
        return Data
    
    #Split data in q2 bins
    Binned_Data = join_split.only_split(Data, bins_json_)
    if Bin=='ALL':
        return Binned_Data
    else:
        return Binned_Data[Bin]