#!/eos/home-h/hcrottel/conda_env/bin/python
import pdb
import pandas as pd
import numpy as np
import uproot3 as uproot
import os
import best_cand
#from index_tools import create_GEN_cand_mask
import resonance_rejection_antiRad_Veto
from termcolor import colored, cprint
from awkward0.array.jagged import JaggedArray

from time import time
import index_tools



b_branches = ['BToKMuMu_fit_pt', 'BToKMuMu_PDL', 
        'BToKMuMu_svprob', 'BToKMuMu_fit_mass', 'BToKMuMu_cosThetaKMu',
        #'BToKMuMu_significance*', 'BToKMuMu_cosAlpha_*', 
        #DiMuon's variables:
        'BToKMuMu_mll_llfit', 'BToKMuMu_mllErr_llfit','BToKMuMu_mll_pt',
        #K's variables:
        'BToKMuMu_fit_k_pt', 'BToKMuMu_fit_k_phi', 'BToKMuMu_fit_k_eta', 'BToKMuMu_k_charge',
        'BToKMuMu_k_nValidHits', 'BToKMuMu_k_lostInnerHits', 
        'BToKMuMu_k_matchLooseMuon', 'BToKMuMu_k_matchMediumMuon',
        'BToKMuMu_k_matchMuon','BToKMuMu_k_matchSoftMuon',
        #Muons variables:, 
        'BToKMuMu_fit_l1_pt', 'BToKMuMu_fit_l2_pt',
        'BToKMuMu_fit_l1_eta', 'BToKMuMu_fit_l2_eta',
        'BToKMuMu_fit_l1_phi', 'BToKMuMu_fit_l2_phi',
        'BToKMuMu_l1_charge', 'BToKMuMu_l2_charge',
    ]
mu_branches = ['Muon_isTriggering', 'Muon_nPixelHits',
                   'Muon_nPixelLayers', 'Muon_nTrackerLayers', 
                   'Muon_nValidHits'  , 'Muon_IP_BeamSpot', 
                   'Muon_IPerr_BeamSpot', 'Muon_isSoft_0']
    
trg_branches = ['HLT_Mu7_IP4',
             'HLT_Mu8_IP6',
             'HLT_Mu8_IP5',
             'HLT_Mu8_IP3',
             'HLT_Mu8p5_IP3p5',
             'HLT_Mu9_IP6',
             'HLT_Mu9_IP5',
             'HLT_Mu9_IP4',
             'HLT_Mu10p5_IP3p5',
             'HLT_Mu12_IP6',
             'L1_SingleMu7er1p5',
             'L1_SingleMu8er1p5',
             'L1_SingleMu9er1p5',
             'L1_SingleMu10er1p5',
             'L1_SingleMu12er1p5',
             'L1_SingleMu22']



def create_GEN_cand_mask(file, resonance=None, report=True):
    
    ### INDICES DE KAONES Y MUONES INVOLUCRADOS
    ### EN EL CANDIDATO
    kIdx  = file.array('BToKMuMu_kIdx')
    m1Idx = file.array('BToKMuMu_l1Idx')
    m2Idx = file.array('BToKMuMu_l2Idx')
    
    
    ##Collection ::: GenPart    interesting gen particles 
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




    
def get_branch(file, branch):
    try: 
        arr = file.array(branch)
    except KeyError:
        print(' ---> '+ branch + '  not found')
        arr = None
    return arr



def get_branches(file, list_branches):
    out_dict = dict()
    for branch in list_branches:
        arr = get_branch(file, branch)
        if type(arr)!=JaggedArray: continue
        out_dict[branch] = arr
    return out_dict


def get_index(file, index_branch):
    return file.array(index_branch)


def adjust_mu_branches(mu_branches, mu1_index, mu2_index):
    mu_branches_adjust = dict()
    for k in mu_branches:
        mu_branches_adjust[k.replace('Muon', 'Mu1')] = mu_branches[k][mu1_index].flatten()
        mu_branches_adjust[k.replace('Muon', 'Mu2')] = mu_branches[k][mu2_index].flatten()
    return mu_branches_adjust



def expand_simple_branch(file, branch, counts):
    """There are some branches that have a single value per event for example run, evt, lumi
    In order to flatten the file into a table we need to repeat this number for each candidate"""
    branch_arr = file.array(branch)
    flattened_branch = list()
    for i, n_cands in enumerate(counts):
            for n in range(n_cands): flattened_branch.append(branch_arr[i])
    return flattened_branch

def expand_simple_branches(file, branches, counts):
    """Call back to expand_simple_branch"""
    out_dict=dict()
    for branch in branches:
        out_dict[branch] = expand_simple_branch(file, branch, counts)
    return out_dict

def create_evt_info(file, counts):
    """Special function used for evt lumi run"""
    branches=['event', 'luminosityBlock', 'run']
    return expand_simple_branches(file, branches, counts)
                
                
def create_evt_info_old(file, counts):
    Event, Lumi, Run = file.arrays(['event', 'luminosityBlock', 'run'], outputtype=tuple)
    event_n, lumi_n, run_n = list(), list(), list()
    for i, n_events in enumerate(counts):
        for n in range(n_events):
            event_n.append(Event[i])
            lumi_n.append(Lumi[i])
            run_n.append(Run[i])
    return {'event':event_n, 'luminosityBlock':lumi_n, 'run':run_n}






def produce_matrix(file, branch):
    """Produce a np.array of shape (n_cands, 4)"""
    values = [file.array(f'{branch}{i}').flatten() for i in range(4)]
    return np.stack(values, axis=1)

def softMuon_matrix(file, indx):
    soft = [file.array(branch)[indx].flatten() for branch in 
                ['Muon_isSoft_0', 'Muon_isSoft_1', 'Muon_isSoft_2', 'Muon_isSoft_3']]
    soft = np.stack(soft, axis=1)
    return soft

def best_PV(file):
    """Select the best PV per candidate via the highest cosine of pointing anlge"""
    cos_alphas = produce_matrix(file, 'BToKMuMu_cosAlpha_')
    bestPV_    = cos_alphas.argmax(axis=1)
    return bestPV_


def quality_cuts(file, mu1, mu2, bPV):
    """Apply quality cuts defined by Jhovanny"""
    #pdb.set_trace()
    
    mu1_soft = softMuon_matrix(file, mu1)
    mu2_soft = softMuon_matrix(file, mu2)
    
    soft_muons = (mu1_soft[np.arange(len(mu1_soft)), bPV]) & (mu2_soft[np.arange(len(mu2_soft)), bPV])
    
    
    tracks_b   = ['BToKMuMu_k_HighPurity',
		          'BToKMuMu_k_numberOfPixelHits',
	   			  'BToKMuMu_k_numberOfHits'] 
    high_p, pixel_h, hits = file.arrays(tracks_b, 
		                                outputtype=tuple)
    
    quality_tracks = (high_p.flatten()==1) &\
                     (pixel_h.flatten()>=1) &\
                     (hits.flatten()>=5)

    
    quality_mask   = soft_muons & quality_tracks
    
    return quality_mask
    


def create_flat_dict(file, mu_dict, b_dict, info, mu1, mu2, bPV):
    
    flat_Dictionary = dict()

    
    cos_alpha_matrix    = produce_matrix(file, 'BToKMuMu_cosAlpha_')
    significance_matrix = produce_matrix(file, 'BToKMuMu_significance')
    
    
    for k in info:
        flat_Dictionary[k] = info[k]
    
    for k in b_dict:
        branch = k
        if 'significance' in k: branch = k.replace('significance', 'signLxy')
        elif 'cosAlpha' in k: branch = k.replace('cosAlpha_', 'cosA')
        flat_Dictionary[branch.replace('BToKMuMu_', '')] = b_dict[k].flatten()
    
    flat_Dictionary['cosA']    = cos_alpha_matrix[np.arange(len(cos_alpha_matrix)), bPV]
    flat_Dictionary['signLxy'] = significance_matrix[np.arange(len(significance_matrix)), bPV]
    
    #flat_Dictionary['Mu1_isSoft'] = mu1_soft[np.arange(len(mu1_soft)), bPV]
    #flat_Dictionary['Mu2_isSoft'] = mu2_soft[np.arange(len(mu2_soft)), bPV]
    
    for k in mu_dict:
        flat_Dictionary[k.replace('Muon', 'Mu1')] = mu_dict[k][mu1].flatten()
        flat_Dictionary[k.replace('Muon', 'Mu2')] = mu_dict[k][mu2].flatten()
    
    return flat_Dictionary


def create_df(flat_dict):
    df = pd.DataFrame.from_dict(flat_dict)
    names = {'fit_pt' : 'Bpt', 'svprob':'prob', 'fit_mass':'BMass', 
        'mll_llfit':'DiMuMass', 'mllErr_llfit':'errDiMuMass', 'mll_pt':'DiMupt',
        'fit_k_pt':'kpt', 'fit_l1_pt':'l1pt', 'fit_l2_pt':'l2pt'}
    cols_ = [names.get(k,k) for k in df.keys()]
    df.columns = cols_
    return df


def flatten_to_df(file, mu_branches, b_branches, mu1, mu2, isMC=False, qcuts=False):
    
    B_v0  = get_branches(file, 
                         b_branches)
    Mu_v0 = get_branches(file, 
                         mu_branches)
    mu1Indx = get_index(file, mu1)
    mu2Indx = get_index(file, mu2)
    
    bPV = best_PV(file)
    
    info = create_evt_info(file, mu1Indx.count())
    
    try:
        triggers_dict = expand_simple_branches(file, trg_branches, mu1Indx.count())
        for t in triggers_dict:
            info[t] = triggers_dict[t]
    except Exception as e:
        print(e)
    
    flat_dict = create_flat_dict(file, 
                                 Mu_v0, B_v0, info,
                                 mu1Indx, mu2Indx, bPV)

    
    
    if isMC:
        gen_mask = create_GEN_cand_mask(file)
        flat_dict['GENCand'] = gen_mask.flatten()

    df = create_df(flat_dict)
    if qcuts:
        quality_mask = quality_cuts(file, mu1Indx, mu2Indx, bPV)
        return df[quality_mask.astype(bool)]
    else:
        return df

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':

    import tools
    import cuts
    import pickle
    import xgboost as xgb
    import argparse
    
    parser    = argparse.ArgumentParser(description='Flatten NanoAODs and apply cuts')
    
    
    type_data = parser.add_mutually_exclusive_group(required=True)
    
    type_data.add_argument('--MC', 
                            action='store_true',
                            help  ='Set if MonteCarlo Dataset')
    type_data.add_argument('--RD',
                            action='store_true', 
                            help='Set if Real Data (Collision Data)')
	
    
    parser.add_argument('-c',
						action='store',
						default=None, 
                       help='ID for cuts.json')
    
    parser.add_argument('-w',
					   action='store',
					   default='all', 
                       help='which cuts to apply')

    parser.add_argument('-wn',
                       action='store',
                       default='none', 
                       help='which cuts NOT to apply, if more than 1, must be separated by _')

    
    parser.add_argument('--inputdir',
                       action='store',
                       required=False, 
                       help='Parent directory to look for NanoAODs')
    parser.add_argument('--inputfile',
                       action='store',
                       required=False, 
                       help='Path to read the pd.DataFrame')
    
    parser.add_argument('--outputfile',
                       action='store',
                       required=False, 
                       type=str,
                       default='-',
                       help='Parent directory to save the pkl, \
                        defaults to ./Skim{c}/RR_ARV_XGB.pkl')
    
    parser.add_argument('--skipFiles', 
                       action='store',
                        type=int,
                       default=0,
                       help='Skip the first n files')
        
    parser.add_argument('--maxFiles', 
                        action='store', 
                        type=int, 
                        default=-1,
                        help='Max number of files -> [skipFiles:skipFiles+maxFiles]')

    parser.add_argument('--multindex',
                        action='store', 
                        type=int, 
                        default=-1,
                        help='Use updated algorithm to make skim, with pd Multindex')
    
    parser.add_argument('--fraction',
                        action='store', 
                        type=float, 
                        default=0.01,
                        help='Fraction of complete dataset to be used, you can set random seed')
    
    parser.add_argument('--randomseed',
                        action='store', 
                        type=int, 
                        default=0,
                        help='Seed for random selection of events. Given fraction')


    parser.add_argument('--columns',
                        action='store', 
                        type=str, 
                        default='all',
                        help='Which columns to save. Allowed strings are:\
                        all -> All columns returned so far, all B Table from NanoAOD, and some from muon Tables, and Triger branches\
                        minimal -> All columns needed to apply all cuts, produce XGB column, a Kinematics, as well as the mass and angular var for the 2D fit'

                    )
    
    args = parser.parse_args() 
    
    
    
    
    
    
    
    
    b_branches = ['BToKMuMu_fit_pt', 'BToKMuMu_PDL', 
            'BToKMuMu_svprob', 'BToKMuMu_fit_mass', 'BToKMuMu_cosThetaKMu',
            #'BToKMuMu_significance*', 'BToKMuMu_cosAlpha_*', 
            #DiMuon's variables:
            'BToKMuMu_mll_llfit', 'BToKMuMu_mllErr_llfit','BToKMuMu_mll_pt',
            #K's variables:
            'BToKMuMu_fit_k_pt', 'BToKMuMu_fit_k_phi', 'BToKMuMu_fit_k_eta', 'BToKMuMu_k_charge',
            'BToKMuMu_k_nValidHits', 'BToKMuMu_k_lostInnerHits', 
            'BToKMuMu_k_matchLooseMuon', 'BToKMuMu_k_matchMediumMuon',
            'BToKMuMu_k_matchMuon','BToKMuMu_k_matchSoftMuon',
            #Muons variables:, 
            'BToKMuMu_fit_l1_pt', 'BToKMuMu_fit_l2_pt',
            'BToKMuMu_fit_l1_eta', 'BToKMuMu_fit_l2_eta',
            'BToKMuMu_fit_l1_phi', 'BToKMuMu_fit_l2_phi',
            'BToKMuMu_l1_charge', 'BToKMuMu_l2_charge',
        ]
    mu_branches = ['Muon_isTriggering', 'Muon_nPixelHits',
                   'Muon_nPixelLayers', 'Muon_nTrackerLayers', 
                   'Muon_nValidHits'  , 'Muon_IP_BeamSpot', 
                   'Muon_IPerr_BeamSpot','Muon_isSoft_0']
    
    
    trg_branches = ['HLT_Mu7_IP4',
                 'HLT_Mu8_IP6',
                 'HLT_Mu8_IP5',
                 'HLT_Mu8_IP3',
                 'HLT_Mu8p5_IP3p5',
                 'HLT_Mu9_IP6',
                 'HLT_Mu9_IP5',
                 'HLT_Mu9_IP4',
                 'HLT_Mu10p5_IP3p5',
                 'HLT_Mu12_IP6',
                 'L1_SingleMu7er1p5',
                 'L1_SingleMu8er1p5',
                 'L1_SingleMu9er1p5',
                 'L1_SingleMu10er1p5',
                 'L1_SingleMu12er1p5',
                 'L1_SingleMu22']
    
    
    
    
    lists_of_columns = dict(
        minimal = ['Bpt',  'fit_eta',    'fit_phi',
                      'kpt',  'fit_k_eta',  'fit_k_phi',  
                      'l1pt', 'fit_l1_eta', 'fit_l1_phi', 
                      'l2pt', 'fit_l2_eta', 'fit_l2_phi',
                      'DiMuMass', 'BMass', 'cosThetaKMu',
                      'PDL', 'prob', 'signLxy', 'cosA',
                      'mu1_IP_sig', 'mu2_IP_sig'
                     ],
        )
    
    
    #Select input files
    list_of_files = list()

    if not args.inputfile and args.inputdir:
        
        #Look for all subdirectories
        for root,dirs,files in os.walk(args.inputdir):
            files_ = [os.path.join(root,f) 
                      for f in files if f.endswith('root') and 'BParkNANO' in f]

            #Append files that end with root and have BParkNANO
            if any(files_): list_of_files+=files_

        #All files?
        if args.maxFiles==-1:
            files = list_of_files[ args.skipFiles : ]
        else:
            files = list_of_files[ args.skipFiles : args.skipFiles+args.maxFiles ]
    
    #If not specified add the ??
    elif args.inputdir:
        files = [args.inputfile]

    else:
        raise NotImplementedError
    
    
    
    
    
    
    
    
    
    #String for cuts - hardcoded!!
    if args.c: 
        import cuts
        json_cuts = cuts.read_cuts_json(int(args.c))
        if 'all' in args.w:
            list_cuts = ['resonance_rejection', 
                         'anti_radiation_veto',
                         'Quality', 
                         'XGB',
                         'missID',
                         'Bpt','l1pt', 'l2pt', 
                         'k_min_dr_trk_muon', 
                         'triggering_muons']
            if any(var in args.w for var in ['jpsi', 'psip', 'resonances']):
                list_cuts.remove('resonance_rejection')
                list_cuts.remove('anti_radiation_veto')
        elif args.w == 'rab':
            list_cuts = ['resonance_rejection', 'anti_radiation_veto', 'Bpt']
        elif args.w == 'keep_resonances':
            list_cuts = ['Quality', 
                         'XGB',
                         'missID',
                         'Bpt','l1pt', 'l2pt', 
                         'k_min_dr_trk_muon', 
                         'triggering_muons']
        elif args.w == 'noFake':
            list_cuts = ['resonance_rejection', 'anti_radiation_veto', 'Bpt', 'XGB']
        elif args.w == 'ra':
            list_cuts = ['resonance_rejection', 'anti_radiation_veto']
        elif args.w == 'r':
            list_cuts = ['resonance_rejection']
        elif args.w in ['jpsi', 'psi2s', 'none', 'sidebands']:
            list_cuts = []
        elif args.w == 'XGB':
            list_cuts = ['XGB']
        else:
            raise NotImplementedError



    #Remove cuts from the list_cuts
    if args.wn!='none':
        for val in args.wn.split('_'):
            list_cuts.remove(val)












    # Default Model
    # models_dir = tools.analysis_path('XGB/Models/Oct20/')        

    # ### Read once XGB models!
    # if "pathR" in json_cuts['XGB']:
    #     print('---->>>> HERE! -- R')
    #     cprint(f'Right Model  :  {json_cuts["XGB"]["pathR"]}', 'yellow')
    #     #log+=f"Right Model  :  {json_cuts['XGB']['pathR']}\n"
        
    #     if json_cuts['XGB']["pathR"].endswith('.pkl'):
    #         right_model = pickle.load(open(tools.analysis_path(json_cuts['XGB']["pathR"]), 'rb')) 
    #     else:
    #         right_model = xgb.XGBClassifier()                   
    #         right_model.load_model(tools.analysis_path(json_cuts['XGB']["pathR"]))
    # else:
    #     cprint(f'Right Model  :  {models_dir}model2_rightSB_2.pickle.dat', 'yellow')
    #     #log+=f"Right Model  :  {models_dir+'model2_rightSB_2.pickle.dat'}\n"
    #     try:
    #         right_model = pickle.load(open(models_dir+'model2_rightSB_2.pickle.dat', 'rb')) 
    #     except pickle.UnpicklingError as e:
    #         cprint(f'Cannot import Default model. Maybe due to missmatch versions of pickle:', 'magenta')
    #         cprint(e, 'red')
    # json_cuts["XGB"]["pathR"] = right_model
    
    
    
    
    
    
    # if "pathL" in json_cuts['XGB']:
    #     cprint(f'Left Model   :  {json_cuts["XGB"]["pathL"]}', 'yellow')
    #     #log+=f"Left Model  :   {json_cuts['XGB']['pathL']}\n"
    #     if json_cuts['XGB']["pathL"].endswith('.pkl'):
    #         left_model = pickle.load(open(tools.analysis_path(json_cuts['XGB']["pathL"]), 'rb')) 
    #     else:
    #         left_model = xgb.XGBClassifier()                   
    #         left_model.load_model(tools.analysis_path(json_cuts['XGB']["pathL"]))                    

    # else:
    #     cprint(f'Left Model   :   {models_dir}model2_leftSB_2.pickle.dat', 'yellow')
    #     #log+=f"Left Model  :   {models_dir+'model2_leftSB_2.pickle.dat'}\n"
    #     left_model  = pickle.load(open(models_dir+'model2_leftSB_2.pickle.dat', 'rb'))  
    # json_cuts["XGB"]["pathL"] = left_model
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


    from platform import python_version
    print('PYTHON VERSION: ',python_version(), '\n\n')
        
    DF = pd.DataFrame()
    print('n Files: ', len(files))
    print('FILES\n', files)
    info = ['run', 'luminosityBlock', 'event']

    start = time()

    if 'XGB' in list_cuts:
         XGBoost_ = True
         list_cuts.remove('XGB')
    
    else: XGBoost_  = False
            
    for i, f in enumerate(files):
        
        print(colored('\n---------------------------------------------------------', 'yellow'))
        print(colored(f.split('/')[-1], 'green'))
        file_ = uproot.open(f)['Events']
        print('\t -- Initial Len : ', len(file_))
        
        
        
        if args.multindex==-1:
            temp = flatten_to_df(file_, mu_branches, b_branches, 
				       'BToKMuMu_l1Idx', 'BToKMuMu_l2Idx', args.MC, qcuts=True)
        else:
            temp = index_tools.create_skim(file_, info, 'BToKMuMu', isMC=args.MC, softMuons=True)

        
        if args.c:


            print(list_cuts, f' + Quality Cuts')
            temp = cuts.apply_quality_cuts(temp)
            print(len(temp))
            print(list_cuts)
            temp = cuts.apply_cuts(list_cuts, json_cuts, temp)
            print(len(temp))
            #temp = cuts.apply_simple_cut(5.0, temp, column = 'BMass', type_='ge')
            #temp = cuts.apply_simple_cut(5.7, temp, column = 'BMass', type_='le')
            #is_res = [x if x in args.w.lower() for x in ['jpsi', 'psip', 'psi2s', 'resonances']]
            is_res = [x for x in ['jpsi', 'psip', 'psi2s', 'resonances'] if x in args.w]
            if is_res:
                dimuon_mass_windows = dict(jpsi = [2.8, 3.4], 
                                            psip = [3.4, 4.0], 
                                            psi2s = [3.4, 4.0], 
                                            resonances=[2.8, 4.0])
                dimuon_mass_window =  dimuon_mass_windows[is_res[0]]
                print('\n', args.w, '\t->Dimuon window : ',dimuon_mass_window)
                temp = temp.query(f'{dimuon_mass_window[0]}<=DiMuMass<{dimuon_mass_window[1]}')
            elif args.w == 'sidebands':
                leftw  = [5.0, 5.15] 
                rightw = [5.4, 5.7]
                print('\n', args.w, '\t->Left  window : ',leftw)
                print('\n', args.w, '\t->Right window : ',rightw)
                temp = temp.query(f'({leftw[0]}<=BMass<{leftw[1]}) or ({rightw[0]}<=BMass<{rightw[1]})')
                if args.fraction==1:
                    XGB_cols   = ['Bpt', 'kpt', 'PDL', 'prob', 'cosA', 'signLxy']
                    Other_cols = ['BMass', 'cosThetaKMu', 'DiMuMass'] 
                    temp = temp[XGB_cols+Other_cols]
            elif args.w in ['none', 'keep_resonances']:
                print('\n-No more cuts :)\n')
        

            # elif args.w != 'r':
            #     temp = cuts.apply_simple_cut(9.0, temp, column = 'Bpt', type_='ge')
            #     temp = cuts.apply_simple_cut(2.0, temp, column = 'mu1_pt', type_='ge')
            #     temp = cuts.apply_simple_cut(2.0, temp, column = 'mu2_pt', type_='ge')
            #     #temp = cuts.apply_simple_cut(temp.BMass>=5.0) & (temp.BMass<=5.7)
            

        print('\n\t -- Final   Len   : ', len(temp))
        
        
        if args.MC:
            print('\n\t -- Final Len Gen : ', len(temp[temp.GENCand]))
            
        
        ### Keeping only with the desired fraction
        if args.fraction<1:
            temp = temp.sample(frac=args.fraction, random_state=args.randomseed)
            print('\n\t -- Final Len of the fraction : ', len(temp))
        elif args.fraction>1:
            print('BEWARE fraction is greater than 1 !!!')
            print('If you want to have more events please code this part')
            print('Keeping only the 100%', '\n')
        ### Keeping only with the desired fraction
        
        
        if args.multindex==-1:
            DF = DF.append(temp, ignore_index=True)
    
        else:
            DF = DF.append(temp)

        print(colored('---------------------------------------------------------\n', 'yellow'))

        print(len(DF), len(temp))
        
        if i%10 == 0: print(f'\tFile {i+1}  |-->  Time = {round(time()-start)}\n')
        
    if XGBoost_:
        print('\n\n\n',colored('----> Applying XGBoost Cut\n', 'red'))
        print('Initial :' , len(DF))
        DF = cuts.apply_cuts(['XGB'], json_cuts, DF)
        print('Final   :', len(DF))


    if args.columns != 'all':
        print('Saving columns...\n')
        save_cols = lists_of_columns.get(args.columns, list(DF.keys()))
        save_cols += [col for col in DF.keys() if 'XGB' in col] # if there is a column named *XGB* add it
        DF = DF[save_cols]
        print(save_cols, '\n... saved :)')

    if args.outputfile:
        name = args.outputfile
        if name.endswith('pkl'):
            DF.to_pickle(name, protocol=4)
        else:
            DF.to_pickle(name+'.pkl', protocol=4)
    else:
        DF.to_pickle('flatten.pkl', protocol=4)

    print(time()-start)
