import root_pandas
import numpy as np
import pandas as pd
from track_propagation_in_CDC import *
from pathlib import Path 

def compute_nCDCHits28(cdc, nTrks_valid, LLancestor, charge, mcPX, mcPY, mcPZ, mcProductionVertexX, mcProductionVertexY, mcProductionVertexZ):
        if (np.isnan(mcPX)==True) or (LLancestor!=0) or (nTrks_valid<2):
            nCDCHits28 = -999
        else:
            d =  {
                'q':  charge,
                'px': mcPX,
                'py': mcPY,
                'pz': mcPZ,
                'prodVtxX': mcProductionVertexX, 
                'prodVtxY': mcProductionVertexY, 
                'prodVtxZ': mcProductionVertexZ
            }
            traj = trajectory(d, cdc, 0.1, step_in_phi=False)
            nCDCHits28 = traj.CDChits_clean28
        return nCDCHits28 

if __name__ == '__main__':
    to_cm, unit_label = 0.1, '[cm]'

    cdc_TDR = {
        'A' : np.array([ -(474+208+149),  1082+150  ])*to_cm, ## in TDR it's 1082 but the rho of the last layer ids larger than that... I added a dummy value so that all the layers are inside the 'frame'
        'B' : np.array([  (877+572+137),  1082+150  ])*to_cm, ## in TDR it's 1082 but the rho of the last layer ids larger than that... I added a dummy value so that all the layers are inside the 'frame'
        'C' : np.array([ -(474+208),     371.4  ])*to_cm,  ## corresponds to C2 in TDR
        'D' : np.array([  (877+572),     438.0  ])*to_cm, 
        'E' : np.array([       -474,     249.5  ])*to_cm,
        'F' : np.array([        877,     249.5  ])*to_cm,
        'G' : np.array([   -999,     160*to_cm  ]),
        'H' : np.array([   -999,     160*to_cm  ]),
        'rho_min'  : np.array([168.0, 257.0, 365.2, 476.9, 584.1, 695.3, 802.5,  913.7, 1020.9])*to_cm,
        'rho_max'  : np.array([238.0, 348.0, 455.7, 566.9, 674.1, 785.3, 892.5, 1003.7, 1111.4])*to_cm,
        'nLayers'  : np.array([   8,    6,     6,     6,     6,     6,     6,      6,      6]),
        'nCells'   : np.array([ 160,  160,   192,   224,   256,   288,   320,    352,    384]),  
    }
    cdc = CDC(cdc_TDR)

    batch_low  = int(sys.argv[1])
    batch_high = int(sys.argv[2])
    mass       = sys.argv[3]
    folder     = sys.argv[4]

    # batch_low  = 0
    # batch_high = 5
    # mass = '2700'
    # fileID = '0'
    file = f'/group/belle2/users2022/eberthol/RPV_pheno/signal_preproc/Btopchi_{mass}MeV.root'
    print(file)
    df = root_pandas.read_root(file)
    df.reset_index(drop=True, inplace=True)
    Nentries = df.shape[0]
    print(Nentries)

    # folder = 'signal_final_nTuples'
    Path(f'{folder}/logs').mkdir(parents=True, exist_ok=True)
    filename = f'sig_{mass}MeV_{batch_low}-{batch_high}'
    output     = f'{folder}/{filename}.root'

    df = df.loc[batch_low:batch_high].copy()
    prefs = ['p_trk1', 'p_trk2']
    for pref in prefs:
        df[f'{pref}_nCDCHits28'] = df.apply(lambda x: compute_nCDCHits28(cdc, x[f'nTrks_valid'], x[f'{pref}_hasLLancestor'], x[f'{pref}_charge'], \
                                                                         x[f'{pref}_mcPX'], x[f'{pref}_mcPY'], x[f'{pref}_mcPZ'], \
                                                                         x[f'{pref}_mcProductionVertexX'], x[f'{pref}_mcProductionVertexY'], x[f'{pref}_mcProductionVertexZ']), axis=1)
    # associate nCDCHits to prompt p
    df['p_prompt_nCDCHits28'] = np.where( df.p_trk1_isPrompt==1, df[f'p_trk1_nCDCHits28'], df[f'p_trk2_nCDCHits28'])

    pis = [f'pi_trk{i}' for i in range(1, 10)]
    Ks  = [f'K_trk{i}' for i in range(1, 4)]
    es  = [f'e_trk{i}' for i in range(1, 6)]
    mus = [f'mu_trk{i}' for i in range(1, 3)]
    tracks = pis+Ks+es+mus
    for pref in tracks:
        df[f'{pref}_nCDCHits28'] = df.apply(lambda x: compute_nCDCHits28(cdc, x[f'nTrks_valid'], x[f'{pref}_hasLLancestor'], x[f'{pref}_charge'], \
                                                                         x[f'{pref}_mcPX'], x[f'{pref}_mcPY'], x[f'{pref}_mcPZ'], \
                                                                         x[f'{pref}_mcProductionVertexX'], x[f'{pref}_mcProductionVertexY'], x[f'{pref}_mcProductionVertexZ']), axis=1)
    
    root_pandas.to_root(df, output, key='tree') 
    print(f'file wrote to {output}')

    # print(df.loc[:, ['__event__', 'p_trk1_mcP', 'p_trk2_mcP', 'p_trk1_hasLLancestor', 'p_trk2_hasLLancestor', 
    #        'p_trk1_isPrompt', 'p_trk2_isPrompt', 'p_trk1_nCDCHits28', 'p_trk2_nCDCHits28', 'p_prompt_nCDCHits28']])

    ## after inspection of the original tuples:
    # nMax particles per events
    # p: 2
    # pi:9
    # K: 3
    # e: 5
    # mu 2

    # bsub -q s -J CDC python3 add_nCDChits.py 







