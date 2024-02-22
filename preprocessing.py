import sys
import numpy as np
import pandas as pd
import tabulate as tab
import pdg
import root_pandas
import glob


def merge_neutralino_w_main_tree(main, neutralino):
    # sanity check
    evts_main = main.__event__.unique()
    evts_schi = neutralino.__event__.unique()
    if evts_main.sum() != evts_schi.sum(): sys.exit('number of events does not match')
        
    ds = pd.merge(main, schi, on="__event__")
    
    # more sanity checks
    for v in ['__experiment__',  '__run__',  '__production__',  '__candidate__','__ncandidates__',  '__weight__']:
        if ds[f'{v}_x'].all()==ds[f'{v}_y'].all():
            ds[v] = ds[f'{v}_x']
            ds = ds.loc[:,~ds.columns.str.contains(f'{v}_x')]
            ds = ds.loc[:,~ds.columns.str.contains(f'{v}_y')]
        else:
            sys.exit('Something wrong in the variables.')
        
    for v in [ 'IPX', 'IPY', 'IPZ' ]:
        if ds[f'{v}'].all()==ds[f'schi_{v}'].all():
            ds = ds.loc[:,~ds.columns.str.contains(f'schi_{v}')]
        else:
            sys.exit('Something wrong in the variables.')
    
    return ds

def add_charge(df):
    prefixes  = [f'{p}_trk{i}' for p in ['p', 'pi', 'K', 'e', 'mu'] for i in range(1, 10)]
    for prefix in prefixes:
        if 'e' in prefix or 'mu' in prefix:
            df[f'{prefix}_charge'] = -np.sign(df[f'{prefix}_mcPDG'])
        else:
            df[f'{prefix}_charge'] = np.sign(df[f'{prefix}_mcPDG']) 
    return df

def count_tracks(df):
    # includes the prompt proton
    mcPxs  = [f'{p}_trk{i}_mcPX' for p in ['p', 'pi', 'K', 'e', 'mu'] for i in range(1, 10)]
    df['nTrks_all'] = df[mcPxs].count(axis = 1)
    return df
    
def flag_LLps(df): 
    ## final tuple: we want to remove them...
    LLps = ['Lambda0', 'Sigma+', 'Sigma-', 'K_S0', 'K_L0', 'Xi0', 'Xi-', 'Omega-']
    LLps_ids = pdg.from_names(LLps)
    prefixes  = [f'{p}_trk{i}' for p in ['p', 'pi', 'K', 'e', 'mu'] for i in range(1, 10)]
    for p in prefixes:
        df[f'{p}_hasLLancestor'] = np.where( abs(df[f'{p}_genMotherPDG']).isin(LLps_ids), 1, 0 )
    ancestors = [f'{p}_hasLLancestor' for p in prefixes]
    df['nTrks_wLLancestors']=df.loc[:,ancestors].sum(axis=1)
    return df
    
def flag_prompt_p(df):
    prefixes  = [f'p_trk{i}' for i in range(1, 10)]
    for p in prefixes:
        df[f'{p}_isPrompt'] = np.where( abs(df[f'{p}_genMotherPDG'])==521, 1, 0 )
    return df
    
def find_neutralino_vertex(df):
    df['neutralino_vtxX'] = df['schi_mcDecayVertexX'] - df['schi_mcProductionVertexX']
    df['neutralino_vtxY'] = df['schi_mcDecayVertexY'] - df['schi_mcProductionVertexY']
    df['neutralino_vtxZ'] = df['schi_mcDecayVertexZ'] - df['schi_mcProductionVertexY']
    return df
    
def find_prompt_proton_momenta(df):
    df['p_prompt_mcPX'] = np.where( df.p_trk1_isPrompt==1, df[f'p_trk1_mcPX'], np.nan)
    df['p_prompt_mcPY'] = np.where( df.p_trk1_isPrompt==1, df[f'p_trk1_mcPY'], np.nan)
    df['p_prompt_mcPZ'] = np.where( df.p_trk1_isPrompt==1, df[f'p_trk1_mcPZ'], np.nan)
    i=2
    while (df['p_prompt_mcPX'].isna().sum()>0) & (i<10):
        df['p_prompt_mcPX'] = np.where( df[f'p_trk{i}_isPrompt']==1, df[f'p_trk{i}_mcPX'], df['p_prompt_mcPX'])
        df['p_prompt_mcPY'] = np.where( df[f'p_trk{i}_isPrompt']==1, df[f'p_trk{i}_mcPY'], df['p_prompt_mcPY'])
        df['p_prompt_mcPZ'] = np.where( df[f'p_trk{i}_isPrompt']==1, df[f'p_trk{i}_mcPZ'], df['p_prompt_mcPZ'])
        i+=1
    print(f' sanity check: there are {df.p_prompt_mcPX.isna().sum()} entries for which the prompt proton momenta has not been found')
    return df
    
def count_tracks_available_for_vertex(df):
    # nTrks_all - nLLps - promptProton
    df['nTrks_valid'] = df.nTrks_all - df.nTrks_wLLancestors - 1
    return df
     
def clean(df):
    # remove useless vars
    useless_vars = ['__weight__', 'roeMC_Pt', 'roeMC_Px', 'roeMC_Py', 'roeMC_Pz', 'roeMC_P', 'roeMC_PTheta', 'roeMC_E', 'roeMC_M']
    trk_vars = ['genParticleID', 'isCloneTrack', 'mcInitial', 'mcVirtual', 'nMCMatches']
    useless_trk_vars = [f'{p}_trk{t}_{v}' for p in ['p', 'pi', 'K', 'e', 'mu', 'schi'] for t in range(1,10) for v in trk_vars]
    for v in useless_vars+useless_trk_vars:
        df = df.loc[:,~df.columns.str.contains(v)]
        
    # remove useless columns
    for tr in [f'p_trk{i}' for i in range(3,10)]:
        df = df.loc[:,~df.columns.str.startswith(tr)]
    for tr in [f'K_trk{i}' for i in range(4,10)]:
        df = df.loc[:,~df.columns.str.startswith(tr)]
    for tr in [f'e_trk{i}' for i in range(6,10)]:
        df = df.loc[:,~df.columns.str.startswith(tr)]
    for tr in [f'mu_trk{i}' for i in range(3,10)]:
        df = df.loc[:,~df.columns.str.startswith(tr)]
        
    # I decided NOT to remove these events to simplify the efficiency estimation
    # df = df[df.nTrks_valid>1]

    return df

if __name__ == '__main__':
    masses = [ 2700, 2800, 2900, 
              3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900,
              4000, 4100, 4200, 4300] 
    for mass in masses: # done
        print(f'\nMass = {mass}')
        # files = glob.glob(f'/gpfs/home/belle/nayak/signalMC_Btopchi/output_root/output_Btopchi_{mass}MeV.root')
        files = glob.glob(f'/group/belle2/users2022/eberthol/RPV_pheno/signal_nTuples/output_Btopchi_{mass}MeV.root')
        main = root_pandas.read_root(files, 'roe_e_gen')
        print(f' main tree original shape: {main.shape}')
        schi = root_pandas.read_root(files, 'schi')
        print(f' neutralino tree original shape: {schi.shape}')

        df = merge_neutralino_w_main_tree(main, schi)

        # df = add_charge(df)
        df = count_tracks(df)
        df = flag_LLps(df)
        df = flag_prompt_p(df)
        df = find_neutralino_vertex(df)
        df = find_prompt_proton_momenta(df)
        df = count_tracks_available_for_vertex(df)
        df = clean(df)
        # write file
        root_pandas.to_root(df, f'/group/belle2/users2022/eberthol/RPV_pheno/signal_preproc/Btopchi_{mass}MeV.root', key='tree')

        print(f' number of entries with multiple candidates: {df[df.__ncandidates__>1].shape[0]}')
        print(f' number of entries where the trk1 IS NOT the prompt proton: {df[abs(df.p_trk1_genMotherPDG)!=521].shape[0]} ({df[abs(df.p_trk1_genMotherPDG)!=521].shape[0]*100/df.shape[0]:.1f} %)')
        print(f' number of events with less than 2 additional (valid) tracks: {df[df.nTrks_valid<2].shape[0]} (not removed!)')
        print(f' final shape: {df.shape}')