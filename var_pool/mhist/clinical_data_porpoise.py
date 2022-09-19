import pandas as pd
import os
import numpy as np

from var_pool.file_utils import safe_drop_suffix

avail_subtypes = ['blca', 'brca', 'coadread', 'gbmlgg', 'hnsc',
                  'kirc', 'kirp', 'lihc', 'luad',
                  'lusc', 'paad', 'skcm', 'stad', 'ucec']


def download_tcga_clinical_data(subtype, save_dir=None, verbose=True):
    """
    Downloads TCGA clinical data from the PORPOSE github repo https://github.com/mahmoodlab/PORPOISE/tree/master/dataset_csv/
    
    Parameters
    ----------
    subtype: str
        Which cancer subtype e.g. ['blca', 'brca', ...].
        
    save_dir: None, str
        (Optional) Directory where to save the csv file.
        
    Output
    ------
    df: pd.DataFrame
    """
    # download csv file
    url = 'https://raw.githubusercontent.com/mahmoodlab/PORPOISE/master/dataset_csv/tcga_{}_all.csv'.format(subtype)
    df = pd.read_csv(url)
    df = df.rename(columns={'Unnamed: 0': 'case_id'})
    
    # maybe save to disk
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fpath = os.path.join(save_dir, 'tcga_{}_all.csv'.format(subtype))
        df.to_csv(fpath, index=False)
        
    if verbose:
        n_slides = len(np.unique(df['slide_id']))
        n_cases = len(np.unique(df['case_id']))
        
        print('Downloading clinical data for {}'.format(subtype))
        print("Clinical shape {} with {} unique slides and {} unique case ids".\
              format(df.shape, n_slides, n_cases))
    
    return df


def load_clinical_data(save_dir, subtype, verbose=True):
    """
    Loads TCGA clinical data.

    Parameters
    ----------
    save_dir: None, str
        Directory where to save the csv file.

    subtype: str
        Which cancer type e.g. ['blca', 'brca', ...].

    Output
    ------
    df: pd.DataFrame
        The clinical data file with slide_id as the index.
    """
    # load file
    fpath = os.path.join(save_dir, 'tcga_{}_all.csv'.format(subtype))
    df = pd.read_csv(fpath)

    # make slide id the index
    n_slides = len(np.unique(df['slide_id']))
    assert n_slides == df.shape[0]

    # make slide id the index and drop file extension from slide ids
    df = df.set_index('slide_id')
    df.index = [safe_drop_suffix(s=i, suffix='.svs') for i in df.index]

    if verbose:
        n_cases = len(np.unique(df['case_id']))

        print('Clinical data for {}: shape {} with '\
              '{} unique slides and {} unique case ids'.\
              format(subtype, df.shape, n_slides, n_cases))

    return df
