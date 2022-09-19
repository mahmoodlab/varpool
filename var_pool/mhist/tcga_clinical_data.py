import pandas as pd
import numpy as np


def get_participant_from_tcga_barcode(barcode):
    """
    Takes a TCGA barcode and returns the participant part 
    
    TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291 -> TCGA-3C-AALI
    
    https://docs.gdc.cancer.gov/Encyclopedia/pages/TCGA_Barcode/
    
    Parameters
    ----------
    barcode: str
        The barcode.
        
    Output
    ------
    participant: str
        The first three -s from the barcode.
    """
    return '-'.join(barcode.split('-')[0:3])


def load_patient_data_from_cell_df(fpath, subtype=None, verbose=True):
    """
    Download clinical data file from (Liu et al, 2018) from
    https://www.cell.com/cms/10.1016/j.cell.2018.02.052/attachment/bbf46a06-1fb0-417a-a259-fd47591180e4/mmc1
    
    it should be named mmc1.xlsx

    Paper can be found at
    https://www.cell.com/cell/fulltext/S0092-8674(18)30229-0
    
    
    Parameters
    ----------
    fpath: str
        File path to data file from Table S1 (Liu et al, 2018).
    
    subtype: None, str
        (Optional) Subtype to subset to.
    
    Output
    ------
    patient_clinical_df, cols

    patient_clinical_df: pd.DataFrame, (n_patients, n_features)

    cols: dict of lists
        Datatypes for each column

    References
    ----------
    Liu, J., Lichtenberg, T., Hoadley, K.A., Poisson, L.M., Lazar, A.J., Cherniack, A.D., Kovatich, A.J., Benz, C.C., Levine, D.A., Lee, A.V. and Omberg, L., 2018. An integrated TCGA pan-cancer clinical data resource to drive high-quality survival outcome analytics. Cell, 173(2), pp.400-416.
    """
    
    cell_df = pd.read_excel(fpath, sheet_name='TCGA-CDR', index_col=0)

    if verbose:
        print("{} patients with TCGA clinical data".format(cell_df.shape[0]))

    # set index
    # bcr_patient_barcode should be unique for each person
    assert cell_df.shape[0] == len(np.unique(cell_df['bcr_patient_barcode']))
    cell_df = cell_df.set_index('bcr_patient_barcode')

    if subtype is not None:
        # subset to this subype
        cell_df = cell_df.query("type=='{}'".format(subtype.upper()))

        if verbose:
            print("{} patients with TCGA clinical data in {}".
                  format(cell_df.shape[0], subtype))

    # subset out columns we want
    cols = {}
    cols['survival_event'] = ['OS', 'DSS', 'DFI', 'PFI']
    cols['survival_time'] = [c + '.time' for c in cols['survival_event']]
    cols['cts_feats'] = ['age_at_initial_pathologic_diagnosis']
    cols['ordinal_feats'] = ['ajcc_pathologic_tumor_stage']
    cols['cat_feats'] = ['gender', 'race', 'histological_type']

    if subtype is None:
        cols['other'] = ['type']
    else:
        cols['other'] = []

    cols_we_want = np.concatenate(list(cols.values()))
    patient_clinical_df = cell_df[cols_we_want]

    # Handle NaNs
    feat_maps = {'ajcc_pathologic_tumor_stage': {'[Discrepancy]': 'NaN',
                                                 '[Not Available]': 'NaN'},

                 'race': {'[Not Available]': 'NaN',
                          '[Not Evaluated]': 'NaN',
                          },

                 'histological_type': {'[Not Available]': 'NaN'}
                 }

    patient_clinical_df.replace(to_replace=feat_maps, inplace=True)

    return patient_clinical_df, cols


def restrct_patients_to_avail_slides(patient_df, avail_slides, verbose=True):
    """
    Restricts a patient df to those patients with available slides. Also prints out summary.
    
    Parameters
    ----------
    patient_df: pd.DataFrame, (n_patients, n_features)
        The patient level data frame indexed by 'bcr_patient_barcode'
        
    avail_slides: list of str
        The available slides.
        
    Output
    ------
    patient_df, slides_with_patients, match_info
    
    patient_df: pd.DataFrame, (n_patients_avail, n_features)
        The patient df with available patients.
        
    slides_with_patients: list of str

    match_info: dict
    """
    # find patietns we have both dataset for
    patient_ids = []
    patient_ids_missing_slides = []

    slide_patients = []
    patient_slides = {}
    for slide in avail_slides:
        patient = get_participant_from_tcga_barcode(slide)
        slide_patients.append(patient)

        # add this slide to patient
        if patient in patient_slides:
            patient_slides[patient].append(slide)
        else:
            patient_slides[patient] = [slide]

    slide_patients_set = set(slide_patients)

    # got through patient clincial data
    slides_with_patients = []
    for patient_id in patient_df.index:

        if patient_id in slide_patients_set:
            # this patient has slides
            patient_ids.append(patient_id)
            slides_with_patients.extend(patient_slides[patient_id])
        else:
            # this patient has not slides
            patient_ids_missing_slides.append(patient_id)

    patient_ids_slides_no_clincal = slide_patients_set.\
        difference(patient_df.index.values)

    if verbose:
        print("{} patients have both cinical data and slides".
              format(len(patient_ids)))
        print("{} total participants have slides".
              format(len(slide_patients_set)))
        print("{} patients have cinical data, but no slides".
              format(len(patient_ids_missing_slides)))
        print("{} patients have slides, but no clincal data".
              format(len(patient_ids_slides_no_clincal)))
        print("{} slides have patient data".
              format(len(slides_with_patients)))

    match_info = {'slides_and_clinical': patient_ids,
                  'all_slide_patients': slide_patients,
                  'clincal_no_slide': patient_ids_missing_slides,
                  'slide_no_clinical': patient_ids_slides_no_clincal
                  }
    
    return patient_df.loc[patient_ids, :], slides_with_patients, match_info


def broadcast_patient_to_slide(slide_names, patient_df):
    """
    Broadcasts patient level information to slide level information.
    
    Parameters
    ----------
    slide_names: list of str
        The names of the slides.
        
    patient_df: pd.DataFrame, (n_total_patients, n_features)
        The patient data.
        
    Output
    ------
    slide_df
    
    slide_df: pd.DataFrame, (n_slides_with_data, n_features)
    """

    # add patient id as a colum
    assert len(patient_df.index.name) > 0
    patient_df = patient_df.copy()
    patient_df[patient_df.index.name] = patient_df.index.values

    # broadcast patient level data to slide level data
    slide_df = {}
    for slide_name in slide_names:
        # get participant id from slid ename
        participant_id = get_participant_from_tcga_barcode(slide_name)

        # get the patient info for this participant
        if participant_id in patient_df.index:
            slide_df[slide_name] = patient_df.loc[participant_id]

    slide_df = pd.DataFrame(slide_df).T
    
    return slide_df
