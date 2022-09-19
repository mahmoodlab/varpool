"""
Preprocessing for discrete survival model
1) make train/val/test split for slides, keeping patients in the same group
2) save slide survival response data frame. This has columns: ['sample_id', 'time_bin', 'survival_time', 'censorship']
"""
import os
import argparse
import numpy as np
from joblib import dump

from var_pool.processing.data_split import train_test_val_split
from var_pool.mhist.tcga_clinical_data import load_patient_data_from_cell_df,\
    restrct_patients_to_avail_slides, broadcast_patient_to_slide
from var_pool.processing.discr_surv_utils import get_discrete_surv_bins
from var_pool.file_utils import find_fpaths
from var_pool.file_utils import get_file_names

parser = argparse.\
    ArgumentParser(description='Makes the response data frame discritized survial model.')

parser.add_argument('--tcga_clincal_fpath', type=str,  required=True,
                    help='Where the TCGA clincal data frame is saved.'
                         'This should be Table S1 from (Liu et al, 2018)')

parser.add_argument('--feats_dir', type=str,  required=True,
                    help='Directory containing the slide features as .pt or .h5 files. Used to subset patients who we have data form.')

parser.add_argument('--save_dir', type=str,  required=True,
                    help='Where the response data frame should be stored.')

parser.add_argument('--subtype', type=str, default='brca',
                    help='Which cancer subtype.')

parser.add_argument('--endpoint', type=str, default='pfi',
                    choices=['os', 'pfi', 'dfi', 'dss'],
                    help="Survival endpoint."
                         "os = Overall Survival,"
                         "pfi = Progression Free Interval,"
                         "dfi = disease free interval,"
                         "dss = Disease Specific Survival.")

parser.add_argument('--bag_level', type=str, default='patient',
                    choices=['patient', 'slide'],
                    help='Are the bags at the slide level or patient level; for the latter we concatenated the bags for each slide for a patient.')


parser.add_argument('--prop_train', type=float, default=0.7,
                    help='The proportion of samples that go in the training dataset. The remaining samples are split evenly between the validation and test sets.')

parser.add_argument('--no_test_split',
                    action='store_true', default=False,
                    help='Only make train/val splits, not test split.')

parser.add_argument('--seed', type=int, default=1,
                    help='The random seed for splitting the data.')

parser.add_argument('--n_bins', type=int, default=4,
                    help='Number of bins for survival time.')

args = parser.parse_args()


os.makedirs(args.save_dir, exist_ok=True)
response_fpath = os.path.join(args.save_dir, 'discr_survival.csv')

other_data_fpath = os.path.join(args.save_dir, 'other_data')


#################################
# Load and process patient data #
#################################

# column names with survival
event_col = args.endpoint.upper()
time_col = event_col + '.time'
censor_col = 'censorship'
patient_id_name = 'patient_id'

# load patient clincal data
patient_clincal_df, col_types = \
    load_patient_data_from_cell_df(fpath=args.tcga_clincal_fpath,
                                   subtype=args.subtype,
                                   verbose=False)
print("Loaded clinical data for {} patients from {}".
      format(patient_clincal_df.shape[0], args.subtype))

# drop patients who are missing this survival data
nan_mask = patient_clincal_df[event_col].isna() | \
     patient_clincal_df[time_col].isna()
patient_clincal_df = patient_clincal_df[~nan_mask]
print("Dropping {} patients with missing survival data, left with {} patients".
      format(nan_mask.sum(), patient_clincal_df.shape[0]))

# slides we have data for
slide_fpaths = find_fpaths(folders=args.feats_dir, ext=['h5', 'pt'])
avail_slide_names = get_file_names(slide_fpaths)

# restrict patients to those we have slides for
patient_clincal_df, slides_with_clincal_data, match_info = \
    restrct_patients_to_avail_slides(patient_df=patient_clincal_df,
                                     avail_slides=avail_slide_names,
                                     verbose=True)

print("\n{} patients with both clincal data and slides".
      format(patient_clincal_df.shape[0]))

# create survival response df
patient_surv_df = patient_clincal_df[[time_col, event_col]].copy()
patient_surv_df[event_col] = patient_surv_df[event_col].astype(bool)
patient_surv_df['censorship'] = ~patient_surv_df[event_col]

print('{}% of patients censored'.
      format(100 * patient_surv_df[censor_col].mean()))

#####################
# Compute time bins #
#####################

# compute discrete surival response data
time_bin_idx, bins = \
    get_discrete_surv_bins(patient_surv_df,
                           n_bins=args.n_bins,
                           time_col=time_col,
                           censor_col=censor_col)

patient_surv_df['time_bin'] = np.array(time_bin_idx).astype(int)

######################################
# split patients into train/val/test #
######################################

# stratify on time bin X censorship status
# TODO: double check this is what we want to do
time_bin_X_censor = patient_surv_df['time_bin'].astype(str)\
    + '_X_' \
    + patient_surv_df[censor_col].astype(str)

if args.no_test_split:
    val_size = 1 - args.prop_train
    test_size = 0
else:
    val_size = (1 - args.prop_train) / 2
    test_size = (1 - args.prop_train) / 2

train_idxs, val_idxs, test_idxs = \
    train_test_val_split(n_samples=patient_surv_df.shape[0],
                         train_size=args.prop_train,
                         val_size=val_size,
                         test_size=test_size,
                         shuffle=True,
                         random_state=args.seed,
                         stratify=time_bin_X_censor)

patient_ids = patient_surv_df.index.values
patient_surv_df['split'] = None
patient_surv_df.loc[patient_ids[train_idxs], 'split'] = 'train'
patient_surv_df.loc[patient_ids[val_idxs], 'split'] = 'val'
patient_surv_df.loc[patient_ids[test_idxs], 'split'] = 'test'

# Format to standardized names
cols2keep = [time_col, censor_col, 'time_bin', 'split']
patient_surv_df = patient_surv_df[cols2keep]
patient_surv_df.rename(columns={time_col: 'survival_time',
                                censor_col: 'censorship'
                                },
                       inplace=True)


#############
# Save data #
#############

if args.bag_level == 'patient':

    # standardize name
    patient_surv_df.index.name = 'sample_id'

    patient_surv_df.to_csv(response_fpath)

    dump({'time_bins': bins,
          'patient_clincal_df': patient_clincal_df,
          'col_types': col_types,
          'slides_with_clincal_data': slides_with_clincal_data,
          'match_info': match_info},
         filename=other_data_fpath)


elif args.bag_level == 'slide':

    ###########################
    # Create slide level data #
    ###########################

    slide_surv_df =\
         broadcast_patient_to_slide(slide_names=slides_with_clincal_data,
                                    patient_df=patient_surv_df)

    # standardize names
    slide_surv_df.index.name = 'sample_id'
    slide_surv_df.rename(columns={patient_surv_df.index.name: patient_id_name})

    slide_surv_df.to_csv(response_fpath)

    dump({'patient_surv_df': patient_surv_df,
          'time_bins': bins,
          'patient_clincal_df': patient_clincal_df,
          'col_types': col_types,
          'slides_with_clincal_data': slides_with_clincal_data,
          'match_info': match_info},
         filename=other_data_fpath)
