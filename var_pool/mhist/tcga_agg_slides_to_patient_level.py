from var_pool.mhist.tcga_clinical_data import get_participant_from_tcga_barcode
from var_pool.file_utils import get_file_names


def tcga_agg_slides_to_patient_level(fpaths, names=None):
    """
    Aggregates a list of slides to the patient level.

    Parameters
    ----------
    fpaths: list of str
        A list of slide file paths.

    names: None, list of str
        A list of patient names to subset to.

    Output
    ------
    patient2fpaths: dict of lists
        The file paths for the slides of each patient.
    """

    names = set(names) if names is not None else None

    patient2fpaths = {}
    for slide_fpath in fpaths:

        # get patient name from slide file path
        slide_fname = get_file_names(slide_fpath)
        patient_id = get_participant_from_tcga_barcode(slide_fname)

        # maybe skip this patient.
        if names is not None and patient_id not in names:
            continue

        # add this slide file path to this patient
        if patient_id in patient2fpaths.keys():
            patient2fpaths[patient_id].append(slide_fpath)
        else:
            patient2fpaths[patient_id] = [slide_fpath]

    return patient2fpaths
