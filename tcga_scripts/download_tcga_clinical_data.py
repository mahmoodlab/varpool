import argparse
import urllib.request
import os

parser = argparse.\
    ArgumentParser(description='Downloads the TCGA clinical data from (Liu et al, 2018). See https://www.cell.com/cell/fulltext/S0092-8674(18)30229-0.')

parser.add_argument('--save_dir',
                    type=str,
                    help="Where to save the results.")

parser.add_argument('--overwrite',
                    action='store_true', default=False,
                    help="Whether or not to overwrite the file if it exists.")

parser.add_argument('--merge_coadread_gbmlgg',
                    action='store_true', default=False,
                    help="Merge GBM-LGG and COAD-READ.")


args = parser.parse_args()


url = 'https://www.cell.com/cms/10.1016/j.cell.2018.02.052/attachment/bbf46a06-1fb0-417a-a259-fd47591180e4/mmc1'


fname = 'TCGA-CDR.xlsx'  # 'mmc1.xlsx'
os.makedirs(args.save_dir, exist_ok=True)
fpath = os.path.join(args.save_dir, fname)
if not os.path.exists(fpath) or args.overwrite:
    # TODO: this isn't working...
    urllib.request.urlretrieve(url=url, filename=fpath)

if args.merge_coadread_gbmlgg:
    raise NotImplementedError("TODO: add -- we did this manually... lets do it programatically")
