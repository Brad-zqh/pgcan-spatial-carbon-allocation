"""Import authorized local results required by the original plotting scripts."""
from pathlib import Path
import argparse,shutil
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visuals',type=Path,required=True,help='Existing nature_visuals working directory')
    args=parser.parse_args()
    for name in ['source_data','experiments']:
        source=args.visuals/name
        if not source.is_dir():raise FileNotFoundError(source)
        shutil.copytree(source,ROOT/'analysis'/name,dirs_exist_ok=True)
    print('Imported locally into gitignored directories; set PGCAN_PROJECT_ROOT before plotting.')
if __name__=='__main__':main()
