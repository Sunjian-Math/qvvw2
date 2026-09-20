from pathlib import Path
import argparse
from run_e03 import print_quantitative_table, SCENARIOS

HERE=Path(__file__).resolve().parent

if __name__=='__main__':
    ap=argparse.ArgumentParser(description='Print saved E03 terminal accuracy and efficiency metrics without rerunning the inversion.')
    ap.add_argument('--P',type=int,default=8)
    ap.add_argument('--scenario',choices=SCENARIOS,default='clean')
    ap.add_argument('--outroot',default=str(HERE.parent/'results'))
    args=ap.parse_args()
    print_quantitative_table(args.outroot,args.P,args.scenario)
