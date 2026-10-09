"""Write manuscript tables from freshly computed experiment results."""
from pathlib import Path
import json,pandas as pd,argparse
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'.qvvw2_generated'/'paper_tables');p.add_argument('--data-root',type=Path,default=ROOT/'.qvvw2_generated'/'plot_inputs');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
methods=['L2','Normalized-L2','Softplus-W2sq','Marginal-W2sq','UOT','Q-vvW2'];rows=[]
for m in methods:
 s=json.loads((args.data_root/'E03/original/P8/clean'/m/'summary.json').read_text())
 rows.append({k:s[k] for k in ['method','relative_model_error','correlation','common_std_residual']})
pd.DataFrame(rows).to_csv(args.output/'wave_clean_25.csv',index=False)
pd.read_csv(args.data_root/'E05/summary.csv')[['method','n_local_minima','success_tol_0p01','mean_abs_depth_error']].to_csv(args.output/'electromagnetic_depth.csv',index=False)
print('Paper table data written.')
