"""Render the stage comparison from published aggregate results."""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def render(directory, output):
    scores = pd.read_csv(directory/'stage_results.csv').set_index('model')
    blocks = pd.read_csv(directory/'stage_blocks.csv').set_index('block')
    ci = pd.read_csv(directory/'stage_intervals.csv').set_index(['model','reference','metric'])
    names = ['tabular','same_block','separated']
    if list(scores.index)!=names or list(blocks.index)!=['early','middle','test']:
        raise ValueError('Unexpected fitted models or split')
    if blocks.loc['early','end'] >= blocks.loc['middle','start'] or blocks.loc['middle','end'] >= blocks.loc['test','start']:
        raise ValueError('Blocks should be strictly ordered')
    if not scores[['auc','ap']].apply(lambda x:x.between(0,1).all()).all() or not scores.retained_three.between(0,3).all():
        raise ValueError('Invalid discrimination or overlap')
    for (model,ref,metric),row in ci.iterrows():
        np.testing.assert_allclose(scores.loc[model,metric]-scores.loc[ref,metric],row.difference,atol=1e-12)
        if row.lower>row.upper or row.block_days!=7 or row.draws!=1000:
            raise ValueError('Invalid block interval')
    lines=[r'\begin{tabular}{lrrr}',r'\toprule',
           r'Graph training period & AUC & Average precision & Fields retained (of 3) \\',r'\midrule']
    for name,label in zip(names,['Tabular reference','Same period as tree','Earlier period']):
        row=scores.loc[name]
        lines.append(f'{label} & {row.auc:.4f} & {row.ap:.4f} & {row.retained_three:.2f} '+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}',''])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'results/strict_ieee/stages')
    parser.add_argument('--output',type=Path,default=ROOT/'submission/source/sections/strict_ieee_stages.tex')
    args=parser.parse_args()
    render(args.directory,args.output)
