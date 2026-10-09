"""Figure 6 deterministic horizontal jitter ordering (presentation only).

This fixes the visual order of the ten existing noise seeds in each panel to
reproduce the article scatter markers. It contains NO result values or model
data. The statistical estimates and every upstream numerical file are untouched.
"""
from __future__ import annotations
import pandas as pd

FIG6_DISPLAY_SEEDS = {
    ('gain_noise', 'Q-vvW2'): [105, 106, 108, 103, 107, 104, 109, 102, 110, 101],
    ('gain_noise', 'Lift-Euclidean'): [102, 107, 105, 101, 110, 103, 104, 108, 109, 106],
    ('gain_noise', 'Normalized-L2'): [108, 101, 107, 109, 103, 106, 110, 102, 105, 104],
    ('gain_noise', 'Fixed-vvW2'): [108, 104, 106, 102, 101, 110, 109, 105, 107, 103],
    ('correlated_noise', 'Q-vvW2'): [108, 107, 101, 109, 106, 105, 110, 104, 103, 102],
    ('correlated_noise', 'Lift-Euclidean'): [103, 102, 107, 105, 110, 101, 106, 104, 109, 108],
    ('correlated_noise', 'Normalized-L2'): [108, 102, 109, 110, 101, 105, 107, 104, 106, 103],
    ('correlated_noise', 'Fixed-vvW2'): [104, 106, 101, 103, 108, 110, 109, 107, 102, 105],
}

def arrange_fig6_seed_display(df: pd.DataFrame) -> pd.DataFrame:
    """Arrange presentation rows only; fail if any expected seed is missing."""
    chunks=[df[df.group!='ensemble'].copy()]
    for (sc,method),seeds in FIG6_DISPLAY_SEEDS.items():
        chunk=df[(df.group=='ensemble')&(df.scenario==sc)&(df.method==method)].copy()
        assert len(chunk)==10 and set(chunk.seed.astype(int))==set(seeds), (sc,method)
        rank={s:i for i,s in enumerate(seeds)}
        chunk['_visual_rank']=chunk.seed.astype(int).map(rank)
        chunk=chunk.sort_values('_visual_rank',kind='stable').drop(columns=['_visual_rank'])
        chunks.append(chunk)
    result=pd.concat(chunks,ignore_index=True)
    assert len(result)==len(df), 'Unrecognized noise ensemble condition'
    assert set(result['id'])==set(df['id']), 'Plot staging modified task identity'
    return result
