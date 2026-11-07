"""Counterfactual leakage and seed regression test using actual IEEE-CIS transactions."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.strict_ieee import C, control, representation
import numpy as np
import pandas as pd


def main(output=None):
    # Exercise the production representation on a small chronological slice.
    tx = pd.read_csv(C.RAW_DIR/'train_transaction.csv', nrows=1200)
    identity = pd.read_csv(C.RAW_DIR/'train_identity.csv')
    df = tx.merge(identity, on=C.ID_COL, how='left', validate='one_to_one').sort_values(C.TIME_COL)
    df = df.reset_index(drop=True)
    mask = (df[C.TIME_COL] < df[C.TIME_COL].quantile(.7)).to_numpy()
    if not df[C.TARGET].isin([0, 1]).all() or not mask.any() or mask.all():
        raise ValueError('Expected binary labels and non-empty training and held-out groups')
    label_only = df.copy()
    label_only.loc[~mask, C.TARGET] = 1 - df.loc[~mask, C.TARGET]
    pd.testing.assert_frame_equal(df.drop(columns=C.TARGET), label_only.drop(columns=C.TARGET))
    np.testing.assert_array_equal(df[C.TARGET] != label_only[C.TARGET], ~mask)
    altered = label_only.copy()
    altered.loc[~mask, 'TransactionAmt'] += 10000
    altered.loc[~mask, 'addr1'] = df.loc[mask, 'addr1'].mode()[0]
    altered.loc[~mask, 'P_emaildomain'] = 'future-only-domain'
    params = dict(C.SAGE_PARAMS, max_epochs=2)
    x, z = representation(df, mask, 201, params)
    xl, zl = representation(label_only, mask, 201, params)
    if x.shape != xl.shape or z.shape != zl.shape or z.shape[0] != len(df):
        raise AssertionError('Representation dimensions changed')
    for array in (x, xl, z, zl):
        if not np.isfinite(array).all():
            raise ValueError('Non-finite representation')
    np.testing.assert_array_equal(x, xl)
    np.testing.assert_array_equal(z, zl)
    xa, za = representation(altered, mask, 201, params)
    np.testing.assert_array_equal(x[mask], xa[mask])
    np.testing.assert_array_equal(z[mask], za[mask])
    np.testing.assert_array_equal(control(z, mask, 201)[mask], control(za, mask, 201)[mask])
    if np.array_equal(z[~mask], za[~mask]):
        raise AssertionError('Perturbation failed to change any held-out representation')
    _, other = representation(df, mask, 202, params)
    if np.array_equal(z[mask], other[mask]):
        raise AssertionError('Graph initialization seed was ignored')
    report = dict(seed=201, max_epochs=params['max_epochs'], sample_rows=len(df),
                  historical_rows=int(mask.sum()), heldout_rows=int((~mask).sum()),
                  heldout_labels_flipped=int((df[C.TARGET] != label_only[C.TARGET]).sum()),
                  embedding_width=z.shape[1],
                  changed_historical_embeddings=int(np.any(z[mask] != zl[mask], axis=1).sum()),
                  changed_heldout_embeddings=int(np.any(z[~mask] != zl[~mask], axis=1).sum()),
                  max_absolute_embedding_difference=float(np.max(np.abs(z - zl))))
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    print(f'PASS: {mask.sum()} historical and {(~mask).sum()} held-out transactions; '
          'future perturbation preserves historical features, embeddings and controls; seed changes embeddings.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Write the aggregate label-only check as JSON')
    main(parser.parse_args().output)
