"""Counterfactual leakage and seed regression test using actual IEEE-CIS transactions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.strict_ieee import C, control, representation
import numpy as np
import pandas as pd


def main():
    # A small chronological slice is enough to detect forbidden message paths.
    tx = pd.read_csv(C.RAW_DIR/'train_transaction.csv', nrows=1200)
    identity = pd.read_csv(C.RAW_DIR/'train_identity.csv')
    df = tx.merge(identity, on=C.ID_COL, how='left', validate='one_to_one').sort_values(C.TIME_COL)
    df = df.reset_index(drop=True)
    mask = (df[C.TIME_COL] < df[C.TIME_COL].quantile(.7)).to_numpy()
    altered = df.copy()
    altered.loc[~mask, C.TARGET] = 1 - altered.loc[~mask, C.TARGET]
    altered.loc[~mask, 'TransactionAmt'] += 10000
    altered.loc[~mask, 'addr1'] = df.loc[mask, 'addr1'].mode()[0]
    altered.loc[~mask, 'P_emaildomain'] = 'future-only-domain'
    params = dict(C.SAGE_PARAMS, max_epochs=2)
    x, z = representation(df, mask, 201, params)
    xa, za = representation(altered, mask, 201, params)
    np.testing.assert_array_equal(x[mask], xa[mask])
    np.testing.assert_array_equal(z[mask], za[mask])
    np.testing.assert_array_equal(control(z, mask, 201)[mask], control(za, mask, 201)[mask])
    if np.array_equal(z[~mask], za[~mask]):
        raise AssertionError('Perturbation failed to change any held-out representation')
    _, other = representation(df, mask, 202, params)
    if np.array_equal(z[mask], other[mask]):
        raise AssertionError('Graph initialization seed was ignored')
    print(f'PASS: {mask.sum()} historical and {(~mask).sum()} held-out transactions; '
          'future perturbation preserves historical features, embeddings and controls; seed changes embeddings.')


if __name__ == '__main__':
    main()
