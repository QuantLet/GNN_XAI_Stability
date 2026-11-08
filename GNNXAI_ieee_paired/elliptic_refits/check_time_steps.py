"""Check every directed edge in the original Elliptic data, including unlabelled nodes."""
import argparse
from pathlib import Path

import pandas as pd


def check_time_steps(directory):
    nodes = pd.read_csv(directory / "elliptic_txs_features.csv", header=None,
                        usecols=[0, 1], names=["id", "step"]).set_index("id", verify_integrity=True)
    edges = pd.read_csv(directory / "elliptic_txs_edgelist.csv", usecols=["txId1", "txId2"])
    source = edges.txId1.map(nodes.step)
    destination = edges.txId2.map(nodes.step)
    if edges.empty or source.isna().any() or destination.isna().any():
        raise ValueError("The edge list is empty or contains an endpoint without a recorded time step.")
    print(f"{len(edges)} directed edges; {int(source.ne(destination).sum())} join different time steps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_directory", type=Path)
    check_time_steps(parser.parse_args().raw_directory)
