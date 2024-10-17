import pandas as pd
import numpy as np

def process(predictions: pd.DataFrame, x_test, horizon):
    mapping = create_mapping(predictions)
    test = create_matching_dataframe(x_test, mapping, horizon)
    groups = extract_groups(predictions, test, horizon)
    return groups

def create_mapping(predictions: pd.DataFrame) -> dict:
    mapping = {}
    for stream_id, stream in predictions.groupby("stream"):
        num_points = len(stream)
        mapping[num_points] = stream_id
    return mapping

def create_matching_dataframe(streams, mapping, horizon):
    data = []
    accumulate = []
    num_streams = 0
    for stream in streams:
        accumulate.append(stream)
        num_points = sum(len(s) for s in accumulate)
        if num_points not in mapping:
            continue
        name = mapping[num_points]
        num_streams += 1
        for acc in accumulate:
            for i, d in enumerate(acc):
                ignore_break = i + horizon >= len(acc)
                data.append({
                    'stream': name,
                    'value': d.get('x', None),
                    'ignore': ignore_break
                })
        accumulate = []
    if num_streams != len(mapping):
        raise ValueError("Mismatched number of streams")
    df = pd.DataFrame(data)
    return df

def extract_groups(predictions, tests, horizon):
    # Ensure both dataframes have the same length
    assert len(predictions) == len(tests), "DataFrames must have the same length"
    preds = {}
    for stream, group in predictions.groupby('stream'):
        preds[stream] = group

    out = []
    for stream, group in tests.groupby('stream'):
        if stream not in preds:
            continue
        pred = preds[stream]
        assert len(group) == len(pred), "DataFrames must have the same length"
        merged_df = pd.concat([group.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
        group = merged_df.reset_index(drop=True)  # Reset index for each group
        group['shifted_diff'] = group['value'].shift(-horizon) - group['value']
        group.loc[group['ignore'], 'shifted_diff'] = np.nan  # Set shifted_diff to NaN where ignore is True
        out.append((stream, group))
    return out

def calculate_pnl(groups, epsilon) -> float:
    total_pnl = 0

    for stream, df in groups:
        df['pnl'] = (df['prediction'] ) * (df['shifted_diff'] - epsilon)
        group_pnl = df['pnl'].sum()
        print(f"Stream {stream}: PnL = {group_pnl}")
        total_pnl += group_pnl

    return total_pnl

def pnl(predictions: pd.DataFrame, x_test, horizon, epsilon) -> float:
    groups = process(predictions, x_test, horizon)
    return calculate_pnl(groups, epsilon)
