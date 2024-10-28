#%pip install --upgrade crunch-cli -q


#!crunch setup --notebook mid-one viper --token rZMdcHya7eLNfwUZkZz32FxIEWjdjHtfP2Ip0VLeKEQ95qNGrrzw9LaXFo5UA14w


#%pip install --upgrade endersgame -q


#%pip install --upgrade crunch-cli -q



import json
import os
import typing

import numpy
import pandas
import scipy.optimize as opt
from endersgame import HORIZON, EPSILON, Attacker
from tqdm.auto import tqdm
#EPSILON = 0.0025


import crunch

#crunch = crunch.load_notebook()

#x_train, x_test = crunch.load_streams()


#%load_ext autoreload
#%autoreload 2


#from snarimax import SNARIMAX


import matplotlib.pyplot as plt

def add(plt, pts = [], label = "Label", alpha=1., color = None):
    if len(pts) == 0:
        return
    xs = [pt[0] for pt in pts]
    ys = [pt[1] for pt in pts]
    # Plot actual data as scatter plot
    plt.scatter(x=xs, y=ys, label=label, alpha=alpha, color=color)

def plot(actual, directions_up, directions_down):
    # Plotting
    plt.figure(figsize=(12, 6))
    add(plt,  actual, color="b", label="Actual", alpha=0.1)
    add(plt,  directions_up, color="r", label="Detected Up", alpha=0.5)
    add(plt,  directions_down, color="g", label="Detected Down", alpha=0.5)
    

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Actual Data Points')
    plt.legend()
    plt.grid(True)
    plt.show()



import numpy as np

import numpy as np
class MovementDetector:
    def __init__(self, buffer_size=8):
        self.buffer_size = buffer_size
        self.buffer = []

    def tick(self, x):
        self.buffer.append(x)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        
        if len(self.buffer) == self.buffer_size:
            return self.detect()
        return None

    def detect_direction(self, direction):
        if len(self.buffer) < self.buffer_size:
            return None
        
        score = 0
        
        # Check for overall move (more lenient scoring)
        overall_change = (self.buffer[-1] - self.buffer[0]) / self.buffer[0]
        if (direction == "up" and overall_change < 0) or (direction == "down" and overall_change > 0):
            return None
        score_1 = min(100, abs(overall_change) * 30 * 10000)
        
        score += score_1
        
        # Check percentage of directional moves
        diffs = np.diff(self.buffer)
        directional_moves = np.sum((diffs > 0) if direction == "up" else (diffs < 0))
        move_percentage = (directional_moves / (self.buffer_size - 1)) * 100
        score_2 = move_percentage if move_percentage >= 75 else 0
        score += score_2
        
        # Check acceleration
        mid_point = self.buffer_size // 2
        first_half_moves = np.sum(diffs[:mid_point])
        second_half_moves = np.sum(diffs[mid_point:])
        score_3 = 0
        if (direction == "up" and first_half_moves > 0 and second_half_moves > 0) or \
           (direction == "down" and first_half_moves < 0 and second_half_moves < 0):
            acceleration_ratio = abs(second_half_moves / first_half_moves)
            if acceleration_ratio > 4:
                score_3 = min(100, (acceleration_ratio - 4) * 100)
                
        score += score_3
        
        score = score / 3
        if score >= 50:  # Threshold for detection
            self.buffer = []
            return direction
        return None

    def detect(self):
        up_result = self.detect_direction("up")
        if up_result:
            return up_result
        
        down_result = self.detect_direction("down")
        if down_result:
            return down_result
        
        return None


#num_stream = 0

from endersgame.accounting.pnl import Pnl

#score = 0.
#for num_stream, stream in enumerate(x_train):


#x_train, x_test = crunch.load_streams()


#%pip install git+https://github.com/microprediction/enderswidgets.git@main --force-reinstall -q


#from enderswidgets.replay import replay

#replay(x_test, HORIZON)


# First define the objective as negative total profit and test it



def get_parameter_file_path(model_directory_path: str):
    return os.path.join(model_directory_path, 'params.json')


def train(
    streams: typing.List[typing.Iterable[dict]],
    model_directory_path: str
):
    pass


# Here is how you would use it on the training data
#train(


def infer(
    stream: typing.Iterator[dict],
    model_directory_path: str
):
    # Instantiate your attacker
    detector = MovementDetector()

    
    # Signals to the system that your attacker is initialized and ready.
    yield  # Leave this here.

    for message in stream:
        p = detector.tick(message["x"])
        decision = 0.
        if p is not None and p == 'up':
            decision = 1.
        if p is not None and p == 'down':
            decision = -1.

        # Be sure to yield, even if the decision is zero.
        yield decision



#predictions = crunch.test()


#predictions


#mapping = {}
#for stream_id, stream in predictions.groupby("stream"):#print(mapping)


from datetime import datetime, timedelta
import pandas as pd

def create_matching_dataframe(streams, mapping):
    data = []
    for stream in streams:
        name = mapping[len(stream)]
        for d in stream:
            data.append({
                'stream': name,
                'value': d.get('x', None)  # Assuming 'y' is the value, use None if not present
            })
    
    df = pd.DataFrame(data)
    
    return df





#test = create_matching_dataframe(x_test, mapping)




def merge_and_process_dataframes(df1, df2, horizon):
    # Reset index of both dataframes to ensure we're working with a clean slate
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    
    # Ensure both dataframes have the same length
    assert len(df1) == len(df2), "DataFrames must have the same length"
    
    # Create a new index based on the order of df1
    new_index = df1.index
    
    # Assign this new index to both dataframes
    df1 = df1.set_index(new_index)
    df2 = df2.set_index(new_index)
    
   
    
    # Merge the dataframes
    merged_df = pd.concat([df1, df2], axis=1)
    
    # Remove duplicate 'stream' column if it exists in both dataframes
    if 'stream' in df1.columns and 'stream' in df2.columns:
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    
    # Calculate the new column
    merged_df['shifted_diff'] = (
        merged_df.groupby('stream')['value'].shift(-(horizon + 1)) - 
        merged_df.groupby('stream')['value'].shift(-1)
    )
     # Rename 'None' column to 'decision' in df1
    merged_df = merged_df.rename(columns={None: 'decision'})
    
    return merged_df



#merged = merge_and_process_dataframes(predictions, test, HORIZON)


#HORIZON


import pandas as pd

def calculate_pnl(df, epsilon):
    # Calculate PnL for each row
    df['pnl'] = df['decision'] * df['shifted_diff'] - (epsilon * (df['decision'] != 0))

    total_pnl = 0

    # Group by stream and calculate sum of PnL
    for stream, group in df.groupby('stream'):
        group_pnl = group['pnl'].sum()
        print(f"PnL for {stream}: {group_pnl:.6f}")
        total_pnl += group_pnl

    print(f"\nTotal PnL: {total_pnl:.6f}")



#calculate_pnl(merged, EPSILON)

