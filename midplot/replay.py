import time

from midone import EPSILON, HORIZON
from midone.accounting.pnl import Pnl

from crunch.container import StreamMessage
from pydantic import BaseModel
from tqdm import tqdm

from midplot.accounting import AccountingDataVisualizer
from midplot.decision import DecisionManager
from midplot.streams import StreamPoint, Prediction
from midplot.visualization import TimeSeriesVisualizer, DataSelection

from typing import Union, Iterable, List, Dict, TypeVar, Any, Optional, Callable, Iterator
import numpy as np


class NoOpSink:
    def process(self, *args, **kwargs):
        pass
    def display(self):
        pass



def infer_nothing(
        stream: Iterator[dict],
        hyper_parameters = None,
        with_hyper_parameters_load = False
):
    yield  # Leave this here.

    for message in stream:
        yield 0.

class Scenario(BaseModel):
    selection: DataSelection = None
    decision: float = 0.0


class ScenarioResultCheck(BaseModel):
    success: List[Scenario] = []
    failed: List[Scenario] = []
    def __str__(self):
        return f"Success: {len(self.success)}, Failed: {len(self.failed)}"

class ReplayResults:
    def __init__(self):
        self.visualizers: Dict[int, TimeSeriesVisualizer] = {}
        self.accounting_visualizer = None
        self.decision_managers: List[DecisionManager] = []
        self.scores = {}
        self.total_score = 0.0
        self.scenario_idx = None
        self.elapsed = 0.

    def get_selection(self):
        for stream_id in self.visualizers:
            selection = self.visualizers[stream_id].selection
            if selection is not None:
                selection.stream_id = stream_id
                return selection
        return None


    def get_selected_points(self) -> List[float]:
        selection = self.get_selection()
        if selection is not None:
            return selection.data
        return []

    def check_scenarios(self, scenarios: List[Scenario], threshold: float = None) -> (bool, ScenarioResultCheck):
        self.scenario_idx = 0
        num_failed = 0
        result = ScenarioResultCheck()
        for scenario in scenarios:
            if scenario.selection is None:
                raise ValueError("Selection is None")
            decision_manager = self.decision_managers[self.scenario_idx]
            self.scenario_idx += 1
            if decision_manager is None:
                raise ValueError("Decision manager is None")
            check = decision_manager.match(scenario.decision)
            if check:
                result.success.append(scenario)
            if not check:
                num_failed += 1
                result.failed.append(scenario)
                if threshold is None:
                    return False, result
                else:
                    if num_failed > (1.- threshold)*len(scenarios):
                        return False, result
        return True, result

    def save_selected(self, expected_decision: float = 0.):
        selection = self.get_selection()
        if selection is None:
            print("No selection found")
            return
        decision_name = "up" if expected_decision > 0 else "down" if expected_decision < 0 else "none"
        scenario = Scenario(selection=selection, decision=expected_decision)
        # create scenarios folder if not exists
        import os
        if not os.path.exists("scenarios"):
            os.makedirs("scenarios")
        file = f"scenarios/scenario_{decision_name}_{selection.stream_id}_{selection.original_start_index}_{selection.original_end_index}.json"
        with open(file, 'w') as f:
            f.write(scenario.model_dump_json())


_replay_result = ReplayResults()

def get_replay_result():
    return _replay_result



class Movie(BaseModel):
    scenarios: List[Scenario]
    def streams(self):
        data = []
        for scenario in self.scenarios:
            data.append(scenario.selection.data)
        return data

def load_scenarios() -> Movie:
    from pathlib import Path
    scenarios = []
    for file in Path('scenarios').glob('*.json'):
        with open(file, 'r') as f:
            scenarios.append(Scenario.model_validate_json(f.read()))
    return Movie(scenarios=scenarios)


T = TypeVar('T', bound=Dict[str, Any])

def replay(streams: Union[Iterable[Iterable[T]], Iterable[T], Iterable[float]],
           horizon: int = HORIZON,
           epsilon: float = EPSILON,
           only_stream_ids: Optional[Union[int, Iterable[int]]] = None,
           start_index: int = 0,
           stop_index: int = None,
           update_frequency: int = 100,
           with_visualization: bool = False,
           with_accounting_visualizer: bool = False,
           dry: bool = False,
           hyper_parameters: Any = None,
           with_hyper_parameters_load: bool = False) -> ReplayResults:
    """
    Replay a set of streams, visualize the replay_result and return the total profit
    :param epsilon:
    :param stop_index:
    :param start_index:
    :param only_stream_ids:
    :param update_frequency:
    :param streams:
    :param horizon:
    :param with_visualization:
    :param with_accounting_visualizer:
    :param dry:
    :param hyper_parameters:
    :param with_hyper_parameters_load:
    :return: ReplayResults object containing visualizers and total score
    """
    try:
        from __main__ import infer
        if infer is None:
            print("Please define the 'infer' function in the main module: for debugging, showing no attacks.")
            infer = infer_nothing
    except ImportError:
        print("Please define the 'infer' function in the main module: for debugging, showing no attacks.")
        infer = infer_nothing

    try:
        # Record the start time
        start_time = time.time()
        ready_streams = process_streams(streams)
        if not dry:
            global _replay_result
            replay_result = ReplayResults()
            _replay_result = replay_result
        else:
            replay_result = ReplayResults()

        if with_accounting_visualizer:
            replay_result.accounting_visualizer = AccountingDataVisualizer(lambda : Pnl(epsilon=epsilon))
        else:
            replay_result.accounting_visualizer = NoOpSink()

        if isinstance(only_stream_ids, int):
            only_stream_ids = [only_stream_ids]

        for stream_id, stream in enumerate(ready_streams):
        # for stream_id, stream in enumerate(tqdm(ready_streams, desc="Processing streams")):
            if only_stream_ids and stream_id not in only_stream_ids:
                continue
            decision_manager = DecisionManager()

            pnl = Pnl()

            if with_visualization:
                viz = TimeSeriesVisualizer(horizon=horizon)
                replay_result.visualizers[stream_id] = viz
            else:
                viz = NoOpSink()

            prediction_generator = infer(stream, hyper_parameters=hyper_parameters, with_hyper_parameters_load=with_hyper_parameters_load)
            next(prediction_generator)

            for idx, data_point in enumerate(stream):
                if idx < start_index:
                    continue
                if stop_index is not None and idx >= stop_index:
                    break
                x = data_point['x']
                prediction = next(prediction_generator)
                data = StreamPoint(substream_id=str(stream_id), value=x, ndx=idx)
                pred = Prediction(value=prediction, ndx=idx+horizon, horizon=horizon)
                decision_manager.process(data, pred)
                replay_result.accounting_visualizer.process(data, pred)
                viz.process(data, pred)
                pnl.tick(data_point['x'], horizon, prediction)
                if idx % update_frequency == 0:
                    replay_result.accounting_visualizer.display()
                    viz.display()
            viz.display()
            replay_result.decision_managers.append(decision_manager)
            replay_result.accounting_visualizer.display()
            replay_result.scores[stream_id] = pnl.summary()['total_profit']
            replay_result.total_score += pnl.summary()['total_profit']
        # Record the end time
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        replay_result.elapsed = elapsed_time
        return replay_result

    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as e:
        print("Exception", e)
        # Print stack
        import traceback
        traceback.print_exc()


# @timeit
def process_streams(streams: Union[Iterable[Iterable[T]], Iterable[T], Iterable[float]]) -> Iterable[Iterable[T]]:
    def is_iterable(obj):
        return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))

    def wrap_in_message(x: float) -> Dict[str, float]:
        return {"x": float(x)}

    # Check if streams is iterable
    if not is_iterable(streams):
        raise ValueError("Input must be iterable")

    # Handle numpy arrays
    if isinstance(streams, np.ndarray):
        if streams.ndim == 1:
            return [[wrap_in_message(x) for x in streams]]
        elif streams.ndim == 2:
            return [[wrap_in_message(x) for x in row] for row in streams]
        else:
            raise ValueError("Numpy arrays with more than 2 dimensions are not supported")

    # Try to process as Iterable[Iterable[T]]
    try:
        result = []
        for item in streams:
            if is_iterable(item) and not isinstance(item, dict):
                sub_result = []
                for sub_item in item:
                    if isinstance(sub_item, (float, np.floating, int)):
                        sub_result.append(wrap_in_message(sub_item))
                    elif isinstance(sub_item, (dict, StreamMessage)):
                        return streams
                    else:
                        raise ValueError(f"Unsupported type in nested iterable: {type(sub_item)}")
                result.append(sub_result)
            else:
                raise TypeError  # Force to except block to handle as single stream
        return result
    except TypeError:
        # Process as single stream (Iterable[T] or Iterable[float])
        single_stream = []
        for item in streams:
            if isinstance(item, (float, np.floating, int)):
                single_stream.append(wrap_in_message(item))
            elif isinstance(item, dict):
                single_stream.append(item)
            else:
                raise ValueError(f"Unsupported type in stream: {type(item)}")
        return [single_stream]
