from midone import EPSILON, HORIZON
from midone.accounting.pnl import Pnl

from crunch.container import StreamMessage

from midplot.accounting import AccountingDataVisualizer
from midplot.streams import StreamPoint, Prediction
from midplot.visualization import TimeSeriesVisualizer

from typing import Union, Iterable, List, Dict, TypeVar, Any, Optional, Callable
import numpy as np

T = TypeVar('T', bound=Dict[str, Any])

def process_streams(streams: Union[Iterable[Iterable[T]], Iterable[T], Iterable[float]]) -> List[List[T]]:
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
                        sub_result.append(sub_item)
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

class NoOp:
    def process(self, *args, **kwargs):
        pass
    def display(self):
        pass



class ReplayResults:
    def __init__(self):
        self.scores = {}
        self.visualizers: Dict[int, TimeSeriesVisualizer] = {}
        self.accounting_visualizer = None
        self.total_score = 0.0


    def get_selected_points(self) -> List[tuple]:
        for stream_id in self.visualizers:
            if len(self.visualizers[stream_id].selected_data) > 0:
                return [pt[1] for pt in self.visualizers[stream_id].selected_data]
        return []


def replay(streams: Union[Iterable[Iterable[T]], Iterable[T], Iterable[float]],
           horizon: int = HORIZON,
           epsilon: float = EPSILON,
           only_stream_ids: Optional[List[int]] = None,
           start_index: int = 0,
           stop_index: int = None,
           update_frequency: int = 50,
           with_visualization: bool = False,
           with_accounting_visualizer: bool = False) -> ReplayResults:
    """
    Replay a set of streams, visualize the results and return the total profit
    :param epsilon:
    :param stop_index:
    :param start_index:
    :param only_stream_ids:
    :param update_frequency:
    :param streams:
    :param horizon:
    :param with_visualization:
    :param with_accounting_visualizer:
    :return: ReplayResults object containing visualizers and total score
    """
    try:
        from __main__ import infer
    except ImportError:
        print("Please define the 'infer' function in the main module.")
        return None

    ready_streams = process_streams(streams)
    results = ReplayResults()

    if with_accounting_visualizer:
        results.accounting_visualizer = AccountingDataVisualizer(lambda : Pnl(epsilon=epsilon))
    else:
        results.accounting_visualizer = NoOp()

    if isinstance(only_stream_ids, int):
        only_stream_ids = [only_stream_ids]

    for stream_id, stream in enumerate(ready_streams):
        if only_stream_ids and stream_id not in only_stream_ids:
            continue
        print(f"Processing stream {stream_id}")
        pnl = Pnl()

        if with_visualization:
            viz = TimeSeriesVisualizer()
            results.visualizers[stream_id] = viz
        else:
            viz = NoOp()

        prediction_generator = infer(stream, horizon)
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
            results.accounting_visualizer.process(data, pred)
            viz.process(data, pred)
            pnl.tick(data_point['x'], horizon, prediction)
            if idx % update_frequency == 0:
                results.accounting_visualizer.display()
                viz.display()
        viz.display()
        results.accounting_visualizer.display()
        results.scores[stream_id] = pnl.summary()['total_profit']
        print("Profit", stream_id, pnl.summary()['total_profit'])
        results.total_score += pnl.summary()['total_profit']

    return results
