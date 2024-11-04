import plotly.graph_objects as go
from midone import HORIZON
from plotly.subplots import make_subplots
from IPython.display import display
from ipywidgets import IntSlider, VBox
from typing import List, Optional, Dict

from pydantic import BaseModel

from midplot.streams import Prediction, StreamPoint

class DataSelection(BaseModel):
    stream_id: int = None
    data: List[float]
    original_start_index : int
    original_end_index : int

class TimeSeriesVisualizer:
    def __init__(self, max_select: int = 100, horizon: int = HORIZON):
        self.horizon = horizon
        self.max_select = max_select

        self.times: List[int] = []
        self.values: List[float] = []
        self.decision_times: List[int] = []
        self.decision_values: List[float] = []
        self.update_counter = 0

        self.selected_times: List[int] = []
        self.selected_values: List[float] = []
        self.clicked_index: Optional[int] = None
        self.n_select: int = max_select // 2  # Default to half of max_select

        self.fig = go.Figure()
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Realized', line=dict(color='lightgrey')))
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='markers', name='Selected', marker=dict(color='yellow', size=8)))

        self.shade_traces: Dict[int, int] = {}  # Maps decision times to trace indices

        self._setup_plot_style()
        self.fig_widget = go.FigureWidget(self.fig)
        self.fig_widget.data[0].on_click(self._on_click)

        self.slider = self._create_slider()

        self.widget = VBox([self.fig_widget, self.slider])
        display(self.widget)

    def _setup_plot_style(self):
        self.fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#2b2b2b',
            xaxis_title='Time',
            yaxis_title='Value',
            height=300,
            hovermode='x unified'
        )

    def _create_slider(self):
        slider = IntSlider(
            value=self.n_select,
            min=1,
            max=self.max_select,
            step=1,
            description='Selected points:',
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        slider.observe(self._on_slider_change, names='value')
        return slider

    def _on_slider_change(self, change):
        self.n_select = change['new']
        self._update_selection()

    def process(self, data: StreamPoint, pred: Prediction):
        self.times.append(data.ndx)
        self.values.append(data.value)
        if pred.value != 0:
            self.decision_times.append(data.ndx)
            self.decision_values.append(pred.value)
            self._add_shade(data.ndx, pred.value)

    def _add_shade(self, time: int, decision: float):
        color = 'rgba(0, 255, 0, 0.2)' if decision > 0 else 'rgba(255, 0, 0, 0.2)'
        trace = go.Scatter(
            x=[time, time],
            y=[min(self.values), max(self.values)],
            mode='lines',
            line=dict(color=color, width=10),
            showlegend=False,
            hoverinfo='none'
        )
        self.fig_widget.add_trace(trace)
        self.shade_traces[time] = len(self.fig_widget.data) - 1

    def display(self):
        with self.fig_widget.batch_update():
            # Update main data trace
            self.fig_widget.data[0].update(x=self.times, y=self.values)

            # Update selected points trace
            self.fig_widget.data[1].update(x=self.selected_times, y=self.selected_values)

            # Update shades
            y_min, y_max = min(self.values), max(self.values)
            for time, trace_index in self.shade_traces.items():
                self.fig_widget.data[trace_index].update(y=[y_min, y_max])

            self.fig_widget.update_layout(
                xaxis_range=[min(self.times), max(self.times)] if self.times else None,
                yaxis_range=[y_min, y_max] if self.values else None
            )

    def _on_click(self, trace, points, selector):
        if points.point_inds:
            self.clicked_index = points.point_inds[0]
            self._update_selection()

    def _update_selection(self):
        if self.clicked_index is not None:
            start_index = max(0, self.clicked_index - self.n_select + 1)
            self.selected_times = self.times[start_index:self.clicked_index + 1]
            self.selected_values = self.values[start_index:self.clicked_index + 1]
            self.display()

    @property
    def selection(self):
        return DataSelection(data=self.selected_values, original_start_index=self.selected_times[0], original_end_index=self.selected_times[-1])

    def clear(self):
        self.times.clear()
        self.values.clear()
        self.decision_times.clear()
        self.decision_values.clear()
        self.selected_times.clear()
        self.selected_values.clear()
        self.clicked_index = None
        self.shade_traces.clear()
        self.fig_widget.data = self.fig_widget.data[:2]  # Keep only main and selected traces
        self.display()

    def close(self):
        self.fig_widget.close()
