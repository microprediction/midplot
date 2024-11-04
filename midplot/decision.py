from typing import Dict

from algos.threshold import check_decision
from midplot.streams import StreamPoint, Prediction


class DecisionManager:
    def __init__(self):
        self.decisions: Dict[int, float] = {}

    def __repr__(self):
        return f"DecisionManager({self.decisions})"

    def process(self, pt: StreamPoint, pred: Prediction):
        if pred.value == 0.:
            return
        self.decisions[pt.ndx] = pred.value

    def match(self, decision: float) -> bool:
        return check_decision(decision, self.decisions.values())
