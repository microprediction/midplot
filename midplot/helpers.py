import typing
from midone import Attacker, HORIZON


def wrap(model_class):
    if not issubclass(model_class, Attacker):
        raise ValueError("Model must be a subclass of Attacker")
    def infer(
            stream: typing.Iterator[dict],
            hyper_parameters: dict = None,
            with_hyper_parameters_load: bool = False) -> typing.Iterator[float]:
        # Initialize the model
        model = model_class()

        # Required initial yield to signal system readiness
        yield

        # Process the stream
        for message in stream:
            prediction = model.tick_and_predict(message["x"], horizon=HORIZON)
            yield prediction

    return infer
