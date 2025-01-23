import sys
import io
import os
import shutil
import re
import pytest

from my_project.train import train


def capture_output(func, *args, **kwargs):
    # Redirect sys.stdout to capture the output
    buffer = io.StringIO()
    sys.stdout = buffer
    
    try:
        func(*args, **kwargs)  # Call the function that prints
    finally:
        # Reset sys.stdout to its original state
        sys.stdout = sys.__stdout__
    
    # Return the captured output
    return buffer.getvalue()

@pytest.mark.parametrize("batch_size", [16, 32])
def test_training(batch_size: int) -> None:
    printed_output = capture_output(train, epochs=3, testing=True, batch_size=batch_size) 
    
    # Find all occurrences of "Loss: number"
    losses = re.findall(r"\nLoss: (\d+(\.\d+)?)", printed_output)

    # Get the first and last loss values
    first_loss = float(losses[0][0]) if losses else None
    last_loss = float(losses[-1][0]) if losses else None

    # Check that the first loss is higher than the last loss
    assert first_loss is not None and last_loss is not None, "Loss values could not be found in the output."
    assert first_loss > last_loss, f"Expected first loss to be higher than last loss, but got {first_loss} and {last_loss}."
        
    model_path = "./tests/models/model.pth"
    assert os.path.exists(model_path), f"Expected model file does not exist: {model_path}"

    # Remove the folder and its content
    shutil.rmtree(os.path.dirname(model_path))
    
    figure_path = "./tests/reports/training_loss_plot.png"
    assert os.path.exists(figure_path), f"Expected training curve does not exist: {model_path}"

    # Remove the folder and its content
    shutil.rmtree(os.path.dirname(figure_path))