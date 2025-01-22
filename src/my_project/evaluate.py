from pathlib import Path
import typer
import torch

from my_project.model import MyAwesomeModel
from my_project.data import MyDataset


def evaluate(model_checkpoint: str, batch_size: int = 32, processed_data: Path = './data/processed') -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(model_checkpoint)
    
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))
    
    dataset = MyDataset()
    _, _, test_images, test_target = dataset.load(processed_data)  
    
    model.eval()
    for i in range(0, len(test_images), batch_size):
        images_batch = test_images[i:i + batch_size]
        
        output = model(images_batch)
        if i == 0:
            all_outputs = output
        else:
            all_outputs = torch.cat((all_outputs, output), dim=0)
            
    _, predicted_labels = torch.max(all_outputs, 1)
    accuracy = (predicted_labels == test_target).sum().item() / len(test_target)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    
    
if __name__ == "__main__":
    typer.run(evaluate)