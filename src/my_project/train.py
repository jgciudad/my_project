from pathlib import Path
import torch
import typer
from matplotlib import pyplot as plt

from my_project.model import MyAwesomeModel
from my_project.data import MyDataset


def train(lr: float = 1e-3, epochs: int = 5, batch_size: int = 32, processed_data: Path = Path('./data/processed'), testing: bool = False) -> None:
    """Train a model on MNIST."""
    print(testing)
    print(lr)

    model = MyAwesomeModel()
    
    dataset = MyDataset()
    train_images, train_target, _, _ = dataset.load(processed_data)  
      
    indices = torch.randperm(len(train_images)).tolist()
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    
    losses = []
    for e in range(epochs):
        print(f"Epoch {e}")
        
        epoch_losses = []
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            images_batch = train_images[batch_indices]
            targets_batch = train_target[batch_indices]
            
            optimizer.zero_grad()
            output = model(images_batch)
            loss = loss_fn(output, targets_batch)
                
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        losses.append(sum(epoch_losses) / len(epoch_losses))
        print(f"Loss: {losses[-1]}")
    
    print("Training complete")
    if testing:
        model_dir = Path("./tests/models")
    else:
        model_dir = Path("./models")
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pth")

    plt.plot(range(epochs), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    if testing:
        figure_dir = Path("./tests/reports")
    else:
        figure_dir = Path("./reports")
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_dir / 'training_loss_plot.png')        

if __name__ == "__main__":
    typer.run(train)