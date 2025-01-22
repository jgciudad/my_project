from pathlib import Path
import typer
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from my_project.model import MyAwesomeModel
from my_project.data import MyDataset


def visualize(model_checkpoint: Path, processed_data: Path = './data/processed', n_samples: int = 2000) -> None:
    """Visualize the model."""

    print("Visualizing {model_checkpoint}")
    
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))
    
    dataset = MyDataset()
    _, _, test_images, test_target = dataset.load(processed_data)  
    
    model.eval()
    for i in range(0, len(test_images), 32):
        images_batch = test_images[i:i + 32]
        
        output = model.forward(images_batch, visualize=True)
        if i == 0:
            all_outputs = output
        else:
            all_outputs = torch.cat((all_outputs, output), dim=0)
            

    tsne = TSNE(n_components=2, random_state=0)
    
    indices = torch.randperm(all_outputs.size(0))[:n_samples]
    sampled_outputs = all_outputs[indices]
    sampled_targets = test_target[indices]

    print("Running t-SNE")
    tsne_results = tsne.fit_transform(sampled_outputs.detach().numpy())
    
    plt.figure(figsize=(10, 7))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=sampled_targets, cmap='viridis', s=5)
    plt.colorbar()
    plt.title('t-SNE of Model Outputs')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.savefig('./reports/figures/tsne_plot.png')    
    

if __name__ == "__main__":
    typer.run(visualize)