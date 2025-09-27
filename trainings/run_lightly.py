# ssl_train_with_monitor.py

from ultralytics import YOLO, settings
import lightly_train
import torch
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from torchvision import transforms
from PIL import Image
import os

# ---------------------- SSL Training ----------------------
def run_ssl_training():
    lightly_train.train(
        out="out/defect_Unlabeled_aug_lightly",
        data="datasets/defect_Unlabeled_aug/images/train",
        model="ultralytics/yolov12s.yaml",
        epochs=50,
        batch_size=32,
        overwrite=True,
        accelerator="auto",
    )

# ---------------------- Feature Monitor ----------------------
class FeatureMonitor:
    def __init__(self, model, layer_name="backbone", device="cuda"):
        self.model = model
        self.device = device
        self.features = []
        self.hook = None
        self.layer_name = layer_name

    def _hook_fn(self, module, input, output):
        pooled = torch.nn.functional.adaptive_avg_pool2d(output, 1).squeeze()
        self.features.append(pooled.detach().cpu())

    def register_hook(self):
        layer = getattr(self.model.model, self.layer_name)[-1]
        self.hook = layer.register_forward_hook(self._hook_fn)

    def clear(self):
        self.features = []

    def remove_hook(self):
        if self.hook:
            self.hook.remove()

    def visualize(self, title="Feature Embeddings"):
        if not self.features:
            print("No features collected.")
            return
        features = torch.cat(self.features).numpy()
        reducer = UMAP(n_components=2)
        embedding = reducer.fit_transform(features)
        plt.scatter(embedding[:, 0], embedding[:, 1], s=10)
        plt.title(title)
        plt.show()

    def log_stats(self):
        if not self.features:
            print("No features collected.")
            return
        norms = torch.norm(torch.cat(self.features), dim=1)
        print(f"📊 Feature Norms — Mean: {norms.mean():.4f}, Std: {norms.std():.4f}")

    

#-----------------------------------embedding_features------------------------------
    
def embed_features(embed_path, data_dir, checkpoint_path):
    
    output_embeddings = (embed_path)
    data_directory = (data_dir)
    checkpoint = (checkpoint_path)  
    

    lightly_train.embed(
    out=output_embeddings,                            
    data=data_directory,                                 
    checkpoint=checkpoint,
    format="torch",
)
    
# ------------------load_embeddings------------------------------

def load_embeddings(embed_path):
    embeddings = torch.load(embed_path)
    embedding_tensor = embeddings["embeddings"]    # Actual tensor
    filenames = embeddings["filenames"]            # List of image filenames
    print(embedding_tensor.shape)                  # [num_images, embedding_dim]
    return embedding_tensor, filenames



def visualize_embeddings(embeds, filenames, method="pca"):
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "umap":
        from umap import UMAP
        reducer = UMAP(n_components=2)
    else:
        raise ValueError("Choose 'pca' or 'umap'")

    reduced = reducer.fit_transform(embeds)
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=10)
    plt.title(f"{method.upper()} projection of embeddings")
    plt.show()


def cluster_embeddings(embeds, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeds)
    return labels

    

# ---------------------- Main ----------------------
if __name__ == "__main__":
    
    
    
    embeds, filenames = load_embeddings("out/defect_Unlabeled_aug_lightly/embeddings1.pt")
    
    visualize_embeddings(embeds.detach().cpu().numpy(), filenames, method="umap")
    
    
    labels = cluster_embeddings(embeds.detach().cpu().numpy(), n_clusters=10)
    
    
   
    
    
    
    
