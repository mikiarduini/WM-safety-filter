import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import CLIPVisionModel, AutoModel, ViTMAEModel

class HFVisionEncoder(nn.Module):
    def __init__(self, model_name, model_type):
        super().__init__()
        self.model_type = model_type.lower()
        self.name = self.model_type 
        
        if self.model_type == "clip":
            load_kwargs = {"revision": "refs/pr/17"} if "openai" in model_name else {}
            self.model = CLIPVisionModel.from_pretrained(model_name, **load_kwargs)
            self.emb_dim = self.model.config.hidden_size
            self.patch_size = self.model.config.patch_size
            
            self.target_mean = [0.48145, 0.45782, 0.40821]
            self.target_std = [0.26862, 0.26130, 0.27577]
            self.renorm = T.Normalize(mean=self.target_mean, std=self.target_std)
            
        elif self.model_type == "siglip":
            self.model = AutoModel.from_pretrained(model_name).vision_model
            self.emb_dim = self.model.config.hidden_size
            self.patch_size = self.model.config.patch_size
            self.renorm = nn.Identity()
            
        elif self.model_type == "mae":
            self.model = ViTMAEModel.from_pretrained(model_name)
            
            # --- LA FIX CRUCIALE PER MAE ---
            self.model.config.mask_ratio = 0.0  # Impediamo di buttare via il 75% dell'immagine
            
            self.emb_dim = self.model.config.hidden_size
            self.patch_size = self.model.config.patch_size
            
            self.target_mean = [0.485, 0.456, 0.406]
            self.target_std = [0.229, 0.224, 0.225]
            self.renorm = T.Normalize(mean=self.target_mean, std=self.target_std)
            
        else:
            raise ValueError(f"Modello {model_type} non supportato.")

        self.latent_ndim = 2 
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            if self.model_type in ["clip", "mae"]:
                x_01 = (x * 0.5) + 0.5
                x_input = self.renorm(x_01)
            else:
                x_input = x

            outputs = self.model(x_input)
            
            if self.model_type == "clip":
                # Scartiamo il CLS token 
                emb = outputs.last_hidden_state[:, 1:, :]
                
            elif self.model_type == "mae":
                # MAE "mescola" le patch internamente. Dobbiamo usare 'ids_restore' 
                # per rimetterle nel loro esatto ordine spaziale sulla griglia!
                hidden_states = outputs.last_hidden_state[:, 1:, :]
                ids_restore = outputs.ids_restore.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
                emb = torch.gather(hidden_states, dim=1, index=ids_restore)
                
            elif self.model_type == "siglip":
                # SigLIP non ha il CLS token
                emb = outputs.last_hidden_state

        return emb
