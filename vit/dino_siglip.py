from urllib.request import urlopen
from PIL import Image
import timm
from torch import cat
import torch.nn as nn


class VisionEncoder(nn.Module):
    """Combined SigLIP-DINOv2 vision encoder.
    
    Params
    ------
    is_training: bool
        True if training both encoders.
    
    device: str
        Device to load models onto.
    
    Attributes
    ----------
    
    siglip: timm.models.vision_transformer.VisionTransformer
        SigLIP encoder.
    
    dino: timm.models.vision_transformer.VisionTransformer
        DINOv2 encoder.
    
    siglip_config: dict
        Config for SigLIP model.
        
    dino_config: dict
        Config for DINOv2 model.
        
    siglip_transform: torchvision.transforms.transforms.Compose
        Image transformations for SigLIP encoder.
        
    dino_transform: torchvision.transforms.transforms.Compose
        Image transformations for DINOv2 encoder."""
        
    def __init__(self, is_training=False, device="cuda:1"):
        
        super().__init__()
        
        self.is_training = is_training
        self.device = device
        
        self.siglip = timm.create_model(
            'vit_so400m_patch14_siglip_384',
            pretrained=True,
            num_classes=0, # Remover classifier
        ).to(self.device)

        self.dino = timm.create_model(
            'vit_large_patch14_dinov2.lvd142m',
            pretrained=True,
            num_classes=0, # Remover classifier
        ).to(self.device)

        if (not self.is_training):
            self.siglip, self.dino = self.siglip.eval(), self.dino.eval()
        # Initialize both vision encoders and set to evalutation mode unless training.

        self.siglip_config = timm.data.resolve_model_data_config(self.siglip)
        self.dino_config = timm.data.resolve_model_data_config(self.dino)

        self.siglip_transform = timm.data.create_transform(**self.siglip_config, is_training=self.is_training)
        self.dino_transform = timm.data.create_transform(**self.dino_config, is_training=self.is_training)
        # Create both encoder's configs and transforms
        
    def forward(self, img_path):
        """Run the forward pass.
        
        Params
        ------
        img_path: str
            Path to image file.
        
        Returns 
        -------
        
        logits: torch.Tensor
            Encoded image features."""
        
        img = Image.open(img_path)
        logits = cat((self.siglip(self.siglip_transform(img).to(self.device).unsqueeze(0)),
                   self.dino(self.dino_transform(img).to(self.device).unsqueeze(0))),
                   dim=1) # Size (batch_size, num_features)
        
        return logits
        


