from urllib.request import urlopen
from PIL import Image
import timm
from torch import cat
import torch.nn as nn
from torchvision.transforms import Compose, Resize



class VisionEncoder(nn.Module):
    """Combined SigLIP-DINOv2 vision encoder.
    
    Params
    ------
    is_training: bool
        True if training both encoders.
    
    device: str
        Device to load models onto.
        
    default_image_size: int
        Default input image size. 
    
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
        Image transformations for DINOv2 encoder.
        
    target_sizeL: tuple
        Final resolution of images."""
        
    def __init__(self, is_training=False, device="cuda:1", default_image_size=384):
        
        super().__init__()
        
        self.is_training = is_training
        self.device = device
        self.default_image_size = default_image_size
        
        self.siglip = timm.create_model(
            'vit_so400m_patch14_siglip_384',
            pretrained=True,
            num_classes=0, # Remove classifier
            img_size=self.default_image_size
        ).to(self.device)

        self.dino = timm.create_model(
            'vit_large_patch14_reg4_dinov2.lvd142m',
            pretrained=True,
            num_classes=0, # Remove classifier
            img_size=self.default_image_size
        ).to(self.device)

        if (not self.is_training):
            self.siglip, self.dino = self.siglip.eval(), self.dino.eval()
        # Initialize both vision encoders and set to evalutation mode unless training.

        self.siglip_config = timm.data.resolve_model_data_config(self.siglip)
        self.siglip_config["input_size"] = (3, self.default_image_size, self.default_image_size)
        
        self.dino_config = timm.data.resolve_model_data_config(self.dino)
        self.dino_config["input_size"] = (3, self.default_image_size, self.default_image_size)

        self.siglip_transform = timm.data.create_transform(**self.siglip_config, is_training=self.is_training)
        self.dino_transform = timm.data.create_transform(**self.dino_config, is_training=self.is_training)
        # Create both encoder's configs, override image size for larger image size and create transforms
        
        # Fix for SigLIP transforming to size larger than default
        assert isinstance(sl_resize_transform := self.siglip_transform.transforms[0], Resize)
        self.siglip_transform = Compose(
            [
                Resize(self.default_image_size, interpolation=sl_resize_transform.interpolation),
                *self.siglip_transform.transforms[1:],
            ]
        )
        
        
        # Set and ensure resize strategy
        assert isinstance(dino_resize_transform := self.dino_transform.transforms[0], Resize)
        assert isinstance(siglip_resize_transform := self.siglip_transform.transforms[0], Resize)

        target_size = (self.default_image_size, self.default_image_size)
        self.dino_transform = Compose(
            [
                Resize(target_size, interpolation=dino_resize_transform.interpolation),
                *self.dino_transform.transforms[1:],
            ]
        )
        self.siglip_transform = Compose(
            [
                Resize(target_size, interpolation=siglip_resize_transform.interpolation),
                *self.siglip_transform.transforms[1:],
            ]
        )
        

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


        logits = cat([self.dino(self.dino_transform(img).to(self.device).unsqueeze(0)),
                      self.siglip(self.siglip_transform(img).to(self.device).unsqueeze(0))], dim=1)
        return logits
        


