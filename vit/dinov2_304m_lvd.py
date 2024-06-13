from urllib.request import urlopen
from PIL import Image
import timm
import torch.nn as nn


img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'vit_large_patch14_dinov2.lvd142m',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor
print(f"Pooled: {output.shape}")
# or equivalently (without needing to set num_classes=0)

output = model.forward_features(transforms(img).unsqueeze(0))
print(f"Unpooled: {output.shape}")
# output is unpooled, a (1, 1370, 1024) shaped tensor

output = model.forward_head(output, pre_logits=True)
print(f"Raw: {output.shape}")
# output is a (1, num_features) shaped tensor   


img1 = Image.open("car2.jpg")
img2 = Image.open("cart2.jpg")

out1 = model(transforms(img1).unsqueeze(0))
out2 = model(transforms(img2).unsqueeze(0))

cos = nn.CosineSimilarity(dim=1, eps=1e-8)
print(cos(out1, out2))

