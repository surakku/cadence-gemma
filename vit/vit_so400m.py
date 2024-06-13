from urllib.request import urlopen
from PIL import Image
import timm
import torch
import torch.nn as nn

image = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'vit_so400m_patch14_siglip_384',
    pretrained=True,
    num_classes=0,
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(image).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor

image2 = Image.open(urlopen(
    'https://www.pbs.org/wnet/nature/files/2021/08/mohan-moolepetlu-VUr5nmC1IM4-unsplash-scaled-e1628698069781.jpg'
))

output2 = model(transforms(image2).unsqueeze(0))

print(output)
print(output.shape)

cos = nn.CosineSimilarity(dim=1, eps=1e-8)
print(cos(output, output2))