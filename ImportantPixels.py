# %config InlineBackend.figure_format = 'retina'
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T

import numpy as np
torch.set_grad_enabled(False)

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

attention = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
attention.eval()

def get_attention(im):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = attention(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    # use lists to store the outputs via up-values
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []

    hooks = [
        attention.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        attention.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[1])
        ),
        attention.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ),
    ]

    # propagate through the attention
    outputs = attention(img)

    for hook in hooks:
        hook.remove()

    # don't need the list anymore
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0]

    h, w = conv_features['0'].tensors.shape[-2:]

    att_scores_list = []
    upsample = torch.nn.Upsample(size = im.size[::-1])
    for idx in keep.nonzero():
        att_scores = dec_attn_weights[0, idx].view(1, h, w)
        att_scores = upsample(att_scores.unsqueeze(0)).squeeze(0).squeeze(0)
        att_scores = (att_scores - att_scores.min()) / (att_scores.max() - att_scores.min())
        att_scores_list.append(att_scores.detach().numpy())

    return att_scores_list
###############################################################################################
# get the distribution for important pixels
def importantPixels(im, boxes):
    att_scores_list = get_attention(im)

    importantPixels_list = [None for _ in range(len(boxes))]
    check = [False for _ in range(len(att_scores_list))]
    outlier_boxes = 0
    for j, box in enumerate(boxes):
        paired = False
        for i, att_scores in enumerate(att_scores_list):
            if check[i]:
                continue
            if att_scores[box[1]:box[3], box[0]:box[2]].max() > 0.9: # Not really good here!!!
                x = att_scores[box[1]:box[3], box[0]:box[2]]
                x = x.flatten()
                ###############################
                '''
                set the distribution so that the probs of important pixels are double the prob of the others
                '''
                # can set based on threshold
                # thres = (x.max() + x.min()) / 2
                # x[x >= thres] = 2
                # x[x < thres] = 1

                # or 50% of the largest set to double the rest
                argsort = np.argsort(x)
                x[argsort[:len(argsort) // 2]] = 1
                x[argsort[len(argsort) // 2:]] = 2
                ###############################
                x = x / x.sum()
                importantPixels_list[j] = x.copy()

                check[i] = True
                paired = True
                break
        if not(paired):
            outlier_boxes += 1


    if outlier_boxes > 0:
        for j, box in enumerate(boxes):
            if importantPixels_list[j] is None:
                importantPixels_list[j] = np.ones((box[3] - box[1]) * (box[2] - box[0])) / ((box[3] - box[1]) * (box[2] - box[0]))

    return importantPixels_list
###############################################################################################