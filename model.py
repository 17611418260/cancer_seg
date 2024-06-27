import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator


# Create Mask-RCNN Model
def get_mask_rcnn(device, num_classes=1, pretrained=True):
    """
    >>> import torch
    >>> import torchvision
    >>> from torchvision.models.detection import MaskRCNN
    >>> from torchvision.models.detection.anchor_utils import AnchorGenerator
    >>>
    >>> # load a pre-trained model for classification and return
    >>> # only the features
    >>> backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    >>> # MaskRCNN needs to know the number of
    >>> # output channels in a backbone. For mobilenet_v2, it's 1280
    >>> # so we need to add it here
    >>> backbone.out_channels = 1280
    >>>
    >>> # let's make the RPN generate 5 x 3 anchors per spatial
    >>> # location, with 5 different sizes and 3 different aspect
    >>> # ratios. We have a Tuple[Tuple[int]] because each feature
    >>> # map could potentially have different sizes and
    >>> # aspect ratios
    >>> anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128),),
    >>>                                    aspect_ratios=((0.5, 1.0, 2.0),))
    >>>
    >>> # let's define what are the feature maps that we will
    >>> # use to perform the region of interest cropping, as well as
    >>> # the size of the crop after rescaling.
    >>> # if your backbone returns a Tensor, featmap_names is expected to
    >>> # be ['0']. More generally, the backbone should return an
    >>> # OrderedDict[Tensor], and in featmap_names you can choose which
    >>> # feature maps to use.
    >>> roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    >>>                                                 output_size=7,
    >>>                                                 sampling_ratio=2)
    >>>
    >>> mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    >>>                                                      output_size=14,
    >>>                                                      sampling_ratio=2)
    >>> # put the pieces together inside a MaskRCNN model
    >>> model = MaskRCNN(backbone,
    >>>                  num_classes=2,
    >>>                  rpn_anchor_generator=anchor_generator,
    >>>                  box_roi_pool=roi_pooler,
    >>>                  mask_roi_pool=mask_roi_pooler)
    >>> model.eval()
    >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    >>> predictions = model(x)
    """

    # ==========
    # model_ft = torchvision.models.detection.maskrcnn_resnet50_fpn(
    #     pretrained_backbone=pretrained, num_classes=num_classes, min_size=256, max_size=256
    # )

    # ==========
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )
    model_ft = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained_backbone=pretrained, min_size=256, max_size=256,
        rpn_anchor_generator=anchor_generator
    )

    # model_ft = torchvision.models.detection.maskrcnn_resnet50_fpn(
    #     pretrained, min_size=256, max_size=256,
    # )
    in_features = model_ft.roi_heads.box_predictor.cls_score.in_features
    print('in_features: ', in_features)

    model_ft.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model_ft.roi_heads.mask_predictor.conv5_mask.in_channels
    print('in_features_mask: ', in_features_mask)
    hidden_layer = 256
    model_ft.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    print('model_ft: ', model_ft)
    model_ft.to(device)

    # for param in model_ft.parameters():
    #     param.requires_grad = True

    return model_ft


# #
# model =
# state_dict = torch.load('')
# model.load_state_dict(state_dict)
# model.eval()
# #
# im = imread
# img = np.transpose(img, (2, 0, 1))
# x = torch.fromnumpy(img)
# predictions = model(x)




