import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.layers import batched_nms
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage

import sys
import os

# 添加RF-DETR库到系统路径
rf_detr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../rf-detr"))
if rf_detr_path not in sys.path:
    sys.path.append(rf_detr_path)

from rfdetr.models.transformer import Transformer, build_transformer, gen_encoder_output_proposals
from rfdetr.models.lwdetr import MLP
from rfdetr.util.box_ops import box_cxcywh_to_xyxy

@PROPOSAL_GENERATOR_REGISTRY.register()
class RFDETRProposalGenerator(nn.Module):
    """使用RF-DETR生成区域提议的模块"""
    
    @classmethod
    def from_config(cls, cfg, input_shape):
        """从配置构建RFDETRProposalGenerator"""
        in_features = cfg.MODEL.RFDETR.IN_FEATURES
        
        # 构建Transformer
        transformer = build_transformer({
            'hidden_dim': cfg.MODEL.RFDETR.HIDDEN_DIM,
            'sa_nheads': cfg.MODEL.RFDETR.SA_NHEADS,
            'ca_nheads': cfg.MODEL.RFDETR.CA_NHEADS,
            'num_queries': cfg.MODEL.RFDETR.NUM_QUERIES,
            'dropout': cfg.MODEL.RFDETR.DROPOUT,
            'dim_feedforward': cfg.MODEL.RFDETR.DIM_FEEDFORWARD,
            'dec_layers': cfg.MODEL.RFDETR.DEC_LAYERS,
            'num_feature_levels': len(cfg.MODEL.RFDETR.PROJECTOR_SCALE),
            'dec_n_points': cfg.MODEL.RFDETR.DEC_N_POINTS,
            'lite_refpoint_refine': cfg.MODEL.RFDETR.LITE_REFPOINT_REFINE,
            'group_detr': cfg.MODEL.RFDETR.GROUP_DETR,
            'two_stage': cfg.MODEL.RFDETR.TWO_STAGE,
            'decoder_norm': cfg.MODEL.RFDETR.DECODER_NORM,
            'bbox_reparam': cfg.MODEL.RFDETR.BOX_REPARAM,
        })
        
        # 检查输入形状
        for f in in_features:
            assert f in input_shape, f"找不到特征层: {f}"
            
        return {
            "in_features": in_features,
            "transformer": transformer,
            "num_queries": cfg.MODEL.RFDETR.NUM_QUERIES,
            "num_classes": cfg.MODEL.RFDETR.NUM_CLASSES,
            "hidden_dim": cfg.MODEL.RFDETR.HIDDEN_DIM,
            "box_reparam": cfg.MODEL.RFDETR.BOX_REPARAM,
            "two_stage": cfg.MODEL.RFDETR.TWO_STAGE,
            "batch_size_per_image": cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.RPN.POSITIVE_FRACTION,
            "pre_nms_topk": (cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, cfg.MODEL.RPN.PRE_NMS_TOPK_TEST),
            "post_nms_topk": (cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST),
            "nms_thresh": cfg.MODEL.RPN.NMS_THRESH,
            "min_box_size": cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE,
            "loss_weight": {
                "loss_rpn_cls": cfg.MODEL.RPN.LOSS_WEIGHT,
                "loss_rpn_loc": cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT * cfg.MODEL.RPN.LOSS_WEIGHT,
            },
        }
    
    def __init__(
        self,
        in_features,
        transformer,
        num_queries,
        num_classes,
        hidden_dim,
        box_reparam=True,
        two_stage=True,
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_topk=(12000, 6000),
        post_nms_topk=(2000, 1000),
        nms_thresh=0.7,
        min_box_size=0,
        loss_weight=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.transformer = transformer
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.box_reparam = box_reparam
        self.two_stage = two_stage
        
        # 初始化参考点嵌入和查询特征
        self.refpoint_embed = nn.Embedding(num_queries, 4)  # [cx, cy, w, h]
        nn.init.constant_(self.refpoint_embed.weight.data, 0)
            
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        
        # 初始化分类和边界框预测头
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        # 初始化为小偏差以获得更好的训练稳定性
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
            
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        
        # WSOVOD兼容参数
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.nms_thresh = nms_thresh
        self.min_box_size = min_box_size
        self.loss_weight = loss_weight if loss_weight else {
            "loss_rpn_cls": 1.0,
            "loss_rpn_loc": 1.0,
        }
        
    def forward(self, images, features, gt_instances=None):
        """前向传播并产生区域提议"""
        # 从特征字典中提取特征列表
        feature_list = [features[f] for f in self.in_features]
        
        # 创建RF-DETR所需的位置编码和掩码
        srcs = feature_list
        masks = [
            F.interpolate(
                torch.zeros(
                    (images.tensor.shape[0], 1, images.tensor.shape[2], images.tensor.shape[3]), 
                    device=images.tensor.device
                ), 
                size=feature.shape[-2:]).squeeze(1).to(torch.bool)
            for feature in feature_list
        ]
        
        # 获取参考点和查询特征
        refpoint_embed_weight = self.refpoint_embed.weight
        query_feat_weight = self.query_feat.weight
        
        # 使用transformer进行前向传播
        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, None, refpoint_embed_weight, query_feat_weight)
        
        # 解码输出
        if self.box_reparam:
            outputs_coord_delta = self.bbox_embed(hs)
            outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
            outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
            outputs_coord = torch.cat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
        else:
            outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()
        
        # 计算分类分数
        outputs_class = self.class_embed(hs)
        
        # 使用最后一层解码器输出
        pred_boxes = outputs_coord[-1]  # 取最后一层结果 [B, Q, 4]
        pred_logits = outputs_class[-1]  # [B, Q, C]
        pred_objectness_logits = pred_logits.max(dim=2)[0]  # 最大类别分数作为objectness [B, Q]
        
        # 创建提议列表
        proposals = self._create_proposals_from_boxes(
            pred_boxes, pred_objectness_logits, images.image_sizes
        )
        
        # 计算损失
        if self.training and gt_instances is not None:
            losses = self._get_loss(
                pred_boxes, 
                pred_logits,
                gt_instances
            )
            # 如果是弱监督场景，加入ground truth到提议中
            proposals = add_ground_truth_to_proposals(gt_instances, proposals)
            return proposals, losses
        else:
            return proposals, {}
    
    def _create_proposals_from_boxes(self, boxes, objectness_logits, image_sizes):
        """从预测框创建提议实例"""
        # 把中心点坐标转换为左上右下坐标
        boxes = box_cxcywh_to_xyxy(boxes)  # 转换为xyxy格式
        boxes = boxes.detach()
        objectness_logits = objectness_logits.detach()
        
        proposals = []
        for boxes_per_image, logits_per_image, image_size in zip(
            boxes, objectness_logits, image_sizes
        ):
            # 裁剪到图像大小并转换为Boxes对象
            boxes_per_image = boxes_per_image * torch.tensor([image_size[1], image_size[0], 
                                                             image_size[1], image_size[0]],
                                                            dtype=torch.float32, device=boxes_per_image.device)
            boxes_per_image = Boxes(boxes_per_image)
            
            # 移除太小的框
            keep = boxes_per_image.nonempty(threshold=self.min_box_size)
            boxes_per_image = boxes_per_image[keep]
            logits_per_image = logits_per_image[keep]
            
            # 如果是测试阶段，则排序并取前K个
            if not self.training:
                pre_nms_topk = min(self.pre_nms_topk[1], logits_per_image.shape[0])
                topk_indices = torch.topk(logits_per_image, pre_nms_topk)[1]
                boxes_per_image = boxes_per_image[topk_indices]
                logits_per_image = logits_per_image[topk_indices]
            
            # NMS
            keep = batched_nms(
                boxes_per_image.tensor, 
                logits_per_image,
                torch.zeros_like(logits_per_image),  # 所有框都视为同一类
                self.nms_thresh
            )
            
            # 保留前K个
            keep = keep[: self.post_nms_topk[1] if not self.training else self.post_nms_topk[0]]
            boxes_per_image = boxes_per_image[keep]
            logits_per_image = logits_per_image[keep]
            
            proposals.append(
                Instances(
                    image_size,
                    proposal_boxes=boxes_per_image,
                    objectness_logits=logits_per_image,
                )
            )
        
        return proposals
    
    def _get_loss(self, pred_boxes, pred_logits, gt_instances):
        """计算损失函数"""
        # 弱监督情况下，主要关注分类损失
        storage = get_event_storage()
        
        # 分类损失：使用图像级标签
        gt_labels = []
        for gt_per_image in gt_instances:
            batch_size = pred_logits.shape[1]
            labels = torch.zeros((batch_size, self.num_classes), 
                               dtype=torch.float32, device=pred_logits.device)
            if len(gt_per_image) > 0:
                # 对于图像级标签，对所有查询使用相同的标签
                image_classes = torch.unique(gt_per_image.gt_classes)
                for cls_id in image_classes:
                    if cls_id >= 0:  # 跳过背景类
                        labels[:, cls_id] = 1.0
            gt_labels.append(labels)
        
        gt_labels = torch.stack(gt_labels)
        
        # 计算分类损失
        objectness_loss = F.binary_cross_entropy_with_logits(
            pred_logits,
            gt_labels,
            reduction="none",
        )
        
        # 在弱监督场景下，边界框回归损失在早期阶段通常较小或不使用
        # 使用当前训练迭代比例缩放box损失
        cur_iter = storage.iter
        max_iter = storage.max_iter if hasattr(storage, 'max_iter') else 100000
        iter_ratio = min(1.0, cur_iter / (0.5 * max_iter))
        
        # 如果有真实框，计算框回归损失，否则设为零
        localization_loss = torch.tensor(0.0, device=pred_logits.device)
        
        normalizer = self.batch_size_per_image * len(gt_instances)
        losses = {
            "loss_rpn_cls": objectness_loss.sum() / normalizer,
            "loss_rpn_loc": localization_loss * iter_ratio / normalizer if iter_ratio > 0.2 else localization_loss * 0.0 / normalizer,
        }
        
        # 应用损失权重
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        
        return losses 