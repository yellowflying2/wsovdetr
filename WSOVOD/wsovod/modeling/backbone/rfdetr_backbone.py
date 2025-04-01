import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec

import sys
import os

# 添加RF-DETR库到系统路径
rf_detr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../rf-detr"))
if rf_detr_path not in sys.path:
    sys.path.append(rf_detr_path)

from rfdetr.models.backbone import build_backbone as build_rfdetr_backbone
from rfdetr.util.misc import NestedTensor

@BACKBONE_REGISTRY.register()
class RFDETRBackbone(Backbone):
    """RF-DETR backbone adapter for WSOVOD."""
    
    def __init__(self, cfg):
        super().__init__()
        
        # 从配置中构建RF-DETR参数
        encoder = cfg.MODEL.RFDETR.ENCODER
        vit_encoder_num_layers = cfg.MODEL.RFDETR.VIT_ENCODER_NUM_LAYERS
        pretrained_encoder = cfg.MODEL.RFDETR.PRETRAINED_ENCODER
        window_block_indexes = cfg.MODEL.RFDETR.WINDOW_BLOCK_INDEXES
        drop_path = cfg.MODEL.RFDETR.DROP_PATH
        hidden_dim = cfg.MODEL.RFDETR.HIDDEN_DIM
        out_feature_indexes = cfg.MODEL.RFDETR.OUT_FEATURE_INDEXES
        projector_scale = cfg.MODEL.RFDETR.PROJECTOR_SCALE
        use_cls_token = cfg.MODEL.RFDETR.USE_CLS_TOKEN
        position_embedding = cfg.MODEL.RFDETR.POSITION_EMBEDDING
        freeze_encoder = cfg.MODEL.RFDETR.FREEZE_ENCODER
        layer_norm = cfg.MODEL.RFDETR.LAYER_NORM
        resolution = cfg.MODEL.RFDETR.RESOLUTION
        target_shape = (resolution, resolution)
        rms_norm = cfg.MODEL.RFDETR.RMS_NORM
        backbone_lora = cfg.MODEL.RFDETR.BACKBONE_LORA
        force_no_pretrain = cfg.MODEL.RFDETR.FORCE_NO_PRETRAIN
        
        # 构建RF-DETR主干网络
        self.backbone, _ = build_rfdetr_backbone(
            encoder=encoder,
            vit_encoder_num_layers=vit_encoder_num_layers,
            pretrained_encoder=pretrained_encoder,
            window_block_indexes=window_block_indexes,
            drop_path=drop_path,
            out_channels=hidden_dim,
            out_feature_indexes=out_feature_indexes,
            projector_scale=projector_scale,
            use_cls_token=use_cls_token,
            hidden_dim=hidden_dim,
            position_embedding=position_embedding,
            freeze_encoder=freeze_encoder,
            layer_norm=layer_norm,
            target_shape=target_shape,
            rms_norm=rms_norm,
            backbone_lora=backbone_lora,
            force_no_pretrain=force_no_pretrain,
        )
        
        # 设置输出特征名称和通道维度
        self._out_features = cfg.MODEL.RFDETR.OUT_FEATURE_NAMES
        if not self._out_features:
            # 设置默认输出特征名
            self._out_features = ["p3", "p4", "p5", "p6"]
            
        # 获取尺度因子
        level2scalefactor = {"P3": 2.0, "P4": 1.0, "P5": 0.5, "P6": 0.25}
        scale_factors = [level2scalefactor[lvl] for lvl in projector_scale]
        
        # 设置特征通道和步长
        self._out_feature_channels = {name: hidden_dim for name in self._out_features}
        self._out_feature_strides = {
            name: int(1.0 / scale_factor) 
            for name, scale_factor in zip(self._out_features, scale_factors)
        }
        
    def forward(self, x):
        """前向传播函数"""
        # 将输入转换为RF-DETR所需的NestedTensor格式
        tensor_list = NestedTensor(
            x, 
            torch.zeros(x.shape[0], x.shape[2], x.shape[3], 
                      dtype=torch.bool, device=x.device)
        )
            
        # 通过RF-DETR主干网络
        features = self.backbone(tensor_list)
        
        # 转换输出格式为WSOVOD所需的字典
        outputs = {}
        for i, feature_name in enumerate(self._out_features):
            if i < len(features):
                # 提取特征和掩码，忽略掩码部分
                feat, _ = features[i].decompose()
                outputs[feature_name] = feat
                
        return outputs
    
    @property
    def size_divisibility(self):
        # DINOv2需要的最小分辨率对齐
        return 14  # patch size
        
    @property
    def output_shape(self):
        """返回输出特征的形状信息"""
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        } 