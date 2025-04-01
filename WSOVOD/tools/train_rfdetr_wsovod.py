#!/usr/bin/env python
"""
使用RF-DETR作为主干网络和区域提议生成器训练WSOVOD模型。
"""
import os
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    verify_results,
)

from wsovod.config import add_wsovod_config
from wsovod.engine.trainer import WSLTrainer

def setup(args):
    """
    创建配置并执行基本的设置。
    """
    cfg = get_cfg()
    add_wsovod_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    """
    主函数
    """
    cfg = setup(args)

    if args.eval_only:
        model = WSLTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = WSLTrainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    # 创建Trainer
    trainer = WSLTrainer(cfg)
    # 加载权重
    trainer.resume_or_load(resume=args.resume)
    # 开始训练
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    
    # 设置一些环境变量
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids if args.gpu_ids else "0"
    
    # 启动分布式训练
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    ) 