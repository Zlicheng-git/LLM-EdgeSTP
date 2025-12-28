# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 15:33:23 2025

@author: HP
"""

import torch
import torch.nn as nn
#import torch.optim as optim
#from torch.amp import GradScaler
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
#import gc
#from codecarbon import EmissionsTracker
from modeling_llm4stp import LLM4STPModel
from Config_llm4stp import LLM4STPConfig
from data import DataProcessor
# from codecarbon import EmissionsTracker

# 评估指标
def haversine_distance(y_true, y_pred):
    lon1, lat1 = y_true[..., 0], y_true[..., 1]
    lon2, lat2 = y_pred[..., 0], y_pred[..., 1]
    
    lon1_rad = lon1 * (np.pi / 180.0)
    lat1_rad = lat1 * (np.pi / 180.0)
    lon2_rad = lon2 * (np.pi / 180.0)
    lat2_rad = lat2 * (np.pi / 180.0)
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = torch.sin(dlat / 2.0) ** 2 + \
        torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2.0) ** 2
    c = 2 * torch.asin(torch.sqrt(a))
    
    distance = c * 6371000  # 单位：米
    return distance

def haversine_FDE(y_true: torch.Tensor, y_pred: torch.Tensor, lat_min, lat_max, lon_min, lon_max) -> torch.Tensor:
    assert y_true.dim() == 2 and y_pred.dim() == 2, "输入必须是2D: [batch_size, seq_len * 2]"
    assert y_true.shape == y_pred.shape, "y_true和y_pred必须形状相同"
    assert y_true.shape[1] % 2 == 0, "最后一维必须是偶数"

    seq_len = y_true.shape[1] // 2
    y_true_reshaped = y_true.view(-1, seq_len, 2)
    y_pred_reshaped = y_pred.view(-1, seq_len, 2)

    y_true_last = y_true_reshaped[:, -1, :]
    y_pred_last = y_pred_reshaped[:, -1, :]

    y_true_denorm = min_max_denormalize(y_true_last, lat_min, lat_max, lon_min, lon_max)
    y_pred_denorm = min_max_denormalize(y_pred_last, lat_min, lat_max, lon_min, lon_max)

    distances = haversine_distance(y_true_denorm, y_pred_denorm)
    return distances.mean()

def haversine_ADE(y_true: torch.Tensor, y_pred: torch.Tensor, lat_min, lat_max, lon_min, lon_max,
                 sample_weight: torch.Tensor = None) -> torch.Tensor:
    assert y_true.dim() == 2 and y_pred.dim() == 2, "输入必须是2D: [batch_size, seq_len * 2]"
    assert y_true.shape == y_pred.shape, "y_true和y_pred必须形状相同"
    assert y_true.shape[1] % 2 == 0, "最后一维必须是偶数"

    seq_len = y_true.shape[1] // 2
    y_true_reshaped = y_true.view(-1, seq_len, 2)
    y_pred_reshaped = y_pred.view(-1, seq_len, 2)

    y_true_denorm = min_max_denormalize(y_true_reshaped, lat_min, lat_max, lon_min, lon_max)
    y_pred_denorm = min_max_denormalize(y_pred_reshaped, lat_min, lat_max, lon_min, lon_max)

    distances = haversine_distance(y_true_denorm, y_pred_denorm)
    mean_distances_per_sample = distances.mean(dim=-1)

    if sample_weight is not None:
        sample_weight = sample_weight.float().to(mean_distances_per_sample.device)
        return (mean_distances_per_sample * sample_weight).sum() / sample_weight.sum()
    else:
        return mean_distances_per_sample.mean()

def min_max_denormalize(normalize_data, lat_min, lat_max, lon_min, lon_max):
    normalized_lon, normalized_lat = normalize_data[..., 0], normalize_data[..., 1]
    lon = normalized_lon * (lon_max - lon_min) + lon_min
    lat = normalized_lat * (lat_max - lat_min) + lat_min
    return torch.stack([lon, lat], dim=-1)

def mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

def format_num(num):
    """自定义数值格式化：小于0.0001用科学计数法，否则保留4位小数"""
    if abs(num) < 1:  # 绝对值小于0.0001，用科学计数法
        return f"{num:.4e}"
    else:  # 否则保留4位小数（自动去除末尾多余0）
        return f"{num:.4f}".rstrip('0').rstrip('.') if '.' in f"{num:.4f}" else f"{num:.4f}"

if __name__ == "__main__":
    config = LLM4STPConfig()
    config.area = 'osaka'
    config.model_name = "/root/autodl-tmp/Gemma3-1B"
    config.learning_rate = 0.00001
    config.batch_size = 10
    
    
    
    # # 初始化记录指标的数据结构
    processor = DataProcessor(
        data_path='/root/autodl-tmp/llm4stp_data/featuer_AISCN_02-06_daban_202565.csv',
        region = config.area,
        use_cols= ['UnixTime_FEN', 'MMSI_', 'Course', 'Speed', 'Lon_d', 'Lat_d',
            'df0_Course', 'df0_Speed', 'df0_Lon_d', 'df0_Lat_d',
            'df1_Course', 'df1_Speed', 'df1_Lon_d', 'df1_Lat_d',
        ]
    )
    processor.process(input_len=config.seq_len, output_len=config.pred_len, train_ratio=0.8)

    # 然后获取 DataLoader , train_num_samples
    train_loader, test_loader = processor.get_dataloaders(batch_size=config.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    model = LLM4STPModel(config)
    model.to(device)
    need_add = False
    # 优化器设置（保持不变）
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    metrics = {
        'train': {
            'train_mae': [], 'train_mse': [], 'train_ade': [], 'train_fde': [],
        },
        'val': {
            'val_mae': [], 'val_mse': [], 'val_ade': [], 'val_fde': [], 'lr': [],# 'CO2': [],
        }
    }
    
    regression_loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=0.99)
    
    
    
    epochs = 500
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        # tracker.start()
        model.train()
        total_train_mae_loss = 0
        total_train_mse_loss = 0
        train_total = 0
        train_ade = 0.0
        train_fde = 0.0
        
        for step, batch in enumerate(tqdm(train_loader)):
            input_ids = batch[0].to(device)
            # print(input_ids.shape, "input_ids")
            labels = batch[-1].to(device)
            batch_size = labels.size(0)
            train_total += batch_size
            
            optimizer.zero_grad()            
            pred_output = model(input_ids)
            # 损失计算也自动在半精度上下文中进行
            mae_loss = mean_absolute_error(pred_output, labels)
            mse_loss = regression_loss_fn(pred_output, labels)
            
            # 计算ADE和FDE
            ade = haversine_ADE(labels, pred_output, config.REGION_CONFIGS['osaka']['lat_min']
                                , config.REGION_CONFIGS['osaka']['lat_max']
                                , config.REGION_CONFIGS['osaka']['lon_min']
                                , config.REGION_CONFIGS['osaka']['lon_max'])
            fde = haversine_FDE(labels, pred_output, config.REGION_CONFIGS['osaka']['lat_min']
                                , config.REGION_CONFIGS['osaka']['lat_max']
                                , config.REGION_CONFIGS['osaka']['lon_min']
                                , config.REGION_CONFIGS['osaka']['lon_max'])
            

            # 累加损失
            total_train_mae_loss += mae_loss.item() * batch_size
            total_train_mse_loss += mse_loss.item() * batch_size
            train_ade += ade.item() * batch_size
            train_fde += fde.item() * batch_size

            mse_loss.backward()
            optimizer.step()
            # scheduler.step()

        # 计算训练集平均指标
        avg_train_mae_loss = total_train_mae_loss / train_total if train_total > 0 else 0
        avg_train_mse_loss = total_train_mse_loss / train_total if train_total > 0 else 0
        avg_train_ade = train_ade / train_total if train_total > 0 else 0
        avg_train_fde = train_fde / train_total if train_total > 0 else 0
        
        # 记录训练指标
        metrics['train']['train_mae'].append(avg_train_mae_loss)
        metrics['train']['train_mse'].append(avg_train_mse_loss)
        metrics['train']['train_ade'].append(avg_train_ade)
        metrics['train']['train_fde'].append(avg_train_fde)

        # step_co2 = tracker.stop()
        # step_co2_list.append(step_co2)
        # 评估阶段
        model.eval()
        total_eval_mae_loss = 0
        total_eval_mse_loss = 0
        eval_total = 0
        eval_ade = 0.0
        eval_fde = 0.0
       
        
        if len(test_loader) == 0:
            print("警告: 测试集为空，跳过评估")
            for key in metrics['val']:
                metrics['val'][key].append(np.nan)
            # 每个epoch结束后清理缓存
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            continue
        #tracker = EmissionsTracker(
        #     # output_dir=output_dir,
        #     # offline=True,
        #    log_level="error",
        #    save_to_file=False)
        
        #tracker.start()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch[0].to(device)
                labels = batch[-1].to(device)
                batch_size = labels.size(0)
                eval_total += batch_size
                
                pred_output = model(input_ids)
                
                # step_co2_list.append(step_co2)
                # 损失计算也自动在半精度上下文中进行
                mae_loss = mean_absolute_error(pred_output, labels)
                mse_loss = regression_loss_fn(pred_output, labels)
                
                ade = haversine_ADE(labels, pred_output, config.REGION_CONFIGS['osaka']['lat_min']
                                    , config.REGION_CONFIGS['osaka']['lat_max']
                                    , config.REGION_CONFIGS['osaka']['lon_min']
                                    , config.REGION_CONFIGS['osaka']['lon_max'])
                fde = haversine_FDE(labels, pred_output, config.REGION_CONFIGS['osaka']['lat_min']
                                    , config.REGION_CONFIGS['osaka']['lat_max']
                                    , config.REGION_CONFIGS['osaka']['lon_min']
                                    , config.REGION_CONFIGS['osaka']['lon_max'])
                
                # 累加损失
                total_eval_mae_loss += mae_loss.item() * batch_size
                total_eval_mse_loss += mse_loss.item() * batch_size
                eval_ade += ade.item() * batch_size
                eval_fde += fde.item() * batch_size
        #step_co2 = tracker.stop()
        # 计算验证集平均指标
        avg_eval_mae_loss = total_eval_mae_loss / eval_total if eval_total > 0 else 0
        avg_eval_mse_loss = total_eval_mse_loss / eval_total if eval_total > 0 else 0
        avg_eval_ade = eval_ade / eval_total if train_total > 0 else 0
        avg_eval_fde = eval_fde / eval_total if train_total > 0 else 0
        
        # 记录验证指标
        metrics['val']['val_mae'].append(avg_eval_mae_loss)
        metrics['val']['val_mse'].append(avg_eval_mse_loss)
        metrics['val']['val_ade'].append(avg_eval_ade)
        metrics['val']['val_fde'].append(avg_eval_fde)
        metrics['val']['lr'].append(scheduler.get_lr())
        #metrics['val']['CO2'].append(step_co2)
       
        # print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Epoch {epoch + 1}-----"
              f"Train_MSE: {format_num(avg_train_mse_loss)}, Train_MAE: {avg_train_mae_loss:.4f}, Train_ADE: {avg_train_ade:.4f}, Train_FDE: {avg_train_fde:.4f}, "
              #f"CO2: {step_co2 * 1e6:.2f} mg, "
              f"Val_MSE: {format_num(avg_eval_mse_loss)}, Val_MAE: {avg_eval_mae_loss:.4f}, Val_ADE: {avg_eval_ade:.4f}, Val_FDE: {avg_eval_fde:.4f}, "
              )
        
        #scheduler.step()
        print(scheduler.get_last_lr())

        # # # 每个epoch结束后清理CUDA缓存
        # if device.type == 'cuda':
        #     torch.cuda.empty_cache()
    save_metrics = pd.concat([pd.DataFrame(metrics['train']), pd.DataFrame(metrics['val'])], axis=1)
    save_metrics.to_csv("/root/autodl-tmp/llm4stp_gemma/STP2LLM_gemma3-1B.csv")
    
    temp_dir = "/root/autodl-tmp/llm4stp_qwen/safetensors_temp"
    os.makedirs(temp_dir, exist_ok=True)
    os.environ["SAFETENSORS_TEMP_DIR"] = temp_dir  # safetensors专属临时目录
    os.environ["TMPDIR"] = temp_dir                # 系统临时目录
    os.environ["TEMP"] = temp_dir                  # Windows临时目录
    os.environ["TMP"] = temp_dir                   # Windows TMP目录

    # 初始化模型
    # config = LLM4STPConfig()
    # model = STPLLMModel(config)
    # processor = ProcessorLLM4STP(config)
    # processor.process()
    print(f"✅ 模型参数总量：{sum(p.numel() for p in model.parameters()) / 1024**3:.2f} GB")

    # 保存目录（D盘，空间充足）
    save_dir = "/root/autodl-tmp/llm4stp_qwen/model_llm4stp"
    os.makedirs(save_dir, exist_ok=True)
    
    config.save_pretrained(save_dir)
    
    print("12")
    # 核心：即使开启safe_serialization也能保存
    model.save_pretrained(
        save_dir,
        safe_serialization=True,  # 恢复safetensors（可选）
        # max_shard_size="10GB",    # 分片（可选）
        temp_dir=temp_dir         # 显式指定临时目录（双重保障）
    )

    print("13")
    model.save_pretrained(
        save_dir,
        safe_serialization=False,  # 保存为pt格式（如需safetensors则设为True）
        # max_shard_size="10GB",      # 每个分片最大2GB（按需调整：1GB/5GB/10GB）
        save_function=torch.save   # 显式指定保存函数
    )

    # 验证保存结果
    print(f"✅ 模型保存成功！")
    print(f"保存目录：{save_dir}")
    print(f"文件列表：{os.listdir(save_dir)}")
    # processor.save_pretrained(save_dir)
    print("processor save!")
    # processor = AutoProcessor.from_pretrained("llm4stp", data_path=save_dir)
    # ========== 5. 保存分词器（核心：生成所有tokenizer相关文件） ==========
    """
    可选两种方式：
    方式1：基于已有开源分词器（如LLaMA/Qwen/GLM）
    方式2：自定义分词器（从0构建）
    """
    # ------------------- 方式1：基于开源分词器（推荐，对齐LLM） -------------------
    # 加载开源分词器（以Qwen-7B为例，替换为你的LLM对应分词器）
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

    # print("14")
    # # 适配自定义模型的特殊Token（必须与config中的token_id一致）
    # tokenizer.bos_token_id = config.bos_token_id
    # tokenizer.eos_token_id = config.eos_token_id
    # tokenizer.pad_token_id = config.pad_token_id
    tokenizer = model.tokenizer

    tokenizer.save_pretrained(save_dir)

    # ========== 6. 验证最终文件结构（完整HF标准） ==========
    print("\n📁 最终生成的完整文件列表：")
    all_files = sorted(os.listdir(save_dir))
    for file in all_files:
        file_size = os.path.getsize(os.path.join(save_dir, file)) / 1024**3
        print(f"  - {file} (大小：{file_size:.5f} GB)")

    # metrics = main(train_loader, test_loader, model=model, sample_area=sample_area
    #                   , num_added=num_added, original_vocab_size=original_vocab_size
    #                   , tokenizer=tokenizer, loss_weights=loss_weights)
    # save_metrics = pd.concat([pd.DataFrame(metrics['train']), pd.DataFrame(metrics['val'])], axis=1)
    # save_metrics.to_csv("G:/部署相关论文/LLM船舶轨迹预测/结果/STP2LLM_GPT2—3.csv")
    
    
    save_dir = "/root/autodl-tmp/llm4stp_qwen/model_qwen2_trained"
    os.makedirs(save_dir, exist_ok=True)
    
    model.llm_config.save_pretrained(save_dir)
    
    print("12")
    
    llm_model = model.llm_model.to(torch.bfloat16)
    
    # 核心：即使开启safe_serialization也能保存
    llm_model.save_pretrained(
        save_dir,
        safe_serialization=True,  # 恢复safetensors（可选）
        # max_shard_size="10GB",    # 分片（可选）
        temp_dir=temp_dir         # 显式指定临时目录（双重保障）
    )

    print("13")
    llm_model.save_pretrained(
        save_dir,
        safe_serialization=False,  # 保存为pt格式（如需safetensors则设为True）
        # max_shard_size="10GB",      # 每个分片最大2GB（按需调整：1GB/5GB/10GB）
        save_function=torch.save   # 显式指定保存函数
    )
    
    tokenizer = model.tokenizer

    tokenizer.save_pretrained(save_dir)
