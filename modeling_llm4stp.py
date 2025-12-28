# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 15:09:26 2025

@author: HP
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Optional, Tuple, List
from Config_llm4stp import LLM4STPConfig
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig

class VesselSituationReporter(nn.Module):
    """船只状态报告生成器"""
    def __init__(self, num_surrounding_vessels: int):
        super().__init__()
        self.num_surrounding = num_surrounding_vessels
        self.tolerance = 1e-5

    @staticmethod
    def classify_relative_position(relative_bearing: float) -> str:
        if 0 <= relative_bearing < 10 or 350 <= relative_bearing < 360:
            return "directly ahead"
        elif 10 <= relative_bearing < 80:
            return "forward on starboard side"
        elif 80 <= relative_bearing < 100:
            return "abeam to starboard"
        elif 100 <= relative_bearing < 170:
            return "aft on starboard side"
        elif 170 <= relative_bearing < 190:
            return "directly astern"
        elif 190 <= relative_bearing < 260:
            return "aft on port side"
        elif 260 <= relative_bearing < 280:
            return "abeam to port"
        elif 280 <= relative_bearing < 350:
            return "forward on port side"
        else:
            return "unknown aspect"

    @staticmethod
    def calculate_true_bearing(
        lat1: torch.Tensor, lon1: torch.Tensor,
        lat2: torch.Tensor, lon2: torch.Tensor
    ) -> torch.Tensor:
        deg2rad = torch.pi / 180.0
        lat1_rad = lat1 * deg2rad
        lon1_rad = lon1 * deg2rad
        lat2_rad = lat2 * deg2rad
        lon2_rad = lon2 * deg2rad

        y = torch.sin(lon2_rad - lon1_rad) * torch.cos(lat2_rad)
        x = torch.cos(lat1_rad) * torch.sin(lat2_rad) - \
            torch.sin(lat1_rad) * torch.cos(lat2_rad) * torch.cos(lon2_rad - lon1_rad)

        bearing = torch.atan2(y, x)
        bearing_deg = torch.rad2deg(bearing)
        return (bearing_deg + 360) % 360

    def _analyze_course(self, course_data: torch.Tensor) -> str:
        start_course = course_data[0]#.item()
        end_course = course_data[-1]#.item()
        diff = abs(end_course - start_course)
        angle_diff = min(diff, 360 - diff)

        if angle_diff <= 3:
            return f"maintained a steady heading of {int(start_course)}°"
        else:
            turn_angle = (end_course - start_course) % 360
            turn_angle = turn_angle - 360 if turn_angle > 180 else turn_angle
            turn_dir = "turned right" if turn_angle >= 0 else "turned left"
            turn_deg = abs(int(turn_angle))
            return f"gradually {turn_dir} by {turn_deg}° to {int(end_course)}°"
    
    def torch_round_to_n_digits(self, x: torch.Tensor, n: int = 4) -> torch.Tensor:
        multiplier = 10 ** n
        return torch.round(x * multiplier) / multiplier
    
    
    def _analyze_speed(self, speed_data: torch.Tensor) -> str:
        start_speed = speed_data[0]#.item()
        final_speed = speed_data[-1]#.item()
        avg_speed = self.torch_round_to_n_digits(torch.mean(speed_data), 1) #.item()

        if abs(final_speed - start_speed) < 0.3:
            return f"cruising at a stable speed of {avg_speed:.1f} knots"
        else:
            change_dir = "decelerated" if final_speed < start_speed else "accelerated"
            return f"{change_dir} from {start_speed:.1f} to {final_speed:.1f} knots, averaging {avg_speed:.1f} knots"

    def _get_vessel_trajectory(self, lon_data: torch.Tensor, lat_data: torch.Tensor) -> Tuple[str, str, str]:
        positions = [
            (self.torch_round_to_n_digits(lon_data[t], 4)
             , self.torch_round_to_n_digits(lat_data[t], 4))
            # for t in range(len(lon_data))
            for t in range(lon_data.size(0))
        ]
        trajectory_str = "[" + ", ".join(f"({lon:.4f}, {lat:.4f})" for lon, lat in positions) + "]"
        start_pos = f"({positions[0][0]:.4f}, {positions[0][1]:.4f})"
        end_pos = f"({positions[-1][0]:.4f}, {positions[-1][1]:.4f})"
        return start_pos, end_pos, trajectory_str

    def _is_vessel_valid(self, vessel_data: torch.Tensor) -> bool:
        return not (
            torch.allclose(vessel_data[:, 0], torch.zeros_like(vessel_data[:, 0]), atol=self.tolerance) and
            torch.allclose(vessel_data[:, 1], torch.zeros_like(vessel_data[:, 1]), atol=self.tolerance)
        )

    def _get_surrounding_vessel_report(
        self,
        sid: str,
        data_tensor: torch.Tensor,
        target_course: torch.Tensor,
        target_lon: torch.Tensor,
        target_lat: torch.Tensor,
        valid_vessels: List[int]
    ) -> str:
        idx = int(sid) - 1
        if idx not in valid_vessels:
            return ""
        
        start_col = 4 + 4 * idx
        data = data_tensor[:, start_col:start_col+4]
        courses = data[:, 0]
        speeds = data[:, 1]
        lons = data[:, 2]
        lats = data[:, 3]

        start_pos, end_pos, _ = self._get_vessel_trajectory(lons, lats)

        tb_t1 = self.calculate_true_bearing(
            target_lat[0], target_lon[0], lats[0], lons[0]
        )#.item()
        rb_t1 = (tb_t1 - target_course[0]) % 360 #.item()
        pos_t1 = self.classify_relative_position(rb_t1)

        tb_tn = self.calculate_true_bearing(
            target_lat[-1], target_lon[-1], lats[-1], lons[-1]
        )#.item()
        rb_tn = (tb_tn - target_course[-1]) % 360 #.item()
        pos_tn = self.classify_relative_position(rb_tn)

        pos_change = f"moved from {pos_t1} to {pos_tn}" if pos_t1 != pos_tn else f"remained at {pos_t1}"
        course_text = self._analyze_course(courses)
        speed_text = self._analyze_speed(speeds)

        return (f"Surrounding ship {sid} traveled from {start_pos} to {end_pos}, "
                f"positioned relative to the target ship as {pos_change}, "
                f"{course_text}, "
                f"{speed_text}")


    def forward(self, data_tensor: torch.Tensor) -> str:
        if data_tensor.dim() != 2:
            raise ValueError(f"期望2D张量 [T, C], 得到 {data_tensor.shape}")
        T, C = data_tensor.shape
        expected_cols = 4 + 4 * self.num_surrounding
        if C != expected_cols:
            raise ValueError(f"期望 {expected_cols} 列, 得到 {C}")

        target_course = data_tensor[:, 0]
        target_speed = data_tensor[:, 1]
        target_lon = data_tensor[:, 2]
        target_lat = data_tensor[:, 3]

        valid_vessel_indices = []
        for i in range(self.num_surrounding):
            start_col = 4 + 4 * i
            vessel_data = data_tensor[:, start_col:start_col+4]
            if self._is_vessel_valid(vessel_data):
                valid_vessel_indices.append(i)

        surrounding_ids = [str(i+1) for i in valid_vessel_indices]
        start_pos_target, end_pos_target, trajectory_str = self._get_vessel_trajectory(
            target_lon, target_lat
        )

        course_change_text = self._analyze_course(target_course)
        speed_text = self._analyze_speed(target_speed)

        surround_reports = []
        for sid in surrounding_ids:
            try:
                rep = self._get_surrounding_vessel_report(
                    sid, data_tensor, target_course, target_lon, target_lat, valid_vessel_indices
                )
                if rep:
                    surround_reports.append(rep)
            except Exception as e:
                print(f"生成船只 {sid} 报告失败: {e}")
                continue
        # print(surround_reports)
        if len(surround_reports) == 0:
            surround_text = "No surrounding vessel activity detected"
        elif len(surround_reports) == 1:
            surround_text = surround_reports[0]
        else:
            surround_text = "; ".join(surround_reports)

        return (
            f"The target ship traversed from {start_pos_target} to {end_pos_target} over {T} consecutive time steps, "
            f"with trajectory: {trajectory_str}; "
            f"its heading {course_change_text}, "
            f"and {speed_text}. "
            f"{surround_text}."
        )

class FlashAttention(nn.Module):
    """基于PyTorch SDPA的Flash Attention实现"""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model必须能被nhead整除"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size = q.size(0)
        q_orig = q
        
        # if q.dim() == 3 and q.size(0) != batch_size:
        #     q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        
        q = self.w_q(q).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        
        # with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True
        #                                     , enable_mem_efficient=True):
        attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p= self.dropout.p if self.training else 0.0,
                is_causal=False
            )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        output = self.dropout(self.w_o(attn_output))
        output = self.layer_norm(q_orig + output)
        
        return output

class FusionLayer(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        
        num_concat = 3
        self.FlashAttention = FlashAttention(d_model=num_concat * d_model, nhead=3)
        
        self.fusion = nn.Sequential(
            nn.Linear(num_concat * d_model, num_concat * d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        
        self.fc_fusion_1 = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.fc_fusion_2 = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.layernorm = nn.LayerNorm(num_concat * d_model, eps=1e-6)
        self.fc_out = nn.Linear(num_concat * d_model, d_model, bias=True)

    def forward(self, input_x, input_y):
        # if input_x.shape != input_y.shape:
        #     raise ValueError(f"形状不匹配: {input_x.shape} vs {input_y.shape}")
        # if input_x.size(-1) != self.d_model:
        #     raise ValueError(f"期望最后一维为{self.d_model}, 实际为{input_x.size(-1)}")
        
        x_mut = self.fc_fusion_1(input_x) * input_x 
        y_mut = self.fc_fusion_2(input_y) * input_y 
        mut_dot = x_mut * y_mut
        
        concat_xy = torch.cat([x_mut, y_mut, mut_dot], dim=-1)
        fusion_output = self.fusion(concat_xy)
        fusion_output = self.layernorm(fusion_output + concat_xy)
        fusion_output = self.FlashAttention(fusion_output, fusion_output, fusion_output)
        
        return self.fc_out(fusion_output)

class DualFeatureExtractionStructureBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.filters = config.out_channels
        self.seq_len = config.seq_len
        
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.fusionlayer = FusionLayer(d_model=128, dropout=config.dropout)
        self.gelu = nn.GELU()

        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=self.filters, kernel_size=3, padding=1)
        self.Conv2 = nn.Conv1d(in_channels=4, out_channels=self.filters, kernel_size=3, padding=1)
        
        self.Conv4 = nn.Conv2d(in_channels=self.filters, out_channels=self.filters,
            kernel_size=(1, config.num_vessels), stride=1)

    def _process_with_cnn(self, targetship, aroundship):
        # print(targetship.shape, "targetship")
        
        x1 = self.Conv1(targetship)
        # print(x1.shape, "x1")
        x1 = x1.transpose(1, 2)
        x1 = self.gelu(x1)
        x1_1 = self.dropout_layer(x1)#.transpose(1, 2)
        
        around_num = aroundship.shape[1] // 4
        feature_list = []
        
        for i in range(around_num):
            xx = aroundship[:, 4*i:(i+1)*4, :]
            x2 = self.Conv2(xx)
            x2 = x2.transpose(1, 2)
            x2 = self.gelu(x2)
            x2_1 = self.dropout_layer(x2)#.transpose(1, 2)
            fusion_vector = self.fusionlayer(x1_1, x2_1)
            feature_list.append(fusion_vector.unsqueeze(-1))

        return feature_list, x1_1

    def _process_with_lstm(self, targetship, aroundship):
        x1_output, _ = self.feature_extractor_target(targetship)
        x1_1 = self.dropout_layer(x1_output)

        around_num = aroundship.shape[2] // 4
        feature_list = []
        
        for i in range(around_num):
            xx = aroundship[:, :, 4*i:(i+1)*4]
            x2_output, _ = self.feature_extractor_around(xx)
            x2_1 = self.dropout_layer(x2_output)
            fusion_vector = self.fusionlayer(x1_1, x2_1)
            feature_list.append(fusion_vector.unsqueeze(-1))

        return feature_list, x1_1

    def forward(self, x):
        targetship = x[:, :, 0:4]
        aroundship = x[:, :, 4:]
 
        targetship = targetship.transpose(1, 2)
        aroundship = aroundship.transpose(1, 2)
        
        aroundship_info, targetship_info = self._process_with_cnn(targetship, aroundship)
        
        around_feature = torch.cat(aroundship_info, dim=-1).transpose(1, 2)
        
        # if around_feature.shape[-1] < 2:
        #     raise ValueError(f"周围船只数量不足: {around_feature.shape[-1]}")
        output = self.Conv4(around_feature)
        # print(output.shape, around_feature.shape, "output")
        output = self.Conv4(around_feature).squeeze(-1).transpose(1, 2)
        return output

class FlattenMultiHeads(nn.Module):
    """输出头（保持逻辑不变）"""
    def __init__(self, config):
        super().__init__()  
        self.config = config
        self.dense = nn.Linear(in_features=config.last_state_num, out_features=config.d_model)
        self.dense_reg = nn.Linear(
            in_features=config.d_model * config.input_seq_len,
            out_features=config.d_model
        )
        self.dropout = nn.Dropout(p=config.dropout)
        self.head = nn.Linear(in_features=config.d_model, out_features=config.pred_len * config.pred_regress_len)
        self.attn = nn.MultiheadAttention(embed_dim=config.d_model, num_heads=config.atten_num_heads, batch_first=True)
        self.ln = nn.LayerNorm(config.d_model, eps=1e-6)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.dense(x)
        x_flat = x.reshape(batch_size, -1)
        x_flat = self.dropout(x_flat)
        x_flat = self.dense_reg(x_flat)
        
        # 适配MultiheadAttention输入形状
        x_flat = x_flat.unsqueeze(1)  # [B, 1, d_model]
        x_flat_, _ = self.attn(x_flat, x_flat, x_flat)
        x_flat = self.ln(x_flat_ + x_flat).squeeze(1)
        
        x_out = self.head(x_flat)
        return x_out

class MHAttentionLayer(nn.Module):
    """多头注意力层（修复参数名错误）"""
    def __init__(self, config):
        super(MHAttentionLayer, self).__init__()
        self.n_heads = config.ffn_heads
        self.d_keys = config.d_keys or (config.llmmodel_hidden_size // self.n_heads)  # 修复 d_keys -> d_key
        self.d_llm = config.d_llm or config.llmmodel_hidden_size
        
        self.query_projection = nn.Linear(config.d_model, self.d_keys * self.n_heads)
        self.key_projection = nn.Linear(config.num_tokens, self.d_keys * self.n_heads)
        self.value_projection = nn.Linear(config.num_tokens, self.d_keys * self.n_heads)
        self.out_projection = nn.Linear(self.d_keys * self.n_heads, self.d_llm)
        self.dropout = nn.Dropout(config.dropout)
   
    def score_and_values(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape
        scale = 1. / torch.sqrt(torch.tensor(E, dtype=torch.float32, device=target_embedding.device))
        
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)
        return reprogramming_embedding
    
    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S = source_embedding.size(0)
        
        target_embedding = self.query_projection(target_embedding)
        target_embedding = target_embedding.view(-1, L, self.n_heads, self.d_keys)
        
        source_embedding = self.key_projection(source_embedding)
        source_embedding = source_embedding.view(S, self.n_heads, self.d_keys)
        
        value_embedding = self.value_projection(value_embedding)
        value_embedding = value_embedding.view(S, self.n_heads, self.d_keys)

        out = self.score_and_values(target_embedding, source_embedding, value_embedding)
        out = out.reshape(-1, L, out.size(2) * out.size(-1))
        return self.out_projection(out)

class LLM4STPModel(PreTrainedModel):
    """主模型（修复Hugging Face兼容性+维度错误）"""
    config_class = LLM4STPConfig
    base_model_prefix = "qwen2"
    
    def __init__(self, config):
        super().__init__(config)
        self.d_model = config.d_model
        self.region = config.region
        self.config = config
        
        print(config.model_name, "---------config.model_name--------")
        # 加载LLM模型（兼容本地/Hub路径）
        self.llm_config = AutoConfig.from_pretrained(config.model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=config.model_name, attn_implementation="eager")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.llm_model.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.llmmodel_hidden_size = self.llm_model.config.hidden_size
    
        # 子模块初始化
        self.featureextractionblock = DualFeatureExtractionStructureBlock(config)
        self.reporter = VesselSituationReporter(num_surrounding_vessels=config.num_vessels)
        
        # 词嵌入层（修复设备不匹配）
        self.word_embeddings = nn.Parameter(self.llm_model.get_input_embeddings().weight.clone().detach())
        self.num_tokens = config.num_tokens  # 修复硬编码
        self.mapping_layer = nn.Linear(self.word_embeddings.shape[0], config.num_tokens)  # 修复维度
        self.mh_attention = MHAttentionLayer(config)
        self.MFHead = FlattenMultiHeads(config)
    
    def _min_max_denormalize(self, normalized_tensor: torch.Tensor) -> torch.Tensor:
        """反归一化（保持逻辑不变）"""
        cfg = self.region
        eps = 1e-8
        cou = normalized_tensor[..., 0] * (cfg['cou_max'] - cfg['cou_min'] + eps) + cfg['cou_min']
        spd = normalized_tensor[..., 1] * (cfg['spd_max'] - cfg['spd_min'] + eps) + cfg['spd_min']
        lon = normalized_tensor[..., 2] * (cfg['lon_max'] - cfg['lon_min'] + eps) + cfg['lon_min']
        lat = normalized_tensor[..., 3] * (cfg['lat_max'] - cfg['lat_min'] + eps) + cfg['lat_min']
        return torch.stack([cou, spd, lon, lat], dim=-1)

    def denormalize_attributes(self, attribute: torch.Tensor) -> torch.Tensor:
        """批量反归一化（保持逻辑不变）"""
        # assert attribute.shape[-1] % 4 == 0, "最后一维必须是4的倍数"
        num_vessels = attribute.shape[-1] // 4
        denorm_chunks = []
    
        for i in range(num_vessels):
            s, e = 4 * i, 4 * i + 4
            chunk = attribute[..., s:e]
            denorm_chunk = self._min_max_denormalize(chunk)
            denorm_chunks.append(denorm_chunk)
    
        return torch.cat(denorm_chunks, dim=-1)
        
    def forward(self, input_ids):
        batch_size = input_ids.size(0)
        # 1. 数值特征提取
        final_out = self.featureextractionblock(input_ids)
        
        # 2. 词嵌入映射（修复维度+设备）
        self.word_embeddings = self.word_embeddings.to(input_ids.device)
        source_embeddings_mapped = self.mapping_layer(self.word_embeddings.T)
        enc_out = self.mh_attention(final_out, source_embeddings_mapped, source_embeddings_mapped)
        
        # 3. 生成态势报告
        prompt = []
        for b in range(batch_size):
            attribute = input_ids[b, :]
            attribute = self.denormalize_attributes(attribute)
            reporter_text = self.reporter(attribute)
            prompt.append(reporter_text)
            #print(reporter_text)
            #test= self.tokenizer(reporter_text, return_tensors='pt'
            #                       , padding=True
            #                       ).input_ids.to(input_ids.device)
            #print(test.shape)
            
        
        # 4. 文本编码
        prompt_ids= self.tokenizer(prompt, return_tensors='pt'
                                   , padding='max_length', truncation=True
                                   , max_length= 300#, max_length=2048
                                   ).input_ids.to(input_ids.device)
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_ids)

        # 5. 特征拼接 + LLM推理
        if enc_out.shape[-1] != prompt_embeddings.shape[-1]:
            enc_out = nn.Linear(enc_out.shape[-1], prompt_embeddings.shape[-1]).to(input_ids.device)(enc_out)
        
        llm_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llm_enc_out
                                 , output_hidden_states=True).hidden_states[-1][:, :, :256]

        # 6. 最终预测
        regression_output = self.MFHead(dec_out) 
        return regression_output
    
    def output_min_max_denormalize(self, normalize_data: torch.Tensor):
        cfg = self.region
        eps = 1e-8    
        lon = normalize_data[:, 0:2:int(self.config.pred_len * self.config.pred_regress_len)] * (cfg['lon_max'] - cfg['lon_min'] + eps) + cfg['lon_min']
        lat = normalize_data[:, 1:2:int(self.config.pred_len * self.config.pred_regress_len)] * (cfg['lat_max'] - cfg['lat_min'] + eps) + cfg['lat_min']
        
        return torch.stack([lon, lat], dim=-1)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        # temperature: float = 0.7,
        # top_k: int = 50,
        eos_token_id: int = None,
        pad_token_id: int = None,
        
        **kwargs
    ):
        """适配LLM的generate实现（兼容Transformers参数）"""
        # 适配默认参数
        eos_token_id = eos_token_id or self.config.eos_token_id
        pad_token_id = pad_token_id or self.config.pad_token_id

        # 初始化生成序列
        outputs = self.forward(input_ids, return_dict=True)
        logits = outputs.logits
        output_logits = self.output_min_max_denormalize(logits)

        return output_logits    

LLM4STPModel.register_for_auto_class("AutoModelForCausalLM")