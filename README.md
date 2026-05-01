# LLM-EdgeSTP
A Large Language Model-Empowered Approach to Trajectory Prediction for Edge Deployment on Ships

LLM-EdgeSTP 是一个面向船舶轨迹预测（Ship Trajectory Prediction, STP）任务、融合AIS轨迹数据与海事文本知识、并支持边缘设备部署的大语言模型（LLM）框架。该系统通过跨模态推理，将连续的 AIS 轨迹与海事语义知识统一建模，显著提升了在复杂航行场景下的预测精度与工程实用性。

# 主要特点（创新点）

🚀 首创 LLM 驱动的船舶轨迹预测
  
  首次将大语言模型应用于船舶轨迹预测领域，通过“海事语义提示”（Maritime Semantic Prompts）融合轨迹与文本知识，突破了传统方法难以整合非数值语义信息的瓶颈。
  
🌊 创新的航行交互表征单元 (NIRU)

  设计了结合卷积与注意力机制的 NIRU 模块，对船舶间的会遇、避碰等动态交互进行建模，并通过表征-语言重构块 (RLRB) 将其映射到 LLM 的语义空间，使模型能“理解”多船交互意图。

🧠 AIS-海事语义双模态协同推理框架

  利用轻量级 LLM（如 Gemma、Qwen、LLaMA）作为推理引擎，融合原始海事提示与重构后的交互语义，生成自然语言形式的未来轨迹描述，并解码为精确坐标。

# 核心代码结构如下：

LLM-EdgeSTP/

├── Config_llm4stp.py                 # 全局配置文件

├── data.py                           # AIS数据加载

├── modeling_llm4stp.py               # 模型文件

├── traring_llm4stp.py                # 模型训练文件

└── requirements.txt                  # Python 依赖库列表
