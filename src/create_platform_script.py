import json
from pathlib import Path

def generate_platform_inference_script(save_dir: Path, model_path: Path, features_path: Path, join_key: str = "bill_no", ds_id: str = "694b64f7adbe1d000b1651ef") -> None:
    print(f"开始生成补丁脚本... 目标路径: {save_dir}")
    try:
        features = []
        if features_path.exists():
            with open(features_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            features = data.get("features", data)
        
        # 即使 features 为空也继续生成，但在脚本中注释说明
        if not features:
            print("[平台脚本] features 列表在文件中未找到，将生成空列表模板")
            feature_lines = "    # WARNING: AutoModel未能自动提取特征列表，请手动填入\n"
        else:
            feature_lines = ",\n".join(f'    "{c}"' for c in features)

        template = f"""# ============================================================
# 【Cell 0】 数据源配置（发布界面必需）
# ============================================================
import aurai.data_source as ds
# 根据dsId获取数据源信息
dsId="{ds_id}"
dsCycle="yyyyMMdd"
ds.describeDS(dsId)


# ============================================================
# 【Cell 1】 基础配置与环境补丁
# ============================================================
import pandas as pd
import numpy as np
import pickle
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib  # 兼容旧版 sklearn
import sys
import warnings
import os

# ------------------------------------------------------------
# 1. 配置区域
# ------------------------------------------------------------
# 模型文件路径
MODEL_FILE = '{model_path.name}'

# 业务主键列 (从知识图谱自动识别)
ID_COLUMN = '{join_key}'

# ------------------------------------------------------------
# 2. 特征白名单 (自动生成，无需修改)
# ------------------------------------------------------------
FEATURE_WHITELIST = [
{feature_lines}
]

# ------------------------------------------------------------
# 2.1 短字段名映射（用于DM字段名长度受限场景）
# ------------------------------------------------------------
# 若上游宽表SQL输出列为 f001~f0xx，这里会在预测前自动 rename 回长特征名。
ALIAS_TO_FEATURE = {{
    f"f{{i:03d}}": FEATURE_WHITELIST[i - 1]
    for i in range(1, len(FEATURE_WHITELIST) + 1)
}}

# ------------------------------------------------------------
# 3. 环境补丁 (关键：解决 sklearn 版本不一致问题)
# ------------------------------------------------------------
# 定义一个假的 sklearn.neighbors.typedefs 模块
# 因为旧版 saved model 可能引用了这个在新版被移除的模块
from types import ModuleType
if 'sklearn.neighbors.typedefs' not in sys.modules:
    fake_module = ModuleType('sklearn.neighbors.typedefs')
    sys.modules['sklearn.neighbors.typedefs'] = fake_module
    # 同时也注入到 sklearn.neighbors 中
    import sklearn.neighbors
    sklearn.neighbors.typedefs = fake_module

# ------------------------------------------------------------
# 4. 加载模型
# ------------------------------------------------------------
print(f"正在加载模型: {{MODEL_FILE}} ...")
try:
    # 尝试直接加载
    pipeline = joblib.load(MODEL_FILE)
except Exception as e:
    print(f"joblib加载失败，尝试 pickle: {{e}}")
    with open(MODEL_FILE, 'rb') as f:
        pipeline = pickle.load(f)
print("模型加载成功！")

# ============================================================
# 【Cell 2】 数据读取 (模拟平台 Reader)
# ============================================================
# 实际平台中 reader 由系统提供，这里仅做本地测试模拟
if 'reader' not in globals():
    print("警告：未检测到 reader 对象，仅生成代码结构。")
    # reader = ... 

# ============================================================
# 【Cell 3】 循环预测 (核心逻辑)
# ============================================================
batch_size = 10000
i = 0

# 准备结果 CSV
if os.path.exists('predict_result.csv'):
    os.remove('predict_result.csv')

# 写入表头
with open('predict_result.csv', 'w') as f:
    f.write(f"{{ID_COLUMN}},probability,prediction\\n")

while True:
    try:
        # 1. 读取数据块
        # df = reader.read_pandas(batch_size) 
        # if len(df) == 0: break
        break # 本地测试直接跳出
        
        i += 1
        # print(f"正在处理第 {{i}} 批次，大小: {{len(df)}}...")

        # 2. [关键] 列名还原 (Short Alias -> Long Feature Name)
        # 检查是否是 f001, f002 这种格式
        current_cols = set(df.columns)
        rename_dict = {{}}
        for col in current_cols:
            if col in ALIAS_TO_FEATURE:
                rename_dict[col] = ALIAS_TO_FEATURE[col]
        
        if rename_dict:
            # print(f"检测到短列名，正在还原 {{len(rename_dict)}} 个特征名...")
            df = df.rename(columns=rename_dict)

        # 3. 特征对齐 (确保只有白名单内的特征进入模型)
        # 补全缺失特征为 0
        valid_features = [c for c in FEATURE_WHITELIST if c in df.columns]
        missing_features = [c for c in FEATURE_WHITELIST if c not in df.columns]
        
        if missing_features:
            # print(f"警告：本批次缺失 {{len(missing_features)}} 个特征，将填充为 0")
            for c in missing_features:
                df[c] = 0
                
        # 剔除多余特征，并按白名单顺序排列
        X_batch = df[FEATURE_WHITELIST]

        # 4. 预测
        # 注意：这里直接调用 pipeline.predict_proba
        # 只要 pipeline 包含了 preprocessor，它就会自动处理缺失值填充和独热编码
        pred_prob = pipeline.predict_proba(X_batch)[:, 1]
        
        # 5. 生成结果 DataFrame
        result_df = pd.DataFrame({{
            ID_COLUMN: df[ID_COLUMN],
            'probability': pred_prob,
            'prediction': (pred_prob > 0.5).astype(int)
        }})
        
        # 6. 追加写入结果
        result_df.to_csv('predict_result.csv', mode='a', header=False, index=False)
        
    except Exception as e:
        print(f"错误: {{e}}")
        import traceback
        traceback.print_exc()
        break

# reader.close()
print(f"循环预测完成，共处理 {{i}} 批次")


# ============================================================
# 【Cell 4】 读取预测结果
# ============================================================
predDF = pd.read_csv('predict_result.csv')
print(predDF.shape)


# ============================================================
# 【Cell 5】 保存预测结果
# ============================================================
# predDF.to_csv('predict_result.csv', index=False)
"""

        out_path = save_dir / "platform_inference_template.py"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(template)
        print(f"[平台脚本] 已生成: {out_path}")
    except Exception as e:
        print(f"[平台脚本] 生成失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_platform_inference_script(
        save_dir=Path("saved_model"),
        model_path=Path("saved_model/automl_pipeline_compatible.pkl"),
        features_path=Path("saved_model/expected_feature_columns.json")
    )