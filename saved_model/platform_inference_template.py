# ============================================================
# 【Cell 0】 数据源配置（发布界面必需）
# ============================================================
import aurai.data_source as ds
# 根据dsId获取数据源信息
dsId="YOUR_DS_ID"
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
MODEL_FILE = 'saved_model\automl_pipeline_compatible.pkl'

# 业务主键列 (从知识图谱自动识别)
ID_COLUMN = 'bill_no'

# ------------------------------------------------------------
# 2. 特征白名单 (自动生成，无需修改)
# ------------------------------------------------------------
FEATURE_WHITELIST = [
    "LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_l1m_pay_fee",
    "LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_sec_imei_l_use_date",
    "LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_fir_imei_mon_gprs",
    "LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_n3m_fact_fee",
    "LATEST_temp_user_base_info_m_bak_encrypt_202511_occu",
    "LATEST_temp_user_base_info_m_bak_encrypt_202511_sub_occu_name",
    "LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_thir_imei_use_days",
    "LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_sec_imei_use_days",
    "LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_sc_freeze",
    "LATEST_temp_user_base_info_m_bak_encrypt_202511_community_price",
    "LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_fir_imei_f_use_date",
    "LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_sms_comm_fee",
    "LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_fir_imei_gpu_rate",
    "LATEST_temp_user_base_info_m_bak_encrypt_202511_income_level",
    "LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_l2m_pay_fee",
    "LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_sc_used",
    "LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_fir_imei_price",
    "LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_fir_imei_mdl",
    "LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_fir_imei_mon_gprs_4g",
    "LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_fir_imei_mode",
    "LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_book_bj_bal",
    "LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_book_kzj_bal",
    "LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_l3m_call_fee",
    "LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_woff_bj_fee",
    "LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_month_fee",
    "LATEST_temp_user_base_info_m_bak_encrypt_202511_car_probabi",
    "LATEST__"
]

# ------------------------------------------------------------
# 2.1 短字段名映射（用于DM字段名长度受限场景）
# ------------------------------------------------------------
# 若上游宽表SQL输出列为 f001~f0xx，这里会在预测前自动 rename 回长特征名。
ALIAS_TO_FEATURE = {
    f"f{i:03d}": FEATURE_WHITELIST[i - 1]
    for i in range(1, len(FEATURE_WHITELIST) + 1)
}

# ------------------------------------------------------------
# 3. 兼容性补丁 (必须执行)
# ------------------------------------------------------------
# 【新增补丁】解决 sklearn 版本不一致导致的 _RemainderColsList 缺失问题
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, '_RemainderColsList'):
        print("检测到平台 sklearn 版本较新，正在注入 _RemainderColsList 补丁...")
        class _RemainderColsList(list):
            pass
        _ct._RemainderColsList = _RemainderColsList
except ImportError:
    pass

# 【新增补丁2】解决 sklearn.preprocessing._label 缺失问题
try:
    # 1. 获取真正的 LabelEncoder
    from sklearn.preprocessing import LabelEncoder
    import types
    
    # 2. 检查或创建 sklearn.preprocessing._label 模块
    if 'sklearn.preprocessing._label' not in sys.modules:
        _label_mod = types.ModuleType('sklearn.preprocessing._label')
        sys.modules['sklearn.preprocessing._label'] = _label_mod
    
    # 3. 强制挂载 LabelEncoder 到该模块
    sys.modules['sklearn.preprocessing._label'].LabelEncoder = LabelEncoder
    print("已注入 sklearn.preprocessing._label.LabelEncoder 补丁")

except Exception as e:
    print(f"补丁2注入警告: {e}")

# 【新增补丁3】解决 OneHotEncoder._legacy_mode 缺失问题
try:
    from sklearn.preprocessing import OneHotEncoder
    if not hasattr(OneHotEncoder, '_legacy_mode'):
        print("检测到平台 sklearn 版本较新，正在注入 OneHotEncoder._legacy_mode 补丁...")
        setattr(OneHotEncoder, '_legacy_mode', False)
except ImportError:
    pass

class CompatibleImputer:
    def __init__(self, strategy='mean', fill_value=None, missing_values=np.nan):
        self.strategy = strategy
        self.fill_value = fill_value
        self.missing_values = missing_values
        self.statistics_ = None
    def transform(self, X):
        X = np.asarray(X, dtype=np.float64).copy()
        for i, stat in enumerate(self.statistics_):
            mask = np.isnan(X[:, i])
            X[mask, i] = stat
        return X
    def fit_transform(self, X, y=None): return self.transform(X)

sys.modules['__main__'].CompatibleImputer = CompatibleImputer
warnings.filterwarnings('ignore')
print("【Cell 1】环境配置完成")


# ============================================================
# 【Cell 2】 加载模型
# ============================================================
model = joblib.load(MODEL_FILE)
print("模型加载完成")

# 【关键补丁】修复已加载模型中的 OneHotEncoder.sparse 属性
# 遍历 Pipeline 中的所有 Transformer，给 OneHotEncoder 实例添加缺失的 sparse 属性
try:
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    
    def patch_onehot_encoders(obj):
        """递归修补对象中的所有 OneHotEncoder 实例"""
        if isinstance(obj, OneHotEncoder):
            if not hasattr(obj, 'sparse'):
                obj.sparse = getattr(obj, 'sparse_output', True)
                print(f"已修补 OneHotEncoder 实例: sparse={obj.sparse}")
        elif isinstance(obj, Pipeline):
            for name, step in obj.steps:
                patch_onehot_encoders(step)
        elif isinstance(obj, ColumnTransformer):
            for name, transformer, columns in obj.transformers_:
                patch_onehot_encoders(transformer)
    
    patch_onehot_encoders(model)
    print("已完成模型中所有 OneHotEncoder 的补丁注入")
except Exception as e:
    print(f"OneHotEncoder 补丁注入失败: {e}")


# ============================================================
# 【Cell 3】 执行预测
# ============================================================

def preprocess_and_filter(df, feature_whitelist):
    """根据白名单严格过滤和对齐特征列"""
    # 兼容短字段名：若存在 f001... 列，先映射回长特征名
    if any(col in df.columns for col in ALIAS_TO_FEATURE.keys()):
        df = df.rename(columns=ALIAS_TO_FEATURE)

    # 只保留白名单中的列，缺失列用 NaN 填充
    df_clean = df.reindex(columns=feature_whitelist)
    
    # 强制将每一列都转为 numeric，无法转换的变为 NaN
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # 将所有的 NaN 填充为 0
    df_clean = df_clean.fillna(0)
    
    # 【关键修复】转为字符串类型，避免模型内部 Encoder 的类型冲突
    # 模型中的分类编码器期望字符串输入，如果传入 float 会导致类型转换错误
    df_clean = df_clean.astype(str)
    
    return df_clean

# 初始化数据读取器
print("正在连接数据源...")
reader = ds.load_block(dsId)
i = 0
onceReadLines = 50000

print("开始批量预测...")

while True:
    try:
        # 读取数据块
        df_chunk = reader.get_chunk(onceReadLines)
        if df_chunk.empty: break
        
        # 准备输出数据 (ID列)
        if ID_COLUMN in df_chunk.columns:
            result_ids = df_chunk[ID_COLUMN].values
        else:
            # 如果没有ID列，生成索引作为ID
            result_ids = df_chunk.index.values
            
        # 特征工程 (自动过滤 city_id 等无关列)
        X_test = preprocess_and_filter(df_chunk, FEATURE_WHITELIST)
        
        # 预测
        try:
            preds = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            preds = model.predict(X_test)
            
        # 构造结果集
        predDF = pd.DataFrame({
            ID_COLUMN: result_ids,
            'p_value': preds
        })
        
        # 写入文件
        if i == 0:
            predDF.to_csv("predict_result.csv", index=False)
        else:
            predDF.to_csv("predict_result.csv", index=False, header=False, mode='a')
            print("add " + str(i) + " data to csv")
        
        i += 1
            
    except StopIteration:
        break
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        break

reader.close()
print(f"循环预测完成，共处理 {i} 批次")


# ============================================================
# 【Cell 4】 读取预测结果
# ============================================================
predDF = pd.read_csv('predict_result.csv')
print(predDF.shape)


# ============================================================
# 【Cell 5】 保存预测结果
# ============================================================
predDF.to_csv('predict_result.csv', index=False)
