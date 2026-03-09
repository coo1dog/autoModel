import streamlit as st
import pandas as pd
import json
import time
import subprocess
import re
import sys
from pathlib import Path
import os
import plotly.express as px
import joblib

# --- 配置页面 ---
st.set_page_config(
    page_title="AutoML 智能建模平台",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 常量定义 ---
LOG_FILE = "automl.log"
SAVED_MODEL_DIR = Path("saved_model")
CHAMPION_FILE = SAVED_MODEL_DIR / "champion_chromosome.json"
PIPELINE_FILE = SAVED_MODEL_DIR / "automl_pipeline_compatible.pkl"
FEATURES_FILE = SAVED_MODEL_DIR / "expected_feature_columns.json"

# --- 样式CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #41424b;
        text-align: center;
    }
    .stProgress .st-bo {
        background-color: #00ff00;
    }
    .code-box {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# --- 侧边栏：控制面板 ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("AutoML 控制台")
    st.markdown("---")
    
    st.subheader("⚙️ 实验参数")
    generations = st.slider("进化世代 (Generations)", 1, 50, 2)
    population = st.slider("种群规模 (Population)", 2, 100, 2)
    
    st.markdown("---")
    st.subheader("📂 数据配置")
    
    # 自动扫描当前目录下的 dataset_ 开头的文件夹
    potential_folders = [d.name for d in Path(".").iterdir() if d.is_dir() and d.name.startswith("dataset_")]
    # 如果没找到，给个默认值
    if not potential_folders:
        potential_folders = ["dataset_termchange"]
        
    csv_folder = st.selectbox("数据目录 (CSV Folder)", potential_folders, index=0)
    
    # 根据选择的目录动态建议主表名 (简单的逻辑：找最大的 csv 或者包含 target 的 csv，这里先允许手动输入)
    main_table_input = st.text_input("主表名 (无需.csv后缀)", value="temp_a_utrm_use_3m_inline_encrypt_202508")
    target_col_input = st.text_input("目标列名 (Target Column)", value="flag")

    data_source_mode = st.selectbox("数据源模式", ["CSV (本地)", "ClickHouse (生产)"])
    if data_source_mode == "ClickHouse (生产)":
        st.info("检测到生产环境配置，将连接 NEWYdxzz 集群")
    
    st.markdown("---")
    start_btn = st.button("🚀 启动自动建模", type="primary", use_container_width=True)
    stop_btn = st.button("🛑 停止任务", type="secondary", use_container_width=True)


# --- 核心逻辑函数 ---

def parse_log_metrics(log_content):
    """从日志中解析进化指标 (增强鲁棒性版)"""
    metrics = []
    
    # 使用 split 将日志按世代切分，相关性更强，防止错位
    # split后结构: [前文, 世代号1, 世代1内容, 世代号2, 世代2内容, ...]
    parts = re.split(r"--- 世代 (\d+)/\d+ ---", log_content)
    
    # 从索引1开始遍历 (索引0是第一代之前的内容，通常是初始化日志)
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts): break
        
        try:
            gen_num = int(parts[i])
            block_content = parts[i+1]
            
            # 在该世代的内容块中查找 AUC 统计
            # 格式参考: [统计] 平均基础AUC: 0.8123 (最高: 0.8456)
            m_auc = re.search(r"\[统计\] 平均基础AUC: ([\d\.]+) \(最高: ([\d\.]+)\)", block_content)
            
            if m_auc:
                avg_auc = float(m_auc.group(1))
                max_auc = float(m_auc.group(2))
                
                # 查找综合得分 (非必须)
                max_score = 0.0
                m_score = re.search(r"\[统计\] 平均综合得分: ([\d\.]+) \(最高: ([\d\.]+)\)", block_content)
                if m_score:
                    max_score = float(m_score.group(2))
                    
                metrics.append({
                    "Generation": gen_num,
                    "AUC": max_auc,      # 每一代的最高AUC
                    "Avg AUC": avg_auc,  # 平均AUC (留作参考)
                    "Max Score": max_score
                })
        except Exception:
            continue
            
    return pd.DataFrame(metrics)

def run_automl_process(generations, population, csv_folder, main_table, target_col):
    """启动AutoML子进程"""
    # 清理旧日志 (增加强力重试机制)
    if os.path.exists(LOG_FILE):
        try:
            os.remove(LOG_FILE)
        except PermissionError:
            # 如果删不掉，说明上次进程还占着，尝试清空内容而不是删除文件
            try:
                with open(LOG_FILE, 'w') as f:
                    f.truncate(0)
            except Exception:
                pass # 实在不行就忽略，追加写入也无所谓
    
    # 确定脚本路径 (兼容从根目录运行或从src运行)
    if os.path.exists("src/main.py"):
        script_path = "src/main.py"
        cwd_path = "." # 根目录
        python_path_env = "src"
    elif os.path.exists("main.py"):
        script_path = "main.py" 
        cwd_path = "."
        python_path_env = "."
    else:
        st.error("找不到 main.py，请确保在项目根目录运行")
        return None

    cmd = [
        sys.executable, script_path, 
        "--limit", "1000",
        "--generations", str(generations),
        "--population", str(population),
        "--csv_dir", str(csv_folder),
        "--main_table", str(main_table),
        "--target_column", str(target_col)
    ]
    
    # 设置环境变量以确保模块导入正常和编码正确
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8" # 告诉 Python 进程输出 UTF-8
    env["PYTHONLEGACYWINDOWSSTDIO"] = "utf-8" # 尝试修复 Windows 下的 legacy stdio
    env["NO_LOG_FILE"] = "1" # 告诉主程序不要自己写文件，因为我们会重定向输出
    if python_path_env:
        env["PYTHONPATH"] = python_path_env + os.pathsep + env.get("PYTHONPATH", "")
    
    # 打开日志文件，将子进程的所有输出（stdout和stderr）直接重定向到文件
    # 这样可以确保 crash traceback 也会被写入日志，且没有 PIPE 缓冲区满的问题
    log_f = open(LOG_FILE, "w", encoding="utf-8", errors="replace", buffering=1) # 行缓冲

    process = subprocess.Popen(
        cmd,
        stdout=log_f,        # 直接写入文件
        stderr=subprocess.STDOUT, # stderr 也并入 stdout
        # encoding="utf-8",  # 不再需要，因为 stdout 不是 PIPE
        # errors="replace", 
        bufsize=1,
        cwd=cwd_path,
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        close_fds=False # Windows下必须为False才能继承句柄? 默认False
    )
    return process

# --- 结果展示辅助函数 ---
def load_feature_list():
    if FEATURES_FILE.exists():
        try:
            with open(FEATURES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            features = data.get("features", data)
            if isinstance(features, list):
                return features
        except Exception:
            pass
    return []

def load_feature_importance():
    if not PIPELINE_FILE.exists():
        return None, "未找到模型文件，无法计算特征重要度"

    try:
        pipeline = joblib.load(PIPELINE_FILE)
    except Exception as e:
        return None, f"模型加载失败: {e}"

    model = None
    preprocessor = None
    if hasattr(pipeline, "named_steps"):
        model = pipeline.named_steps.get("classifier")
        preprocessor = pipeline.named_steps.get("preprocessor")
    if model is None:
        model = pipeline

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return None, "当前模型不支持特征重要度"

    feature_names = None
    if preprocessor is not None:
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = None

    if feature_names is None:
        feature_names = load_feature_list()
        if not feature_names:
            feature_names = [f"feature_{i+1}" for i in range(len(importances))]

    # 长度不一致时兜底截断
    if len(feature_names) != len(importances):
        min_len = min(len(feature_names), len(importances))
        feature_names = feature_names[:min_len]
        importances = importances[:min_len]

    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    return df_imp, None

# --- 状态管理 ---
if 'running' not in st.session_state:
    st.session_state.running = False
if 'logs' not in st.session_state:
    st.session_state.logs = ""

# --- 主界面 ---
st.title("🧩 自动化特征工程与模型进化系统")
st.markdown("**Core Engine**: Genetic Programming | **Model**: LightGBM | **Strategy**: Adversarial Co-Evolution")

# 如果点击启动
if start_btn and not st.session_state.running:
    st.session_state.running = True
    st.session_state.process = run_automl_process(generations, population, csv_folder, main_table_input, target_col_input)
    st.rerun()

# 停止逻辑
if stop_btn and st.session_state.running:
    if st.session_state.process:
        st.session_state.process.terminate()
    st.session_state.running = False
    st.warning("任务已手动停止")

# --- 实时监控面板 (当正在运行或有日志时) ---
tab1, tab2, tab3 = st.tabs(["📈 进化监控", "🧬 冠军基因", "💻 自动生成代码"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("进化轨迹 (Evolution Metrics)")
        chart_placeholder = st.empty()
        
    with col2:
        st.subheader("实时日志流")
        log_box = st.empty()
        
    # 运行中仅展示日志，AUC 曲线在任务结束后统一生成
    if st.session_state.running:
        chart_placeholder.info("任务运行中：AUC 曲线将在运行结束后生成")
        while True:
            # 读取新日志
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                    st.session_state.logs = content
                
            # 更新日志窗
            log_lines = st.session_state.logs.split('\n')
            last_lines = '\n'.join(log_lines[-20:])
            log_box.code(last_lines, language="text")
            
            # 检查进程是否结束
            if st.session_state.process.poll() is not None:
                st.session_state.running = False
                st.success("进化任务完成！")
                st.rerun()
                break
                
            time.sleep(1)
            
    else:
        # 静态展示（回放）
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            df_metrics = parse_log_metrics(content)
            if not df_metrics.empty:
                fig = px.line(df_metrics, x="Generation", y="AUC", 
                             title="历史任务：模型迭代效果 (AUC)", markers=True)
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### 📄 详细数据表")
                df_table = df_metrics[["Generation", "AUC", "Avg AUC"]].copy()
                df_table["AUC"] = df_table["AUC"].round(4)
                df_table["Avg AUC"] = df_table["Avg AUC"].round(4)
                st.dataframe(df_table, use_container_width=True)
            else:
                st.info("暂无进化历史数据")
        else:
            st.info("请点击左侧 [启动] 按钮开始任务")

with tab2:
    st.subheader("🏆 冠军染色体解析 (Champion DNA)")
    if CHAMPION_FILE.exists():
        with open(CHAMPION_FILE, "r", encoding="utf-8") as f:
            js = json.load(f)
        
        # 显示元数据
        if "meta" in js:
            meta = js["meta"]
            c1, c2, c3 = st.columns(3)
            c1.metric("主表", meta.get("main_table", "N/A"))
            c2.metric("目标列", meta.get("target_column", "N/A"))
            c3.metric("生成时间", meta.get("generated_at", "N/A"))
        
        st.markdown("#### 🧬 基因结构可视化")
        genes = js.get("genes", [])
        
        # 将基因转为 DataFrame 展示
        gene_data = []
        for g in genes:
            g_type = g.get("_type", "Unknown")
            # 提取关键信息
            desc = ""
            if g_type == "FeatureGene":
                op = g.get("op", "?")
                path = g.get("path", "?")
                window = g.get("window")
                desc = f"{op}({path})" + (f"  window={window}" if window else "")
            elif g_type == "ModelGene":
                alg = g.get("alg", "?")
                params = g.get("params", {})
                key_params = {k: v for k, v in params.items() if k in ("n_estimators", "learning_rate", "num_leaves", "is_unbalance")}
                desc = f"{alg}  {key_params}"
            elif g_type == "GroupByFeatureGene":
                desc = f"{g.get('agg_func')} ( {g.get('target_col')} ) by {g.get('group_cols')}"
            elif g_type == "CrossFeatureGene":
                desc = f"{g.get('operator')} ( {', '.join(g.get('features', []))} )"
            else:
                desc = str(g)
            
            gene_data.append({"Type": g_type, "Description": desc, "Enabled": g.get("enabled", True)})
            
        st.dataframe(pd.DataFrame(gene_data), use_container_width=True)

        # 展示冠军特征列表
        st.markdown("#### 📌 冠军特征列表")
        feature_list = load_feature_list()
        if feature_list:
            st.dataframe(pd.DataFrame({"Feature": feature_list}), use_container_width=True)
        else:
            st.info("未找到 expected_feature_columns.json，无法展示冠军特征列表")

        # 展示特征重要度
        st.markdown("#### ⭐ 特征重要度 (Top 50)")
        df_imp, imp_err = load_feature_importance()
        if df_imp is not None and not df_imp.empty:
            df_imp_show = df_imp.head(50).copy()
            df_imp_show["Importance"] = df_imp_show["Importance"].round(6)
            st.dataframe(df_imp_show, use_container_width=True)
        else:
            st.info(imp_err or "暂无可用的特征重要度")
        
        with st.expander("查看原始 JSON"):
            st.json(js)
    else:
        st.warning("暂未生成冠军模型，请先运行任务。")

with tab3:
    st.subheader("🚀 自动化工程代码交付")
    
    col_sql, col_py = st.columns(2)
    
    with col_sql:
        st.markdown("### 1. 生产环境 SQL")
        st.caption("Auto-generated optimized SQL for Data Warehouse")
        sql_files = list(SAVED_MODEL_DIR.glob("production_query*.sql"))
        if sql_files:
            # 优先显示实际表名的，如果没有则显示默认的
            selected_sql = st.selectbox("选择 SQL 文件", [f.name for f in sql_files])
            with open(SAVED_MODEL_DIR / selected_sql, "r", encoding="utf-8") as f:
                st.code(f.read(), language="sql")
        else:
            st.info("等待 SQL 生成...")

    with col_py:
        st.markdown("### 2. 敏捷平台发布脚本")
        st.caption("Ready-to-deploy Python script for online inference")
        py_file = SAVED_MODEL_DIR / "platform_inference_template.py"
        if py_file.exists():
            with open(py_file, "r", encoding="utf-8") as f:
                st.code(f.read(), language="python")
        else:
            st.info("等待脚本生成...")

# --- 底部 Footer ---
st.markdown("---")
st.caption("© 2026 AI Lab | Powered by Evolutionary AutoML")
