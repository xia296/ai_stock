import streamlit as st
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import time
from google import genai
from datetime import datetime

# =============================== 配置区域 ===============================
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
LLM_MODEL = "gemini-2.5-flash" 
TUSHARE_TOKEN = "637d755d94bbd75d12af19b6b9b87a8c6934b0f1f85d39e211f1048a"

# 初始化 Gemini 客户端
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Gemini API 客户端初始化失败: {e}")

# =============================== 核心量化抓取模块 ===============================

def get_short_term_sentiment():
    """获取短线情绪数据：涨停池、连板高度、人气榜"""
    try:
        # 1. 涨停池（今日最强风向标）
        date_str = datetime.now().strftime('%Y%m%d')
        zt_pool = ak.stock_zt_pool_em(date=date_str)
        
        # 2. 东方财富人气榜（散户关注度）
        hot_rank = ak.stock_hot_rank_em()
        
        return zt_pool, hot_rank
    except Exception as e:
        st.warning(f"实时短线数据获取失败（可能是非交易日）: {e}")
        return None, None

def generate_opportunity_analysis(time_slot, zt_data, hot_data):
    """
    量化短线机会扫描引擎
    time_slot: 早盘/午盘/收盘
    """
    if zt_data is None or zt_data.empty:
        return "当前非交易时段或数据未更新，无法进行机会扫描。"

    # 简化的量化特征提取
    zt_count = len(zt_data)
    highest_limit = zt_data['连板数'].max() if '连板数' in zt_data.columns else "N/A"
    hot_top5 = hot_data.head(5)['代码'].tolist() if hot_data is not None else []
    
    # 模拟量化因子分析
    prompt = f"""
    你是一名精通商业量化的 A 股短线顶级操盘手。现在是【{time_slot}】阶段。
    
    【实时量化数据摘要】：
    1. 涨停总数：{zt_count} 家
    2. 市场最高连板高度：{highest_limit} 板
    3. 人气前五标的代码：{hot_top5}
    4. 涨停池抽样（名称/所属行业/连板）：
    {zt_data[['名称', '所属行业', '连板数']].head(10).to_string(index=False)}

    请从【商业量化短线视角】进行深度扫描：
    1. **情绪水位测算**：根据连板高度和涨停家数，判断当前处于（破冰/确立/发酵/高潮/退潮）哪个阶段？
    2. **机会扫描（核心）**：
       - 若是早盘：关注竞价超预期、一字板封单逻辑。
       - 若是午盘：寻找“回手掏”低吸机会或下午带队突围的新题材。
       - 若是收盘：识别龙虎榜含金量，锁定明日“断板必杀”或“弱转强”标的。
    3. **风险警示**：识别哪些高位股在“派发”，哪些板块在“抽血”。
    
    要求：用词犀利、干货直接，符合职业操盘手晨会/内参风格。
    """
    
    try:
        with st.spinner(f"🧠 量化引擎正在扫描{time_slot}机会..."):
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config={"temperature": 0.3} # 降低随机性，更偏向量化逻辑
            )
            return response.text
    except Exception as e:
        return f"❌ 扫描失败: {e}"

# ----------------- III. 界面增强 -----------------

def main_app():
    st.set_page_config(page_title="商业量化短线决策系统", layout="wide")
    
    st.title("⚡ 商业量化短线决策系统")
    st.sidebar.markdown(f"**交易员模式：活跃**")
    
    # 侧边栏导航
    menu = st.sidebar.radio(
        "功能模块",
        ("🔥 每日机会扫描", "📈 大盘宏观分析", "🔍 个股主力意图")
    )

    # --- 模块：每日机会扫描 ---
    if menu == "🔥 每日机会扫描":
        st.header("🎯 盘中实时机会扫描")
        
        # 自动识别当前时段
        current_hour = datetime.now().hour
        if current_hour < 11:
            default_slot = "早盘（竞价与首板）"
        elif current_hour < 15:
            default_slot = "午盘（转折与加强）"
        else:
            default_slot = "收盘（复盘与定调）"
            
        time_slot = st.select_slider(
            "手动切换时段",
            options=["早盘（竞价与首板）", "午盘（转折与加强）", "收盘（复盘与定调）"],
            value=default_slot
        )

        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("📡 启动全市场雷达扫描"):
                zt_data, hot_data = get_short_term_sentiment()
                
                if zt_data is not None:
                    st.success("数据链路已接通")
                    st.metric("涨停家数", len(zt_data))
                    st.subheader("🔥 实时人气榜 Top 10")
                    st.dataframe(hot_data.head(10)[['代码', '名称']], use_container_width=True)
                    
                    st.session_state['zt_data'] = zt_data
                    st.session_state['hot_data'] = hot_data
                else:
                    st.error("无法获取实时数据，请确认是否在交易时间或 API 限制。")

        with col2:
            if 'zt_data' in st.session_state:
                report = generate_opportunity_analysis(
                    time_slot, 
                    st.session_state['zt_data'], 
                    st.session_state['hot_data']
                )
                st.markdown("### 📝 AI 操盘手内参")
                st.markdown(report)
                
                # 可视化连板梯队
                st.subheader("📊 连板梯队分布")
                ladder = st.session_state['zt_data']['连板数'].value_counts().sort_index(ascending=False)
                st.bar_chart(ladder)

    # --- 模块：大盘分析 (保留并优化逻辑) ---
    elif menu == "📈 大盘宏观分析":
        # ... (此处沿用您原有的 get_market_summary_data 逻辑)
        st.info("大盘宏观分析模块：侧重于择时与仓位管理建议。")
        # 您原有的代码逻辑...

    # --- 模块：个股分析 (保留并优化逻辑) ---
    elif menu == "🔍 个股主力意图":
        # ... (此处沿用您原有的 get_stock_fund_data 逻辑)
        st.info("个股深度分析：结合分时异动与大单净量。")
        # 您原有的代码逻辑...

if __name__ == '__main__':
    main_app()