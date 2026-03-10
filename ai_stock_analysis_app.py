import streamlit as st
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import time
from google import genai
from datetime import datetime

# =============================== 配置区域 ===============================
# Gemini API Key (建议在 Streamlit Cloud 的 Secrets 中配置)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "YOUR_FALLBACK_API_KEY")
LLM_MODEL = "gemini-2.5-flash" 
# =====================================================================

# 初始化 Gemini 客户端
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Gemini API 客户端初始化失败: {e}")

# ----------------- I. 大盘与热点数据抓取 (纯 Akshare) -----------------

def get_market_summary_data():
    """获取大盘和行业板块数据 (Akshare 双重容错)"""
    st.info("📡 正在获取上证指数 K 线和行业板块数据...")
    
    today = datetime.now().strftime('%Y%m%d')
    start_date_ak = (datetime.now() - pd.Timedelta(days=90)).strftime('%Y%m%d')
    
    index_data = None
    
    # === 1. 获取上证指数 K 线 (东方财富优先，新浪备用) ===
    try:
        st.info("📡 尝试通过 Akshare-东方财富获取上证指数...")
        index_data = ak.index_zh_a_hist(symbol="000001", period="daily", start_date=start_date_ak, end_date=today)
        if '日期' in index_data.columns:
            index_data['日期'] = pd.to_datetime(index_data['日期'])
            index_data.set_index('日期', inplace=True)
        index_data = index_data[['收盘', '成交量']]
        st.info("✅ 上证指数数据获取成功 (东方财富)")
        
    except Exception as e_dfcf:
        st.warning(f"东方财富接口异常 ({e_dfcf}), 正在切换至新浪财经备用接口...")
        try:
            time.sleep(1)
            index_data = ak.stock_zh_index_daily(symbol="sh000001")
            if 'date' in index_data.columns:
                index_data['date'] = pd.to_datetime(index_data['date'])
                index_data.set_index('date', inplace=True)
            index_data = index_data.rename(columns={'close': '收盘', 'volume': '成交量'})
            index_data = index_data.sort_index().tail(90)[['收盘', '成交量']]
            st.info("✅ 上证指数数据获取成功 (新浪财经)")
        except Exception as e_sina:
            st.error(f"获取上证指数失败 (所有接口均已尝试): {e_sina}")

    # === 2. 获取行业板块实时盘口 ===
    industry_df = None
    for attempt in range(3):
        try:
            industry_board = ak.stock_board_industry_spot_em()
            # 提取名称、涨幅、市值、换手率，增加量化维度的判断依据
            industry_df = industry_board[['名称', '涨跌幅', '总市值', '换手率']].sort_values(by='涨跌幅', ascending=False).head(15)
            break 
        except Exception as e:
            if attempt == 2:
                st.warning(f"获取行业板块数据失败 ({e})")
            time.sleep(0.5)

    return index_data, industry_df

def get_current_market_phase():
    """判断当前属于短线的哪个交易周期"""
    now = datetime.now()
    current_time = now.time()
    
    if current_time < datetime.strptime("11:00", "%H:%M").time():
        return "早盘 (开盘-11:00) —— 情绪定调与试错期"
    elif current_time < datetime.strptime("14:00", "%H:%M").time():
        return "午盘 (11:00-14:00) —— 分歧检验与承接期"
    else:
        return "收盘/尾盘 (14:00-收盘) —— 资金沉淀与次日预期博弈期"

# ----------------- II. 智能分析模块 -----------------

def generate_sector_scan(industry_df, market_phase):
    """调用 Gemini 进行板块短线盘中机会扫描"""
    if industry_df is None:
        return "数据获取失败，无法生成报告。"
        
    industry_str = industry_df.to_string(index=False)
    
    prompt = f"""
    你是一位顶尖的 A 股短线量化交易员，深谙资金情绪、打板接力与板块轮动规律。
    当前盘面处于：【{market_phase}】
    
    以下是最新实时板块涨幅榜 Top 15 的数据（包含涨跌幅与换手率）：
    {industry_str}

    请根据当前**特定的交易时段**，生成一份《板块短线狙击与机会扫描报告》。
    要求：
    1. **时段定性分析**：
       - 如果是早盘：分析赚钱效应在哪，是否有主线逼空，提示追涨风险或首板试错方向。
       - 如果是午盘：分析早盘强势板块是否出现筹码松动，哪些低位板块有资金做高低切承接。
       - 如果是尾盘/收盘：分析资金在沉淀哪些板块博弈次日溢价，总结全天情绪周期（退潮/混沌/主升）。
    2. **量价与换手洞察**：结合换手率和涨幅，指出哪个板块是“真金白银”的攻击，哪个是缩量诱多。
    3. **短线操作纪律**：给出明确的“关注方向”和“规避雷区”。
    4. **格式：** 使用 Markdown，充满交易实战感，观点一针见血。
    """
    
    try:
        with st.spinner(f"🧠 Gemini 正在进行【{market_phase.split(' ')[0]}】短线机会扫描..."):
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config={"temperature": 0.8}
            )
            return response.text
    except Exception as e:
        return f"❌ Gemini API 调用失败: {e}"

def generate_market_analysis(industry_df):
    """调用 Gemini 分析大盘宏观走势"""
    if industry_df is None: return "数据获取异常。"
    industry_str = industry_df.to_string(index=False)
    
    prompt = f"""
    你是一位经验丰富的 A 股市场首席策略分析师。以下是今日 A 股行业板块涨幅 Top 15：
    {industry_str}
    
    请生成一份《今日大盘宏观分析与明日预测》，要求：
    1. **大盘定调：** 总结今日市场是情绪主导还是价值主导，资金流向何处。
    2. **核心主线：** 分析涨幅榜前排行业，确定市场主线。
    3. **明日预测：** 给出对明日走势的定性预测（看多/看空/震荡），并说明策略建议。
    4. **格式：** Markdown 格式，分段清晰。
    """
    try:
        response = client.models.generate_content(model=LLM_MODEL, contents=prompt, config={"temperature": 0.7})
        return response.text
    except Exception as e:
        return f"调用失败: {e}"

def get_stock_fund_data(symbol):
    """获取个股近期资金流向数据"""
    st.info(f"📡 正在获取 {symbol} 的主力资金流向...")
    for attempt in range(3):
        try:
            fund_data_history = ak.stock_individual_fund_flow(stock=symbol)
            if fund_data_history.empty: return None, "无法获取资金流历史数据。"
            
            if '日期' in fund_data_history.columns:
                fund_data_history['日期'] = pd.to_datetime(fund_data_history['日期'])
                fund_data_history.sort_values('日期', ascending=True, inplace=True)
                fund_data_history['日期'] = fund_data_history['日期'].dt.strftime('%Y-%m-%d')
                fund_data_history.set_index('日期', inplace=True)
            
            return fund_data_history.tail(5), None 
        except Exception as e:
            if attempt < 2: 
                time.sleep(1)
                continue
            return None, f"获取失败: {e}"

def generate_stock_analysis(symbol, fund_data):
    """调用 Gemini 分析个股主力意图"""
    if fund_data is None or fund_data.empty: return "数据获取失败。"
    fund_str = fund_data.to_string()
    
    prompt = f"""
    你是一位顶尖的 A 股量化交易员，擅长识破主力资金的对倒与洗盘动作。
    以下是代码 {symbol} 最近 5 个交易日的主力资金流数据：
    {fund_str}
    
    请生成一份《个股主力资金意图穿透报告》。要求：
    1. **主力意图：** 定性分析资金近 5 日是在“吸筹”、“震荡洗盘”还是“派发/出货”。
    2. **交易建议：** 给出基于资金流的明确短期交易动作。
    3. **格式：** Markdown 格式。
    """
    try:
        response = client.models.generate_content(model=LLM_MODEL, contents=prompt, config={"temperature": 0.7})
        return response.text
    except Exception as e:
        return f"调用失败: {e}"

# ----------------- III. Streamlit UI 界面 -----------------

def plot_index_kline(df):
    """绘制极简指数走势图"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df['收盘'], label='上证指数收盘价', color='#d62728', linewidth=2)
    ax.set_title('近90日上证指数走势 (趋势跟踪)', fontdict={'fontsize': 14, 'fontweight': 'bold'})
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

def main_app():
    st.set_page_config(page_title="AI 量化游资系统", layout="wide")
    st.title("⚡ AI 自动化 A 股短线量化中枢")
    
    st.sidebar.header("🕹️ 战术指挥中心")
    analysis_type = st.sidebar.radio(
        "选择火力侦察模块",
        ("一、大盘宏观分析", "二、板块短线盘中扫描", "三、个股资金穿透"),
        index=1
    )

    if analysis_type == "一、大盘宏观分析":
        st.header("📈 大盘走势与宏观热点")
        if st.button("🚀 启动宏观扫描"):
            index_data, industry_df = get_market_summary_data()
            if index_data is not None:
                plot_index_kline(index_data)
            if industry_df is not None:
                st.dataframe(industry_df.head(10), use_container_width=True)
                st.markdown(generate_market_analysis(industry_df))

    elif analysis_type == "二、板块短线盘中扫描":
        st.header("🎯 板块每日机会扫描 (动态时段)")
        phase = get_current_market_phase()
        st.info(f"⏱️ **当前盘面时区判定：** {phase}")
        
        if st.button("🔥 提取当前时段爆点机会"):
            _, industry_df = get_market_summary_data()
            if industry_df is not None:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.subheader("实时异动板块前排")
                    st.dataframe(industry_df, use_container_width=True)
                with col2:
                    st.subheader("🤖 交易员 AI 决断")
                    report = generate_sector_scan(industry_df, phase)
                    st.markdown(report)
            else:
                st.error("盘口数据抓取失败。")

    elif analysis_type == "三、个股资金穿透":
        st.header("🔍 个股资金流与主力底牌分析")
        stock_symbol = st.text_input("输入 6 位股票代码狙击 (如 000001)", value="600519")
        if st.button("🕵️‍♀️ 穿透主力资金"):
            if len(stock_symbol) == 6 and stock_symbol.isdigit():
                fund_data, error_msg = get_stock_fund_data(stock_symbol)
                if not error_msg:
                    st.dataframe(fund_data, use_container_width=True)
                    st.markdown(generate_stock_analysis(stock_symbol, fund_data))
                else:
                    st.error(error_msg)
            else:
                st.warning("请检查代码格式，仅支持 6 位纯数字。")

if __name__ == '__main__':
    main_app()