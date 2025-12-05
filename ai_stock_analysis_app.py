import streamlit as st
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import io
import time
from google import genai
from datetime import datetime
# 移除 tushare 导入

# =============================== 配置区域 ===============================
# 替换为您的 Gemini API Key
# 商业应用中，此 Key 必须通过环境变量或安全配置服务加载，不应硬编码。
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
LLM_MODEL = "gemini-2.5-flash" 

# 移除 Tushare Pro Token 配置
# =====================================================================

# 初始化 Gemini 客户端
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Gemini API 客户端初始化失败: {e}")

# 移除 Tushare 客户端初始化


# ----------------- I. 大盘宏观分析模块 -----------------

def get_market_summary_data():
    """获取大盘和行业板块数据 (含双重接口容错)"""
    st.info("📡 正在获取上证指数 K 线和行业板块数据...")
    
    today = datetime.now().strftime('%Y%m%d')
    start_date_ak = (datetime.now() - pd.Timedelta(days=90)).strftime('%Y%m%d')
    
    index_data = None
    
    # === 1. 获取上证指数 K 线 (双重容错：Akshare-DFCF > Akshare-SINA) ===
    
    # 方案 A (Primary): 东方财富接口 (index_zh_a_hist)
    try:
        st.info("📡 尝试通过 Akshare-东方财富获取上证指数 K 线...")
        index_data = ak.index_zh_a_hist(symbol="000001", period="daily", start_date=start_date_ak, end_date=today)
        # 清洗数据
        if '日期' in index_data.columns:
            index_data['日期'] = pd.to_datetime(index_data['日期'])
            index_data.set_index('日期', inplace=True)
        index_data = index_data[['收盘', '成交量']]
        st.info("✅ 上证指数 K 线数据获取成功 (东方财富/Akshare)")
        
    except Exception as e_dfcf:
        st.warning(f"Akshare-东方财富连接断开 ({e_dfcf}), 正在切换至备用数据源 (新浪)...")
        
        # 方案 B: 新浪财经接口 (stock_zh_index_daily) - 备用
        try:
            time.sleep(1) # 缓冲一下
            index_data = ak.stock_zh_index_daily(symbol="sh000001")
            
            # 新浪数据清洗
            if 'date' in index_data.columns:
                index_data['date'] = pd.to_datetime(index_data['date'])
                index_data.set_index('date', inplace=True)
            
            # 统一列名 (新浪返回的是英文列名)
            index_data = index_data.rename(columns={'close': '收盘', 'volume': '成交量'})
            # 截取最近 90 天
            index_data = index_data.sort_index().tail(90)
            index_data = index_data[['收盘', '成交量']]
            st.info("✅ 上证指数 K 线数据获取成功 (新浪/Akshare)")
            
        except Exception as e_sina:
            st.error(f"获取上证指数失败 (所有接口均已尝试): {e_sina}")
            index_data = None


    # === 2. 获取行业板块涨幅榜 (双重容错) ===
    industry_df = None
    
    # 方案 A (Primary): 东方财富行业板块，带重试
    for attempt in range(3):
        try:
            st.info(f"📡 尝试通过东方财富获取行业板块数据 (第 {attempt + 1} 次)...")
            industry_board = ak.stock_board_industry_spot_em()
            industry_df = industry_board[['名称', '涨跌幅']].sort_values(by='涨跌幅', ascending=False).head(10)
            st.info("✅ 行业板块数据获取成功 (东方财富)")
            break # 成功则跳出重试循环
        except Exception as e:
            if attempt == 2:
                st.warning(f"东方财富行业板块接口失败 ({e})，尝试切换至备用接口...")
            time.sleep(0.5)
            
    # 方案 B (Fallback): 东方财富概念板块 (作为市场热点代理)
    if industry_df is None:
        try:
            st.info("📡 切换至备用接口：东方财富概念板块...")
            concept_board = ak.stock_board_concept_spot_em()
            
            # 数据清洗：选择概念名称和涨跌幅，并排序
            industry_df = concept_board[['名称', '涨跌幅']].sort_values(by='涨跌幅', ascending=False).head(10)
            
            st.warning("⚠️ 已切换至东方财富**概念板块**数据作为市场热点分析，请知悉。")

        except Exception as e:
            st.error(f"获取行业板块数据失败 (所有接口均已尝试): {e}")
            return index_data, None

    return index_data, industry_df

def generate_market_analysis(industry_df):
    """调用 Gemini 分析大盘走势"""
    if industry_df is None:
        return "数据获取失败，无法生成报告。"
        
    industry_str = industry_df.to_string(index=False)

    prompt = f"""
    你是一位经验丰富的 A 股市场首席策略分析师，风格专业、观点犀利。
    以下是今日 A 股行业板块涨幅 Top 10 的数据：
    {industry_str}

    请根据这些数据，生成一份《今日大盘宏观分析与明日预测》报告。
    要求：
    1. **大盘定调：** 总结今日市场是情绪主导还是价值主导，资金流向何处。
    2. **核心主线：** 分析涨幅榜 Top 3 行业，确定市场主线。
    3. **明日预测：** 给出对明日走势的定性预测（看多/看空/震荡），并说明策略建议。
    4. **格式：** 使用 Markdown 格式，分段清晰。
    """
    
    try:
        with st.spinner("🧠 Gemini 正在进行大盘总结与预测..."):
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config={"temperature": 0.7}
            )
            return response.text
    except Exception as e:
        return f"❌ Gemini API 调用失败: 请检查 Key 或网络。错误信息: {e}"

# ----------------- II. 个股价值分析模块 -----------------

def get_stock_fund_data(symbol):
    """获取个股近期资金流向数据"""
    st.info(f"📡 正在获取 {symbol} 的主力资金流向...")
    
    # 增加重试机制
    for attempt in range(3):
        try:
            # 修正：ak.stock_individual_fund_flow 接口不支持日期筛选参数（位置或关键字都不支持）。
            # 必须先获取全量历史数据，然后在 Pandas 中截取最新的数据。
            fund_data_history = ak.stock_individual_fund_flow(stock=symbol)
            
            if fund_data_history.empty:
                return None, "无法获取资金流历史数据。"

            # 数据预处理：确保按日期排序
            if '日期' in fund_data_history.columns:
                fund_data_history['日期'] = pd.to_datetime(fund_data_history['日期'])
                fund_data_history.sort_values('日期', ascending=True, inplace=True)
                # 格式化日期为字符串，方便展示
                fund_data_history['日期'] = fund_data_history['日期'].dt.strftime('%Y-%m-%d')
                fund_data_history.set_index('日期', inplace=True)
            
            # 取最新的 5 个交易日数据
            latest_fund_data = fund_data_history.tail(5)
            
            return latest_fund_data, None 
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
                continue
            return None, f"获取个股资金流向失败: {e}"

def generate_stock_analysis(symbol, fund_data):
    """调用 Gemini 分析个股和主力意图"""
    if fund_data is None or fund_data.empty:
        return "数据获取失败，无法生成个股报告。"

    # 将包含日期索引的数据框转换为字符串
    fund_str = fund_data.to_string()
    
    prompt = f"""
    你是一位顶尖的 A 股量化交易员，擅长从资金流判断主力意图。
    以下是股票代码 {symbol} 最近 5 个交易日的主力资金流指标数据（日期作为索引）：
    {fund_str}

    请根据这些资金流数据，生成一份《个股主力资金意图分析报告》。
    要求：
    1. **主力意图：** 定性分析主力资金在最近 5 个交易日的行为是“吸筹”、“震荡洗盘”还是“派发/出货”，并说明理由（参考净流入额和趋势）。
    2. **交易建议：** 给出基于资金流的短期交易策略（例如：持股观望、逢高减仓等）。
    3. **格式：** 使用 Markdown 格式，结论清晰，观点明确。
    """
    
    try:
        with st.spinner(f"🧠 Gemini 正在分析 {symbol} 的主力资金意图..."):
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config={"temperature": 0.7}
            )
            return response.text
    except Exception as e:
        return f"❌ Gemini API 调用失败: 错误信息: {e}"

# ----------------- III. Streamlit UI 界面 -----------------

def plot_index_kline(df):
    """绘制简单的收盘价趋势图"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df['收盘'], label='上证指数收盘价', color='blue')
    ax.set_title('近90日上证指数收盘价趋势')
    ax.set_xlabel('日期')
    ax.set_ylabel('收盘价')
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

def main_app():
    st.set_page_config(page_title="AI 自动化投资分析工具", layout="wide")
    
    st.title("🤖 AI 自动化 A 股投资分析系统")
    st.subheader("由 Akshare 数据驱动，Gemini AI 智能分析")

    # 使用侧边栏输入配置
    st.sidebar.header("配置与说明")
    st.sidebar.write("本工具用于演示基于 Gemini API 的 A 股量化分析。")
    
    # 侧边栏导航
    analysis_type = st.sidebar.radio(
        "选择分析模块",
        ("一、大盘宏观分析", "三、个股价值分析"),
        index=0
    )

    if analysis_type == "一、大盘宏观分析":
        st.header("📈 大盘走势与行业热点分析")
        
        if st.button("🚀 开始分析今日大盘"):
            index_data, industry_df = get_market_summary_data()
            
            if index_data is not None:
                st.subheader("1. 上证指数近期走势")
                plot_index_kline(index_data)
            
            if industry_df is not None:
                st.subheader("2. 今日行业涨幅榜 Top 10")
                st.dataframe(industry_df, use_container_width=True)

                st.subheader("3. AI 宏观分析报告")
                report = generate_market_analysis(industry_df)
                st.markdown(report)
                
            else:
                st.error("数据获取失败，无法继续分析。")


    elif analysis_type == "三、个股价值分析":
        st.header("🔍 个股资金流与主力意图分析")
        
        stock_symbol = st.text_input("请输入股票代码 (如 000001)", value="600519") # 默认贵州茅台

        if st.button("🕵️‍♀️ 开始个股分析"):
            if len(stock_symbol) == 6 and stock_symbol.isdigit():
                # 模块四：主力资金监控的实现
                fund_data, error_msg = get_stock_fund_data(stock_symbol)

                if error_msg:
                    st.error(error_msg)
                else:
                    st.subheader(f"1. 股票 {stock_symbol} 资金流指标 (最近 5 个交易日)")
                    # 确保 dataframe 带有日期索引
                    st.dataframe(fund_data, use_container_width=True)
                    
                    st.subheader("2. AI 主力意图分析报告")
                    report = generate_stock_analysis(stock_symbol, fund_data)
                    st.markdown(report)

            else:
                st.error("请输入有效的 6 位数字股票代码。")

if __name__ == '__main__':
    main_app()
