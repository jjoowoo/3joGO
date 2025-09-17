import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import io
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# --- 페이지 설정 ---
st.set_page_config(
    page_title="해수면 상승 데이터 대시보드",
    page_icon="🌊",
    layout="wide"
)

# --- 폰트 설정 ---
FONT_PATH = '/fonts/Pretendard-Bold.ttf'

def get_font_name():
    if os.path.exists(FONT_PATH):
        try:
            prop = fm.FontProperties(fname=FONT_PATH)
            return prop.get_name()
        except:
            return None
    return None

font_name = get_font_name()

if font_name:
    plt.rcParams['font.family'] = font_name
    st.markdown(f"""
    <style>
    @font-face {{
        font-family: 'Pretendard-Bold';
        src: url('file://{FONT_PATH}') format('truetype');
    }}
    html, body, [class*="st-"] {{
        font-family: 'Pretendard-Bold', sans-serif;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 데이터 로드/전처리 함수 ---
@st.cache_data(ttl=3600)
def load_global_data():
    url = "https://datahub.io/core/sea-level-rise/r/epa-sea-level.csv"
    try:
        response = requests.get(url)
        response.raise_for_status()
        csv_data = io.StringIO(response.text)
        df = pd.read_csv(csv_data)
        if "Year" in df.columns and "CSIRO Adjusted Sea Level" in df.columns:
            df = df.rename(columns={'Year':'date','CSIRO Adjusted Sea Level':'value'})
            df['date'] = pd.to_datetime(df['date'].astype(str) + "-01-01")
            df['group'] = 'Global Mean Sea Level'
            return df[['date','value','group']]
        else:
            raise ValueError("필요한 컬럼 없음")
    except Exception as e:
        st.warning(f"글로벌 데이터 로드 실패, 예제 데이터 사용 중: {e}")
        years = np.arange(1993, 2025)
        sea_level = 3.5*(years-1993) + np.random.randn(len(years))*2
        df = pd.DataFrame({
            'date': pd.to_datetime([f'{y}-01-01' for y in years]),
            'value': sea_level,
            'group': 'Global Mean (Sample)'
        })
        return df

@st.cache_data
def create_korea_data():
    years = np.arange(datetime.now().year-30, datetime.now().year+1)
    locations = {
        '서울': (37.5665,126.9783),
        '부산': (35.1796,129.0756),
        '인천': (37.4563,126.7052),
        '목포': (34.8113,126.3925),
        '울산': (35.5396,129.3114),
        '여수': (34.7600,127.6620)
    }
    data = []
    np.random.seed(42)
    for year in years:
        for loc, (lat, lon) in locations.items():
            base_level = 3.05*(year-years.min())
            random_noise = np.random.uniform(-0.5,0.5)
            sea_level = base_level + random_noise
            # 음수 방지
            sea_level = max(sea_level, 0.1)
            data.append({
                'year': year,
                'location': loc,
                'latitude': lat,
                'longitude': lon,
                'sea_level': round(sea_level,2)
            })
    df = pd.DataFrame(data)
    return df

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')

# --- 1. 글로벌 데이터 ---
st.title("🌊 해수면 상승 데이터 대시보드")
st.markdown("---")
st.header("1. 전지구 평균 해수면 변화")

global_df = load_global_data()

# 글로벌 날짜 필터
st.sidebar.header("📊 글로벌 데이터 설정")
g_start, g_end = st.sidebar.date_input(
    "글로벌 기간 필터",
    value=[global_df['date'].min().date(), global_df['date'].max().date()],
    min_value=global_df['date'].min().date(),
    max_value=global_df['date'].max().date()
)
g_start_dt = pd.to_datetime(g_start)
g_end_dt = pd.to_datetime(g_end)

smoothing_window = st.sidebar.slider(
    "글로벌 이동 평균 스무딩",1,24,5,help="데이터 부드럽게 표시"
)

filtered_global_df = global_df[(global_df['date']>=g_start_dt)&(global_df['date']<=g_end_dt)]
filtered_global_df['smoothed_value'] = filtered_global_df['value'].rolling(
    window=smoothing_window,min_periods=1,center=True
).mean()

fig_global = px.line(
    filtered_global_df,
    x='date',
    y=['value','smoothed_value'],
    labels={'date':'연도','value':'해수면 높이 (mm)'},
    title="전지구 평균 해수면 변화",
    template="plotly_white"
)
fig_global.update_traces(patch={"name":"월별 데이터"},selector={"name":"value"})
fig_global.update_traces(patch={"name":f"{smoothing_window}개월 이동평균"},selector={"name":"smoothed_value"})
if font_name:
    fig_global.update_layout(font=dict(family=font_name))
st.plotly_chart(fig_global,use_container_width=True)

st.download_button(
    "📈 글로벌 데이터 다운로드",
    data=convert_df_to_csv(filtered_global_df),
    file_name="global_sea_level.csv",
    mime="text/csv"
)

# --- 2. 한국 30년 데이터 ---
st.markdown("---")
st.header("2. 한국 연안 최근 30년 해수면 상승 데이터")

korea_df = create_korea_data()
st.sidebar.header("📊 한국 데이터 설정")
k_year_range = st.sidebar.slider(
    "한국 연도 범위 선택",
    int(korea_df['year'].min()),
    int(korea_df['year'].max()),
    (int(korea_df['year'].min()),int(korea_df['year'].max()))
)
filtered_korea_df = korea_df[(korea_df['year']>=k_year_range[0]) & (korea_df['year']<=k_year_range[1])]

# 한국 주요 지표
col1,col2,col3 = st.columns(3)
with col1:
    avg_increase = round(filtered_korea_df['sea_level'].diff().mean(),2)
    st.metric("한국 연안 연평균 해수면 상승",f"{avg_increase} mm")
with col2:
    total_increase = round(filtered_korea_df['sea_level'].max()-filtered_korea_df['sea_level'].min(),2)
    st.metric("한국 30년 누적 상승",f"{total_increase} mm")
with col3:
    st.metric("관측 지점 수",filtered_korea_df['location'].nunique())
st.markdown("<br>",unsafe_allow_html=True)

# 한국 지도 애니메이션
st.subheader("📊 한국 주요 연안 해수면 상승 지도 (애니메이션)")
fig_map = px.scatter_geo(
    filtered_korea_df,
    lat='latitude',
    lon='longitude',
    color='sea_level',
    size='sea_level',  # 이미 음수 방지 처리됨
    hover_name='location',
    hover_data=['year','sea_level'],
    animation_frame='year',
    projection="mercator",
    color_continuous_scale="Blues",
    title="한국 주요 연안 연도별 해수면 상승",
    size_max=20
)
fig_map.update_geos(fitbounds="locations",visible=False)
fig_map.update_layout(
    margin={"r":0,"t":50,"l":0,"b":0},
    coloraxis_colorbar=dict(title="해수면(mm)")
)
st.plotly_chart(fig_map,use_container_width=True)

st.download_button(
    "📥 한국 연안 데이터 다운로드",
    data=convert_df_to_csv(filtered_korea_df),
    file_name="korea_sea_level_30yrs.csv",
    mime="text/csv"
)