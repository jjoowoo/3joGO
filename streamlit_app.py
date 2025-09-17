import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import io
from datetime import datetime

st.set_page_config(
    page_title="글로벌 + 한국 해수면 상승 대시보드",
    layout="wide"
)

# --- 글로벌 데이터 로드 ---
@st.cache_data(ttl=3600)
def load_global_data():
    url = "https://datahub.io/core/sea-level-rise/r/epa-sea-level.csv"
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        df = df.rename(columns={'Year':'date','CSIRO Adjusted Sea Level':'value'})
        df['date'] = pd.to_datetime(df['date'].astype(str) + "-01-01")
        df['group'] = 'Global Mean Sea Level'
        return df[['date','value','group']]
    except Exception:
        years = np.arange(1993,2025)
        sea_level = 3.5*(years-1993)+np.random.randn(len(years))*2
        df = pd.DataFrame({
            'date': pd.to_datetime([f'{y}-01-01' for y in years]),
            'value': sea_level,
            'group': 'Global Mean (Sample)'
        })
        return df

global_df = load_global_data()

# --- 한국 데이터 생성 ---
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
            sea_level = max(base_level + random_noise, 0.1)
            data.append({
                'year': year,
                'location': loc,
                'latitude': lat,
                'longitude': lon,
                'sea_level': round(sea_level,2)
            })
    return pd.DataFrame(data)

korea_df = create_korea_data()

# --- CSV 다운로드 함수 ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')

# --- 사이드바 ---
st.sidebar.header("데이터 필터")
# 글로벌 기간 선택
g_start, g_end = st.sidebar.date_input(
    "글로벌 기간 필터",
    value=[global_df['date'].min().date(), global_df['date'].max().date()],
    min_value=global_df['date'].min().date(),
    max_value=global_df['date'].max().date()
)
g_start_dt = pd.to_datetime(g_start)
g_end_dt = pd.to_datetime(g_end)

# 한국 연도 선택
k_year_range = st.sidebar.slider(
    "한국 연도 범위 선택",
    int(korea_df['year'].min()),
    int(korea_df['year'].max()),
    (int(korea_df['year'].min()), int(korea_df['year'].max()))
)

# --- 1. 글로벌 평균 해수면 ---
st.title("🌊 글로벌 + 한국 해수면 상승 대시보드")
st.header("1. 글로벌 평균 해수면 변화")
filtered_global_df = global_df[(global_df['date']>=g_start_dt)&(global_df['date']<=g_end_dt)]
filtered_global_df['smoothed_value'] = filtered_global_df['value'].rolling(
    window=5, min_periods=1, center=True
).mean()

fig_global = px.line(
    filtered_global_df,
    x='date',
    y=['value','smoothed_value'],
    labels={'date':'연도','value':'해수면(mm)'},
    title="전지구 평균 해수면 변화",
    template="plotly_white"
)
fig_global.update_traces(patch={"name":"월별 데이터"},selector={"name":"value"})
fig_global.update_traces(patch={"name":"5년 이동평균"},selector={"name":"smoothed_value"})
st.plotly_chart(fig_global,use_container_width=True)

st.download_button(
    "📥 글로벌 데이터 다운로드",
    data=convert_df_to_csv(filtered_global_df),
    file_name="global_sea_level.csv",
    mime="text/csv"
)

# --- 2. 한국 지도 애니메이션 ---
st.markdown("---")
st.header("2. 한국 주요 연안 해수면 상승 지도 (Mapbox 배경)")
filtered_korea_df = korea_df[(korea_df['year']>=k_year_range[0]) & (korea_df['year']<=k_year_range[1])]

# 한국 통계
col1,col2,col3,col4 = st.columns(4)
with col1:
    avg_increase = round(filtered_korea_df['sea_level'].diff().mean(),2)
    st.metric("연평균 상승",f"{avg_increase} mm")
with col2:
    total_increase = round(filtered_korea_df['sea_level'].max()-filtered_korea_df['sea_level'].min(),2)
    st.metric("누적 상승",f"{total_increase} mm")
with col3:
    st.metric("관측 지점 수",filtered_korea_df['location'].nunique())
with col4:
    korea_mean = filtered_korea_df.groupby('year')['sea_level'].mean().mean()
    global_mean = filtered_global_df['value'].mean()
    factor = round(korea_mean/global_mean,2) if global_mean>0 else 0
    st.metric("글로벌 대비 속도",f"{factor}배")

# Mapbox 애니메이션 지도
fig_map = px.scatter_mapbox(
    filtered_korea_df,
    lat='latitude',
    lon='longitude',
    color='sea_level',
    size='sea_level',
    hover_name='location',
    hover_data=['year','sea_level'],
    animation_frame='year',
    color_continuous_scale='Blues',
    size_max=20,
    zoom=5,
    center={"lat":36.5,"lon":127.5},
    mapbox_style="carto-positron",
    title="한국 주요 연안 연도별 해수면 상승"
)
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

# --- 3. 한국 지점별 상승 추이 ---
st.subheader("한국 주요 연안 지점별 연도별 해수면 상승")
fig_korea_line = px.line(
    filtered_korea_df,
    x='year',
    y='sea_level',
    color='location',
    labels={'year':'연도','sea_level':'해수면(mm)'},
    title="지점별 상승 추이",
    template="plotly_white"
)
st.plotly_chart(fig_korea_line,use_container_width=True)

# --- 4. 글로벌 vs 한국 평균 비교 ---
st.markdown("---")
st.header("3. 글로벌 vs 한국 평균 비교")
korea_avg = filtered_korea_df.groupby('year')['sea_level'].mean().reset_index()
korea_avg['date'] = pd.to_datetime(korea_avg['year'].astype(str)+"-01-01")
korea_avg['group'] = '한국 평균'

combined_df = pd.concat([
    filtered_global_df[['date','value','group']].rename(columns={'value':'sea_level'}),
    korea_avg[['date','sea_level','group']]
])

fig_compare = px.line(
    combined_df,
    x='date',
    y='sea_level',
    color='group',
    labels={'date':'연도','sea_level':'해수면(mm)'},
    title="글로벌 평균 vs 한국 연안 평균 해수면 상승 비교",
    template="plotly_white"
)
st.plotly_chart(fig_compare,use_container_width=True)
