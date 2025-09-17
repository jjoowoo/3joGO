import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import io
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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

# --- 데이터 로드 및 전처리 ---
@st.cache_data(ttl=3600)
def load_sea_level_data():
    url = "https://datahub.io/core/sea-level-rise/r/epa-sea-level.csv"
    try:
        response = requests.get(url)
        response.raise_for_status()
        csv_data = io.StringIO(response.text)
        df = pd.read_csv(csv_data)

        # 컬럼명 확인 후 변환
        if "Year" in df.columns and "CSIRO Adjusted Sea Level" in df.columns:
            df = df.rename(columns={'Year':'date','CSIRO Adjusted Sea Level':'value'})
            df['date'] = pd.to_datetime(df['date'].astype(str) + "-01-01")
            df['group'] = '전지구 평균'
            return df[['date','value','group']]
        else:
            raise ValueError("필요한 컬럼이 없음")
    except Exception as e:
        st.warning(f"데이터 로드 실패, 예제 데이터 사용 중: {e}")
        # 예제 데이터 생성
        years = np.arange(1993, 2025)
        sea_level = 3.5*(years-1993) + np.random.randn(len(years))*2
        df = pd.DataFrame({
            'date': pd.to_datetime([f'{y}-01-01' for y in years]),
            'value': sea_level,
            'group': '전지구 평균 (예시)'
        })
        return df

def preprocess_data(df):
    df = df.dropna().drop_duplicates().sort_values('date')
    today = datetime.now().date()
    df = df[df['date'].dt.date <= today]
    return df

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')

# --- 1. 공식 공개 데이터 ---
st.title("🌊 해수면 상승 및 해양 환경 변화 대시보드")
st.markdown("---")
st.header("1. 공식 공개 데이터 기반 분석")

sea_level_df_raw = load_sea_level_data()
sea_level_df = preprocess_data(sea_level_df_raw)

# 사이드바
st.sidebar.header("📊 대시보드 설정")
start_date, end_date = st.sidebar.date_input(
    "기간 필터",
    value=[sea_level_df['date'].min().date(), sea_level_df['date'].max().date()],
    min_value=sea_level_df['date'].min().date(),
    max_value=sea_level_df['date'].max().date()
)
start_datetime = pd.to_datetime(start_date)
end_datetime = pd.to_datetime(end_date)

smoothing_window = st.sidebar.slider(
    '이동 평균 스무딩', 1, 24, 5,
    help="데이터를 부드럽게 만들어 장기 추세 파악"
)

filtered_df = sea_level_df[
    (sea_level_df['date'] >= start_datetime) &
    (sea_level_df['date'] <= end_datetime)
]
filtered_df['smoothed_value'] = filtered_df['value'].rolling(
    window=smoothing_window, min_periods=1, center=True
).mean()

st.subheader("📈 전지구 평균 해수면 변화 (1993년-현재)")
st.markdown("CSIRO 위성 고도계 데이터를 기반으로 한 전지구 평균 해수면(GMSL) 변화 추이")

fig1 = px.line(
    filtered_df,
    x='date',
    y=['value','smoothed_value'],
    labels={'date':'연도','value':'해수면 높이 (mm)'},
    title="해수면 변화 추이",
    template="plotly_white"
)
fig1.update_traces(patch={"name":"월별 데이터"}, selector={"name":"value"})
fig1.update_traces(patch={"name":f"{smoothing_window}개월 이동평균"}, selector={"name":"smoothed_value"})
if font_name:
    fig1.update_layout(font=dict(family=font_name))
st.plotly_chart(fig1, use_container_width=True)

st.download_button(
    label="📈 전처리된 해수면 데이터 다운로드 (CSV)",
    data=convert_df_to_csv(filtered_df),
    file_name="processed_sea_level_data.csv",
    mime="text/csv"
)

st.markdown("<br>", unsafe_allow_html=True)

# --- 2. 사용자 입력 기반 ---
st.markdown("---")
st.header("2. 사용자 제공 정보 기반 분석")
st.markdown("뉴스 기사 및 연구 자료를 기반으로 구성된 요약 대시보드")

@st.cache_data
def create_user_data():
    korea_years = np.arange(datetime.now().year-35, datetime.now().year+1)
    korea_sea_level = 3.05*(korea_years-korea_years.min())
    korea_df = pd.DataFrame({
        'date': pd.to_datetime([f'{y}-01-01' for y in korea_years]),
        'value': korea_sea_level,
        'group': '한국 연안 (추정)'
    })

    fishery_changes = {
        '어종': ['명태','방어','꽃게','낙지'],
        '변화 경향': ['어획량 급감','어획량 증가','연평어장 어획량 최저','어획량 감소로 가격 상승'],
        '주요 원인': ['수온 상승','수온 상승','해저오염','기후변화']
    }
    fishery_df = pd.DataFrame(fishery_changes)
    return korea_df, fishery_df

korea_sea_level_df, fishery_df = create_user_data()

# 주요 지표
col1,col2,col3 = st.columns(3)
with col1:
    st.metric("[글로벌] 연평균 해수면 상승","약 4.8 mm","사이언스타임즈 인용")
with col2:
    st.metric("[한국] 35년간 해수면 상승","10.7 cm","브릿지경제신문 인용")
with col3:
    st.metric("[한반도] 바다 수온 상승 속도","전 지구 평균의 2배","경향신문 인용")

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📈 한국 해수면 상승 추이 (추정)","🐟 주요 어종 변화"])

with tab1:
    st.subheader("📈 한국 연안 해수면 상승 추이 (추정)")
    fig2 = px.area(
        korea_sea_level_df,
        x='date', y='value',
        labels={'date':'연도','value':'누적 상승 높이 (mm)'},
        title="과거 35년간 한국 연안 해수면 상승 추정치"
    )
    if font_name:
        fig2.update_layout(font=dict(family=font_name))
    st.plotly_chart(fig2,use_container_width=True)
    st.download_button(
        label="📈 추정 데이터 다운로드 (CSV)",
        data=convert_df_to_csv(korea_sea_level_df),
        file_name="korea_estimated_sea_level.csv",
        mime="text/csv"
    )

with tab2:
    st.subheader("🐟 기후변화에 따른 주요 어종 변화")
    st.dataframe(fishery_df,use_container_width=True,hide_index=True)
    st.info("💡 인사이트: 수온 상승으로 한류성 어종 감소, 난류성 어종 북상 ('어종 아열대화')")
    st.download_button(
        label="🐟 어종 변화 데이터 다운로드 (CSV)",
        data=convert_df_to_csv(fishery_df),
        file_name="fishery_changes_summary.csv",
        mime="text/csv"
    )
