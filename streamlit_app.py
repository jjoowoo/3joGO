# streamlit_app.py
"""
Streamlit 대시보드 (한국어 UI)
- 공개 데이터 대시보드: NASA/NOAA 공식 해수면 자료 시계열 (자동 다운로드, 실패 시 예시 데이터로 대체)
    출처(예시):
    - NASA Sea Level Change Portal: https://sealevel.nasa.gov/  (Global Mean Sea Level)
    - NOAA STAR Sea Level Timeseries: https://www.star.nesdis.noaa.gov/socd/lsa/SeaLevelRise/LSA_SLR_timeseries.php
    - NOAA Sea Level Rise Viewer data: https://coast.noaa.gov/slrdata/
    - NOAA PSL monthly sea level timeseries: https://psl.noaa.gov/data/timeseries/month/SEALEVEL/
- 사용자 입력 대시보드: 프롬프트에 포함된 "입력(Input) 섹션"의 텍스트/데이터 진술만을 사용해
  내부적으로 생성한 예시(재구성) 데이터로 대시보드를 구성합니다.
  (입력 섹션에서 추출한 핵심 수치: "최근 10년", "사이언스타임즈: 연간 4.8mm 상승",
   "한국 연안 30년간 평균 10cm 상승" 등)
주의:
- 앱 실행 중 파일 업로드나 텍스트 입력을 요구하지 않습니다.
- 모든 라벨·버튼·툴팁은 한국어로 작성되었습니다.
- 폰트: /fonts/Pretendard-Bold.ttf 를 적용 시도합니다(없으면 무시).
"""

import io
import os
from datetime import datetime, timezone, timedelta
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.font_manager as fm

# ---------------------------
# 설정: 한국어 폰트 시도 (Pretendard)
# ---------------------------
PRETENDARD_PATH = "/fonts/Pretendard-Bold.ttf"
def try_register_pretendard():
    try:
        if os.path.exists(PRETENDARD_PATH):
            fm.fontManager.addfont(PRETENDARD_PATH)
            # matplotlib global rc
            import matplotlib.pyplot as plt
            plt.rcParams['font.family'] = fm.FontProperties(fname=PRETENDARD_PATH).get_name()
    except Exception:
        pass

try_register_pretendard()

# ---------------------------
# 공통 유틸리티
# ---------------------------
LOCAL_TZ = timezone(timedelta(hours=9))  # Asia/Seoul
TODAY_LOCAL = datetime.now(LOCAL_TZ).date()

@st.cache_data(ttl=60*60)
def download_csv(url, timeout=15):
    """
    시도: 다운로드 → 실패 시 raise
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content

def remove_future_dates(df, date_col='date'):
    # date_col must be datetime
    df = df.copy()
    df = df[df[date_col].dt.date <= TODAY_LOCAL]
    return df

def standardize_df(df, date_col, value_col, group_col=None):
    df = df.copy()
    df = df.rename(columns={date_col: 'date', value_col: 'value'} \
                   if date_col!= 'date' or value_col!='value' else {})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # drop rows with NaT or NaN values
    df = df.dropna(subset=['date', 'value'])
    # remove future
    df = remove_future_dates(df, 'date')
    # dedupe
    df = df.drop_duplicates(subset=['date'] + ([group_col] if group_col else []))
    # ensure numeric
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])
    if group_col and group_col in df.columns:
        df = df.rename(columns={group_col: 'group'})
    return df.sort_values('date').reset_index(drop=True)

def df_to_csv_bytes(df):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

# ---------------------------
# 공개 데이터: 시도 순서 및 예시 데이터
# ---------------------------
st.set_page_config(page_title="해수면 상승 대시보드", layout="wide")

st.title("해수면 상승 대시보드 — 공식 데이터 + 사용자 입력 데이터")
st.markdown("한국어 UI | 공개 데이터(공식) → 사용자 입력(프롬프트 내용) 순서로 대시보드를 보여줍니다.")

tab1, tab2 = st.tabs(["공식 공개 데이터 (NASA/NOAA 등)", "사용자 입력 데이터(프롬프트에서 재구성)"])

# 공개 데이터 다운로드 시도: 우선 NASA GMSL (PO.DAAC/JPL) 또는 NOAA PSL monthly series
NASA_GMSL_CSV = "https://sealeveldata.nascom.nasa.gov/files/2019/jpl_global_mean_sea_level.csv"
# Note: 위 URL은 예시형식입니다. (실제 경로가 바뀔 수 있으므로 실패시 예시 데이터로 대체)
NOAA_PSL_MONTHLY = "https://psl.noaa.gov/data/timeseries/month/SEALEVEL/"

# 예시/대체 데이터 (작동 실패 시 사용)
EXAMPLE_PUBLIC_CSV = """date,value,source
1993-01-01,0.0,nasa_example
1994-01-01,2.2,nasa_example
1995-01-01,3.9,nasa_example
1996-01-01,6.1,nasa_example
1997-01-01,7.8,nasa_example
1998-01-01,9.6,nasa_example
1999-01-01,10.5,nasa_example
2000-01-01,11.8,nasa_example
2001-01-01,12.9,nasa_example
2002-01-01,14.0,nasa_example
2003-01-01,15.6,nasa_example
2004-01-01,16.8,nasa_example
2005-01-01,18.0,nasa_example
2006-01-01,19.6,nasa_example
2007-01-01,21.2,nasa_example
2008-01-01,22.8,nasa_example
2009-01-01,24.1,nasa_example
2010-01-01,25.7,nasa_example
2011-01-01,27.1,nasa_example
2012-01-01,28.6,nasa_example
2013-01-01,30.2,nasa_example
2014-01-01,31.7,nasa_example
2015-01-01,33.0,nasa_example
2016-01-01,34.5,nasa_example
2017-01-01,36.2,nasa_example
2018-01-01,38.0,nasa_example
2019-01-01,39.6,nasa_example
2020-01-01,41.4,nasa_example
2021-01-01,43.0,nasa_example
2022-01-01,44.8,nasa_example
2023-01-01,46.5,nasa_example
2024-01-01,48.2,nasa_example
2025-01-01,49.7,nasa_example
"""

@st.cache_data(ttl=60*30)
def fetch_official_data():
    # 시도 1: NASA GMSL CSV
    urls_to_try = [
        NASA_GMSL_CSV,
        # fallback: NOAA PSL page (attempt simple GET)
        NOAA_PSL_MONTHLY
    ]
    last_err = None
    for url in urls_to_try:
        try:
            content = download_csv(url, timeout=12)
            # try to parse CSV
            # If URL points to an HTML page (like NOAA page), this may fail; handle below
            txt = content.decode('utf-8', errors='ignore')
            # If content appears to be CSV (has commas and header), parse
            if ',' in txt and ('date' in txt.lower() or 'year' in txt.lower()):
                df = pd.read_csv(io.StringIO(txt))
                return df, url, None
            else:
                # Not CSV; raise to go to next
                raise ValueError("다운로드된 내용이 CSV 형식이 아님")
        except Exception as e:
            last_err = e
            continue
    # 모두 실패하면 예시 데이터로 대체 (앱 화면에 안내문구 표시)
    df = pd.read_csv(io.StringIO(EXAMPLE_PUBLIC_CSV))
    return df, "예시데이터(자동대체)", str(last_err)

# 공개 데이터 탭
with tab1:
    st.header("공식 공개 데이터 (자동 연결 시도)")
    st.write("데이터 소스: NASA / NOAA / 공공 기관 (자동 다운로드 시도). 실패 시 예시 데이터로 대체됩니다.")
    with st.spinner("공식 데이터 다운로드를 시도하는 중..."):
        public_df_raw, used_url, error_msg = fetch_official_data()

    if used_url == "예시데이터(자동대체)":
        st.warning("공식 데이터 다운로드 실패 → 예시 데이터로 자동 대체했습니다. (하단에 실패 원인 출력)")
        if error_msg:
            st.caption(f"다운로드 실패 원인: {error_msg}")
    else:
        st.success(f"데이터 로드 완료: {used_url}")

    # 시도: 표준화 (date, value, group optional)
    # heuristics: find date-like column and numeric column
    def auto_standardize_public(df):
        df = df.copy()
        col_lower = [c.lower() for c in df.columns]
        # date column detection
        date_col = None
        for c in df.columns:
            if 'date' in c.lower() or 'year' in c.lower() or 'time' in c.lower():
                date_col = c
                break
        if date_col is None:
            # try first column
            date_col = df.columns[0]
        # value column detection
        value_col = None
        for c in df.columns:
            if c==date_col:
                continue
            if 'sea' in c.lower() or 'level' in c.lower() or 'mm' in c.lower() or 'value' in c.lower() or 'gmsl' in c.lower() or 'trend' in c.lower():
                value_col = c
                break
        if value_col is None:
            # choose first numeric-like
            for c in df.columns:
                if c==date_col: continue
                try:
                    pd.to_numeric(df[c].dropna().iloc[:5])
                    value_col = c
                    break
                except Exception:
                    continue
        if value_col is None:
            # fallback: second column
            if len(df.columns) >= 2:
                value_col = df.columns[1]
            else:
                value_col = df.columns[0]
        return standardize_df(df, date_col=date_col, value_col=value_col)

    try:
        public_df = auto_standardize_public(public_df_raw)
    except Exception as e:
        st.error("공개 데이터 전처리 중 오류 발생 — 예시데이터 사용")
        public_df = pd.read_csv(io.StringIO(EXAMPLE_PUBLIC_CSV))
        public_df = standardize_df(public_df, 'date', 'value')

    st.subheader("데이터 미리보기 (전처리된 표)")
    st.dataframe(public_df.head(50))

    st.markdown("**시계열 시각화 (전체)**")
    if public_df.empty:
        st.info("데이터가 없습니다.")
    else:
        fig = px.line(public_df, x='date', y='value', title='공식 데이터 기반 해수면(시계열)', labels={'date':'날짜','value':'해수면 (단위: 원자료 기준)'})
        fig.update_layout(legend_title_text=None)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**최근 10년 추세**")
    if not public_df.empty:
        last10_start = (pd.to_datetime(TODAY_LOCAL) - pd.DateOffset(years=10)).date()
        df_last10 = public_df[public_df['date'].dt.date >= last10_start]
        if df_last10.empty:
            st.info("최근 10년 자료가 충분하지 않습니다. 전체 자료를 표시합니다.")
            df_last10 = public_df
        fig2 = px.area(df_last10, x='date', y='value', title='최근 10년 해수면 변화(면적)', labels={'date':'날짜','value':'해수면'})
        st.plotly_chart(fig2, use_container_width=True)

    # CSV 다운로드
    st.download_button(
        label="전처리된 공개 데이터 CSV 다운로드",
        data=df_to_csv_bytes(public_df),
        file_name="public_sea_level_processed.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.caption("출처(참고): NASA Sea Level Change Portal / NOAA STAR Sea Level Timeseries / NOAA Sea Level Rise Viewer. "
               "코드 내 주석에 원본 URL을 명시했습니다.")

# ---------------------------
# 사용자 입력 데이터 대시보드
# ---------------------------
with tab2:
    st.header("사용자 입력 데이터 대시보드 (프롬프트 기반 재구성)")
    st.write("주의: 이 탭의 데이터는 사용자가 제공한 '입력(Input) 섹션'의 텍스트/수치 진술을 바탕으로 재구성한 내부 데이터입니다. "
             "앱 실행 중 추가 입력을 요구하지 않습니다.")

    # ---- 사용자 입력(프롬프트) 기반 데이터 재구성 ----
    # 사용자가 준 입력 요약에서 착안: "최근 10년간 해수면 상승", "사이언스타임즈: 연간 4.8mm 상승",
    # "한국 연안 최근 30년 평균 10cm 상승", "국내 수산물 생산량 전년 대비 2.2% 감소" 등.
    # 여기서는 2016~2025 (10년) 연간 누적(mm) 예시 데이터를 생성.
    years = list(range(TODAY_LOCAL.year - 9, TODAY_LOCAL.year + 1))  # 10년: e.g., 2016-2025 if today 2025
    # 기본 가정: 연평균 상승 4.8 mm/yr (입력 텍스트 기반)
    annual_mm = 4.8
    # 누적을 계산 (기준연도 = 시작 연도 - 1 => 0부터 시작)
    start_year = years[0] - 1
    cumulative = []
    for i, y in enumerate(years):
        cum = annual_mm * (i+1)  # 누적 mm since start_year
        cumulative.append(cum)

    user_df = pd.DataFrame({
        'date': pd.to_datetime([f"{y}-01-01" for y in years]),
        'value_mm': cumulative,
        '메모': ['입력문서: 연평균 4.8mm/yr 가정'] * len(years)
    })

    # 추가: 급식 관련 영향 지표(간단 시뮬): 수산물 가격 인덱스(기준=100), 어획량 변화 %
    # - 가정: 해수면 상승과 기후영향으로 10년간 누적 어획량 -10% (연평균 약 -1.05%)
    price_index = []
    catch_pct = []
    for i in range(len(years)):
        # price increases proportional to cumulative mm (단순 모델): baseline 100, +0.2% per mm
        price = 100 * (1 + 0.002 * cumulative[i])
        price_index.append(price)
        catch = 100 * (1 - 0.0105 * (i+1))  # 감소
        catch_pct.append(catch)

    user_df['수산물_가격지수(기준100)'] = np.round(price_index, 2)
    user_df['어획량지수(기준100)'] = np.round(catch_pct, 2)

    # 표준화 규칙: date,value,group(optional)
    # For user dashboard, create a standard view for sea level mm -> rename to 'value'
    user_public_standard = user_df[['date', 'value_mm', '수산물_가격지수(기준100)', '어획량지수(기준100)', '메모']].rename(columns={'value_mm':'value'})

    st.subheader("재구성된 데이터 (프롬프트 기반)")
    st.dataframe(user_public_standard)

    # 자동 사이드바 옵션 구성 (기간 필터, 스무딩 옵션, 단위 선택)
    st.sidebar.header("사용자 대시보드 설정")
    # 기간 필터 (자동: full range)
    min_date = user_public_standard['date'].min().date()
    max_date = user_public_standard['date'].max().date()
    sel_period = st.sidebar.date_input("기간 필터 (시작, 종료)", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    smoothing = st.sidebar.slider("스무딩(이동평균 창, 년)", 1, 5, 1)
    unit_choice = st.sidebar.selectbox("해수면 단위", ("mm", "cm"), index=0)

    # apply period filter
    start_sel, end_sel = sel_period
    mask = (user_public_standard['date'].dt.date >= start_sel) & (user_public_standard['date'].dt.date <= end_sel)
    df_vis = user_public_standard.loc[mask].copy()

    # smoothing if window >1
    if smoothing > 1:
        df_vis = df_vis.sort_values('date')
        df_vis['value_smooth'] = df_vis['value'].rolling(window=smoothing, min_periods=1, center=False).mean()
        plot_y = 'value_smooth'
        ylabel = f"해수면 (스무딩={smoothing}년)"
    else:
        plot_y = 'value'
        ylabel = "해수면 (mm)"

    if unit_choice == 'cm':
        df_vis['plot_value'] = df_vis[plot_y] / 10.0
        yaxis_label = "해수면 (cm)"
    else:
        df_vis['plot_value'] = df_vis[plot_y]
        yaxis_label = "해수면 (mm)"

    st.markdown("### 해수면 추이 (프롬프트 기반 재구성 데이터)")
    fig_user = px.line(df_vis, x='date', y='plot_value', markers=True, title='재구성 해수면 추이', labels={'date':'연도','plot_value':yaxis_label})
    st.plotly_chart(fig_user, use_container_width=True)

    st.markdown("### 수산물 가격 지수 / 어획량 지수 변화")
    df_vis2 = df_vis.copy()
    fig_multi = px.line(df_vis2, x='date', y=['수산물_가격지수(기준100)','어획량지수(기준100)'],
                        title='수산물 가격지수(우) 및 어획량지수(좌) 추이', labels={'value':'지수','date':'연도'})
    st.plotly_chart(fig_multi, use_container_width=True)

    st.markdown("### 요약 문장 (프롬프트 기반 해석)")
    st.write("- 입력 텍스트를 바탕으로 연평균 해수면 상승을 **4.8 mm/yr** 로 가정하여 최근 10년(연 단위) 누적 해수면 변화를 재구성했습니다.")
    st.write("- 단순 모형을 이용해 수산물 가격지수와 어획량지수를 생성했고, 이 지수는 급식·가정 식생활에 미치는 변화를 모사합니다.")
    st.write("- 실제 정책/영향 분석에는 지역별 정밀 데이터(연안별 조위·어획 통계 등)가 필요합니다.")

    # CSV 다운로드 (전처리된 사용자 데이터)
    st.download_button(
        label="전처리된 사용자 입력 데이터 CSV 다운로드",
        data=df_to_csv_bytes(df_vis2[['date','plot_value','수산물_가격지수(기준100)','어획량지수(기준100)','메모']].rename(columns={'plot_value':'sea_level_display'})),
        file_name="user_input_reconstructed.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.caption("참고: 사용자 데이터는 입력 프롬프트의 텍스트/수치를 바탕으로 재구성한 예시이며, 실제 관측값과 다를 수 있습니다.")

# ---------------------------
# 끝: 도움말 및 메타
# ---------------------------
st.sidebar.markdown("## 메타")
st.sidebar.markdown("- 개발자 설명: Streamlit + Codespaces에서 즉시 실행 가능하도록 설계되었습니다.")
st.sidebar.markdown("- 공개 데이터 원본(참고):")
st.sidebar.markdown("  - NASA Sea Level Change Portal: https://sealevel.nasa.gov/")
st.sidebar.markdown("  - NOAA STAR Sea Level Timeseries: https://www.star.nesdis.noaa.gov/socd/lsa/SeaLevelRise/LSA_SLR_timeseries.php")
st.sidebar.markdown("  - NOAA Sea Level Rise Viewer Data: https://coast.noaa.gov/slrdata/")
st.sidebar.markdown("- (참고) 만약 Kaggle API를 활용하려면, Codespaces에서 kaggle 패키지를 설치하고 ~/.kaggle/kaggle.json 인증 파일을 업로드해야 합니다.")
st.sidebar.markdown("  예: `pip install kaggle` → `export KAGGLE_CONFIG_DIR=~/.kaggle` → 업로드 후 `kaggle datasets download ...`")

