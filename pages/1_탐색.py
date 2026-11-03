import pandas as pd
import streamlit as st
import plotly.express as px

# 1. 페이지 설정
st.set_page_config(
    page_title="뇌졸중 데이터 탐색",
    page_icon="📊",
    layout="wide"
)

# 데이터 URL 정의
DATA_URL = "https://raw.githubusercontent.com/hyjoo87/data/refs/heads/main/stroke.csv"

# 2. 데이터 로드 함수 (캐싱 적용)
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

df = load_data()

# 타이틀
st.title("📊 뇌졸중 데이터 탐색 (Exploratory Data Analysis)")
st.markdown("데이터의 주요 특성을 시각화와 표를 통해 다각도로 살펴보는 공간입니다.")
st.markdown("---")

# 3. 나이와 평균 혈당의 분포 (히스토그램 두 개 나란히)
st.subheader("1. 나이(Age) 및 평균 혈당(Avg Glucose Level) 분포")
col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(df, x="age", nbins=30, title="나이 분포", labels={"age": "나이", "count": "사람 수"})
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_glucose = px.histogram(df, x="avg_glucose_level", nbins=30, title="평균 혈당 분포", labels={"avg_glucose_level": "평균 혈당", "count": "사람 수"})
    st.plotly_chart(fig_glucose, use_container_width=True)

st.markdown("---")

# 4. 뇌졸중 여부에 따른 나이 및 평균 혈당 비교 (상자그림 및 평균값 표)
st.subheader("2. 뇌졸중 여부에 따른 나이 및 평균 혈당 비교")
# stroke 컬럼을 보기 쉽게 문자열로 변환 (0 -> 정상, 1 -> 뇌졸중)
df_plot = df.copy()
df_plot['stroke_label'] = df_plot['stroke'].map({0: '정상 (0)', 1: '뇌졸중 (1)'})

col3, col4 = st.columns(2)

with col3:
    fig_box_age = px.box(df_plot, x="stroke_label", y="age", color="stroke_label", 
                         title="뇌졸중 여부별 나이 비교", labels={"stroke_label": "뇌졸중 여부", "age": "나이"})
    st.plotly_chart(fig_box_age, use_container_width=True)

with col4:
    fig_box_glucose = px.box(df_plot, x="stroke_label", y="avg_glucose_level", color="stroke_label", 
                             title="뇌졸중 여부별 평균 혈당 비교", labels={"stroke_label": "뇌졸중 여부", "avg_glucose_level": "평균 혈당"})
    st.plotly_chart(fig_box_glucose, use_container_width=True)

# 두 그룹의 평균값 표 (안전한 문법 적용)
st.markdown("##### 📌 두 그룹의 평균값 비교 표")
mean_table = df.groupby('stroke')[['age', 'avg_glucose_level']].mean(numeric_only=True).reset_index()
mean_table['stroke'] = mean_table['stroke'].map({0: '정상 (0)', 1: '뇌졸중 (1)'})
mean_table.columns = ['뇌졸중 여부', '평균 나이', '평균 혈당']
st.dataframe(mean_table, use_container_width=True, hide_index=True)

st.markdown("---")

# 5. 고혈압 및 심장병 여부에 따른 뇌졸중 비율 막대그래프
st.subheader("3. 고혈압 및 심장병에 따른 뇌졸중 비율")
col5, col6 = st.columns(2)

with col5:
    # 고혈압 비율 계산 (안전한 문법 적용)
    hypertension_df = df.groupby('hypertension')['stroke'].mean(numeric_only=True).reset_index()
    hypertension_df['hypertension'] = hypertension_df['hypertension'].map({0: '고혈압 없음 (0)', 1: '고혈압 있음 (1)'})
    hypertension_df['stroke_rate'] = hypertension_df['stroke'] * 100
    
    fig_hyp = px.bar(hypertension_df, x='hypertension', y='stroke_rate', color='hypertension',
                     title="고혈압 여부에 따른 뇌졸중 비율", 
                     labels={'hypertension': '고혈압 여부', 'stroke_rate': '뇌졸중 비율 (%)'})
    st.plotly_chart(fig_hyp, use_container_width=True)

with col6:
    # 심장병 비율 계산 (안전한 문법 적용)
    heart_df = df.groupby('heart_disease')['stroke'].mean(numeric_only=True).reset_index()
    heart_df['heart_disease'] = heart_df['heart_disease'].map({0: '심장병 없음 (0)', 1: '심장병 있음 (1)'})
    heart_df['stroke_rate'] = heart_df['stroke'] * 100
    
    fig_heart = px.bar(heart_df, x='heart_disease', y='stroke_rate', color='heart_disease',
                       title="심장병 여부에 따른 뇌졸중 비율", 
                       labels={'heart_disease': '심장병 여부', 'stroke_rate': '뇌졸중 비율 (%)'})
    st.plotly_chart(fig_heart, use_container_width=True)

st.markdown("---")

# 6. 체질량지수(bmi) 결측치 그룹과 전체 뇌졸중 비율 비교
st.subheader("4. 체질량지수(bmi) 결측치 그룹 vs 전체 뇌졸중 비율")

total_stroke_rate = df['stroke'].mean() * 100
missing_bmi_df = df[df['bmi'].isnull()]
missing_stroke_rate = missing_bmi_df['stroke'].mean() * 100 if len(missing_bmi_df) > 0 else 0

comparison_data = pd.DataFrame({
    "구분": ["전체 데이터", "bmi 비어 있는 사람들"],
    "인원수": [len(df), len(missing_bmi_df)],
    "뇌졸중 비율 (%)": [round(total_stroke_rate, 2), round(missing_stroke_rate, 2)]
})

st.dataframe(comparison_data, use_container_width=True, hide_index=True)

st.markdown("---")

# 7. 흡연 상태 값마다 사람 수 세기 표 (최신 판다스 호환 안전 문법 적용)
st.subheader("5. 흡연 상태(Smoking Status)별 사람 수")
s_counts = df['smoking_status'].value_counts()
smoking_counts = pd.DataFrame({
    '흡연 상태 (smoking_status)': s_counts.index,
    '사람 수': s_counts.values
})
st.dataframe(smoking_counts, use_container_width=True, hide_index=True)
