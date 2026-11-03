import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# 1. 페이지 설정
st.set_page_config(
    page_title="뇌졸중 모델 성능 높이기",
    page_icon="⚡",
    layout="wide"
)

# 데이터 URL 정의
DATA_URL = "https://raw.githubusercontent.com/hyjoo87/data/refs/heads/main/stroke.csv"

# 2. 데이터 로드 함수 (캐싱 적용)
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

df_raw = load_data()

# 3. 타이틀
st.title("⚡ 뇌졸중 모델 성능 높이기 (클래스 불균형 가중치 조정)")
st.markdown("뇌졸중 데이터의 클래스 불균형 문제를 해결하기 위해 가중치를 맞추기(class_weight='balanced') 전과 후의 모델 성능을 비교합니다.")
st.markdown("---")

# 속성 이름 매핑 사전
feature_map = {
    'age': '나이',
    'avg_glucose_level': '평균 혈당',
    'bmi': '체질량지수',
    'hypertension': '고혈압',
    'heart_disease': '심장병'
}
reverse_feature_map = {v: k for k, v in feature_map.items()}

# 4. 입력 속성 선택 (기본값: 나이, 평균 혈당, 고혈압, 심장병)
st.subheader("🛠️ 모델 입력 속성 선택")
all_features_kor = list(feature_map.values())
default_features_kor = ['나이', '평균 혈당', '고혈압', '심장병']

selected_kor = st.multiselect(
    "사용할 속성을 선택하세요 (2개 이상 선택)",
    options=all_features_kor,
    default=default_features_kor
)

if len(selected_kor) < 2:
    st.warning("⚠️ 속성을 최소 2개 이상 선택해 주세요!")
    st.stop()

selected_features = [reverse_feature_map[k] for k in selected_kor]

# 5. 데이터 분할 및 전처리 (번호 순 정렬 후 10명씩 묶어 앞 3명 테스트, 나머지 7명 훈련)
df_sorted = df_raw.sort_values('id').reset_index(drop=True)

test_indices = []
train_indices = []

for i in range(0, len(df_sorted), 10):
    chunk = df_sorted.iloc[i:i+10]
    if len(chunk) >= 3:
        test_indices.extend(chunk.index[:3].tolist())
        train_indices.extend(chunk.index[3:].tolist())
    else:
        test_indices.extend(chunk.index.tolist())

train_df = df_sorted.iloc[train_indices].copy()
test_df = df_sorted.iloc[test_indices].copy()

# 체질량지수(bmi)가 선택되었고 결측치가 있는 경우 훈련용의 중앙값으로 채우기
if 'bmi' in selected_features:
    bmi_train_median = train_df['bmi'].median()
    train_df['bmi'] = train_df['bmi'].fillna(bmi_train_median)
    test_df['bmi'] = test_df['bmi'].fillna(bmi_train_median)

X_train = train_df[selected_features]
y_train = train_df['stroke']
X_test = test_df[selected_features]
y_test = test_df['stroke']

# 6. 모델 학습 (가중치 전 / 후)
# 6-1. 로지스틱 회귀
lr_before = LogisticRegression(random_state=42)
lr_before.fit(X_train, y_train)

lr_after = LogisticRegression(class_weight='balanced', random_state=42)
lr_after.fit(X_train, y_train)

# 6-2. 의사결정트리
dt_before = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_before.fit(X_train, y_train)

dt_after = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, class_weight='balanced', random_state=42)
dt_after.fit(X_train, y_train)

y_pred_lr_b = lr_before.predict(X_test)
y_pred_lr_a = lr_after.predict(X_test)
y_pred_dt_b = dt_before.predict(X_test)
y_pred_dt_a = dt_after.predict(X_test)

# 지표 계산 함수
def get_metrics_and_counts(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    cm_df = pd.DataFrame(
        {
            "예측: 아님 (0)": [f"{tn:,} (TN)", f"{fn:,} (FN)"],
            "예측: 뇌졸중 (1)": [f"{fp:,} (FP)", f"{tp:,} (TP)"],
            "합계": [f"{tn+fp:,}", f"{fn+tp:,}"]
        },
        index=["실제: 아님 (0)", "실제: 뇌졸중 (1)"]
    )
    cm_df.loc["합계"] = [f"{tn+fn:,}", f"{fp+tp:,}", f"{total:,}"]
    
    metrics_series = pd.Series({
        "정확도 (Accuracy)": f"{accuracy:.4f}",
        "재현율 (Recall)": f"{recall:.4f}",
        "정밀도 (Precision)": f"{precision:.4f}" if (tp+fp)>0 else "계산 불가",
        "F1 Score": f"{f1:.4f}" if (tp+fp)>0 else "계산 불가"
    })
    return cm_df, metrics_series, (tp, fn, fp)

cm_lr_b, m_lr_b, counts_lr_b = get_metrics_and_counts(y_test, y_pred_lr_b)
cm_lr_a, m_lr_a, counts_lr_a = get_metrics_and_counts(y_test, y_pred_lr_a)
cm_dt_b, m_dt_b, counts_dt_b = get_metrics_and_counts(y_test, y_pred_dt_b)
cm_dt_a, m_dt_a, counts_dt_a = get_metrics_and_counts(y_test, y_pred_dt_a)

st.markdown("---")

# 7. 모델별 전·후 비교 표 표시
st.subheader("📊 가중치 조정 전·후 혼동행렬 및 평가지표 비교")

def display_model_comparison(model_title, cm_b, m_b, counts_b, cm_a, m_a, counts_a):
    st.markdown(f"#### 🤖 {model_title}")
    col_b, col_a = st.columns(2)
    
    with col_b:
        st.markdown("**[가중치 조정 전]**")
        st.dataframe(cm_b, use_container_width=True)
        st.dataframe(pd.DataFrame(m_b, columns=["지표"]), use_container_width=True)
    
    with col_a:
        st.markdown("**[가중치 조정 후 (balanced)]**")
        st.dataframe(cm_a, use_container_width=True)
        st.dataframe(pd.DataFrame(m_a, columns=["지표"]), use_container_width=True)
        
    tp_b, fn_b, fp_b = counts_b
    tp_a, fn_a, fp_a = counts_a
    st.info(f"📌 **변경 결과 요약**: 찾아낸 환자 수(TP)는 **{tp_b}명**에서 **{tp_a}명**으로, 놓친 환자 수(FN)는 **{fn_b}명**에서 **{fn_a}명**으로, 헛짚은 사람 수(FP)는 **{fp_b}명**에서 **{fp_a}명**으로 바뀌었습니다.")
    st.markdown("---")

display_model_comparison("로지스틱 회귀 (확률로 답하는 모델)", cm_lr_b, m_lr_b, counts_lr_b, cm_lr_a, m_lr_a, counts_lr_a)
display_model_comparison("의사결정트리 (질문으로 답하는 모델)", cm_dt_b, m_dt_b, counts_dt_b, cm_dt_a, m_dt_a, counts_dt_a)

# 8. 결정 경계 시각화 (2차원 또는 3차원)
st.subheader("📈 결정 경계 겹쳐 그리기 (가중치 조정 전 vs 후)")

if len(selected_kor) == 2:
    st.write("속성을 2개 골랐으므로 2차원 산점도 위에 전·후 0.5 경계선을 겹쳐서 그립니다.")
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        x_axis_kor = st.selectbox("가로축 속성", options=selected_kor, index=0)
    with viz_col2:
        y_axis_kor = st.selectbox("세로축 속성", options=selected_kor, index=1)
        
    x_feat = reverse_feature_map[x_axis_kor]
    y_feat = reverse_feature_map[y_axis_kor]
    
    test_df_plot = test_df.copy()
    test_df_plot['stroke_label'] = test_df_plot['stroke'].map({0: '아님 (0)', 1: '뇌졸중 (1)'})
    
    fig = px.scatter(
        test_df_plot, x=x_feat, y=y_feat, color='stroke_label',
        color_discrete_map={'아님 (0)': '#1f77b4', '뇌졸중 (1)': '#d62728'},
        title=f"로지스틱 회귀 가중치 전·후 0.5 경계선 비교 ({x_axis_kor} vs {y_axis_kor})",
        labels={x_feat: x_axis_kor, y_feat: y_axis_kor, 'stroke_label': '실제 뇌졸중 여부'}
    )
    
    x_min, x_max = test_df[x_feat].min(), test_df[x_feat].max()
    y_min, y_max = test_df[y_feat].min(), test_df[y_feat].max()
    x_range = np.linspace(x_min - (x_max - x_min)*0.1, x_max + (x_max - x_min)*0.1, 150)
    y_range = np.linspace(y_min - (y_max - y_min)*0.1, y_max + (y_max - y_min)*0.1, 150)
    xx, yy = np.meshgrid(x_range, y_range)
    
    grid_df = pd.DataFrame({x_feat: xx.ravel(), y_feat: yy.ravel()})
    
    # 전 경계선 (파란색 계열)
    probs_b = lr_before.predict_proba(grid_df)[:, 1].reshape(xx.shape)
    fig.add_trace(go.Contour(z=probs_b, x=x_range, y=y_range, showscale=False,
                             contours=dict(start=0.5, end=0.5, size=0, coloring='lines'),
                             line=dict(color='blue', width=3), name='조정 전 (0.5 경계)'))
    
    # 후 경계선 (주황색/빨간색 계열)
    probs_a = lr_after.predict_proba(grid_df)[:, 1].reshape(xx.shape)
    fig.add_trace(go.Contour(z=probs_a, x=x_range, y=y_range, showscale=False,
                             contours=dict(start=0.5, end=0.5, size=0, coloring='lines'),
                             line=dict(color='orange', width=3), name='조정 후 (0.5 경계)'))
    
    st.plotly_chart(fig, use_container_width=True)

else:
    st.write("속성을 3개 이상 골랐으므로 3차원 산점도 위에 전·후 0.5 평면을 겹쳐서 보여드립니다. 마우스로 회전해 볼 수 있습니다.")
    c1, c2, c3 = st.columns(3)
    with c1:
        x_axis_kor = st.selectbox("X축 속성", options=selected_kor, index=0)
    with c2:
        y_axis_kor = st.selectbox("Y축 속성", options=selected_kor, index=1)
    with c3:
        z_axis_kor = st.selectbox("Z축 속성", options=selected_kor, index=2)
        
    x_f, y_f, z_f = reverse_feature_map[x_axis_kor], reverse_feature_map[y_axis_kor], reverse_feature_map[z_axis_kor]
    
    # 나머지 속성은 중앙값 고정
    fixed_vals = {f: test_df[f].median() for f in selected_features if f not in [x_f, y_f, z_f]}
    
    test_df_plot = test_df.copy()
    test_df_plot['stroke_label'] = test_df_plot['stroke'].map({0: '아님 (0)', 1: '뇌졸중 (1)'})
    
    fig = px.scatter_3d(
        test_df_plot, x=x_f, y=y_f, z=z_f, color='stroke_label',
        color_discrete_map={'아님 (0)': '#1f77b4', '뇌졸중 (1)': '#d62728'},
        title=f"3차원 로지스틱 회귀 경계 평면 비교 ({x_axis_kor}, {y_axis_kor}, {z_axis_kor})"
    )
    
    # 그리드 생성
    xr = np.linspace(test_df[x_f].min(), test_df[x_f].max(), 20)
    yr = np.linspace(test_df[y_f].min(), test_df[y_f].max(), 20)
    xx, yy = np.meshgrid(xr, yr)
    
    # z 평면 계산을 위한 딕셔너리 구축
    grid_data = {}
    for feat in selected_features:
        if feat == x_f:
            grid_data[feat] = xx.ravel()
        elif feat == y_f:
            grid_data[feat] = yy.ravel()
        elif feat == z_f:
            # 로지스틱 회귀식 w_x*x + w_y*y + w_z*z + b = 0 (prob=0.5 -> logit=0)
            # z = (- w_x*x - w_y*y - sum(w_fixed * val) - intercept) / w_z 인자 추출
            pass
        else:
            grid_data[feat] = np.full(xx.ravel().shape, fixed_vals[feat])
            
    # 시각화용 3D surface 추가 (조정 전/후)
    # 3차원 가중치 평면 계산
    coefs_b = lr_before.coef_[0]
    intercept_b = lr_before.intercept_[0]
    feat_list = list(selected_features)
    
    ix, iy, iz = feat_list.index(x_f), feat_list.index(y_f), feat_list.index(z_f)
    wx, wy, wz = coefs_b[ix], coefs_b[iy], coefs_b[iz]
    
    fixed_sum_b = intercept_b
    for idx, f in enumerate(feat_list):
        if f not in [x_f, y_f, z_f]:
            fixed_sum_b += coefs_b[idx] * fixed_vals[f]
            
    if wz != 0:
        zz_b = (-wx * xx - wy * yy - fixed_sum_b) / wz
        fig.add_trace(go.Surface(x=xx, y=yy, z=zz_b, colorscale=[[0, 'blue'], [1, 'blue']], opacity=0.3, showscale=False, name='조정 전 평면'))
        
    coefs_a = lr_after.coef_[0]
    intercept_a = lr_after.intercept_[0]
    wxa, wya, wza = coefs_a[ix], coefs_a[iy], coefs_a[iz]
    fixed_sum_a = intercept_a
    for idx, f in enumerate(feat_list):
        if f not in [x_f, y_f, z_f]:
            fixed_sum_a += coefs_a[idx] * fixed_vals[f]
            
    if wza != 0:
        zz_a = (-wxa * xx - wya * yy - fixed_sum_a) / wza
        fig.add_trace(go.Surface(x=xx, y=yy, z=zz_a, colorscale=[[0, 'orange'], [1, 'orange']], opacity=0.3, showscale=False, name='조정 후 평면'))
        
    st.plotly_chart(fig, use_container_width=True)
