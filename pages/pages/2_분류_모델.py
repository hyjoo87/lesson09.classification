import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# 1. 페이지 설정
st.set_page_config(
    page_title="뇌졸중 분류 모델",
    page_icon="🤖",
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
st.title("🤖 뇌졸중 분류 모델 학습 및 평가")
st.markdown("입력 속성을 선택하고 로지스틱 회귀와 의사결정트리 모델을 학습하여 성능을 비교해 봅니다.")
st.markdown("---")

# 속성 이름 매핑 사전 (우리말 이름 연결)
feature_map = {
    'age': '나이',
    'avg_glucose_level': '평균 혈당',
    'bmi': '체질량지수',
    'hypertension': '고혈압',
    'heart_disease': '심장병'
}
reverse_feature_map = {v: k for k, v in feature_map.items()}

# 4. 입력 속성 선택
st.subheader("🛠️ 모델 입력 속성 선택")
all_features_kor = list(feature_map.values())
default_features_kor = ['나이', '평균 혈당', '고혈압', '심장병'] # 체질량지수 뺀 넷 기본 선택

selected_kor = st.multiselect(
    "사용할 속성을 선택하세요 (2개 이상 선택)",
    options=all_features_kor,
    default=default_features_kor
)

# 두 개보다 적게 고르면 안내문 표시 후 멈춤
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

# 6. 모델 학습
# 6-1. 로지스틱 회귀 모델 (확률로 답하는 모델)
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

lr_train_acc = lr_model.score(X_train, y_train)
lr_test_acc = lr_model.score(X_test, y_test)

# 6-2. 의사결정트리 모델 (질문 3번, 리프 마디 최소 5명 미만 분기 금지, 난수 고정)
dt_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)
dt_model.fit(X_train, y_train)

dt_train_acc = dt_model.score(X_train, y_train)
dt_test_acc = dt_model.score(X_test, y_test)

# 6-3. 다수결 기준 모델 (입력을 전혀 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)
majority_class = y_train.mode()[0]
baseline_train_acc = (y_train == majority_class).mean()
baseline_test_acc = (y_test == majority_class).mean()

# 7. 정확도 큰 숫자 카드 표시
st.markdown("---")
st.subheader("📊 모델별 정확도 비교")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="로지스틱 회귀 (확률로 답하는 모델)",
        value=f"{lr_test_acc*100:.2f}%",
        delta=f"테스트 데이터: {len(y_test):,}명"
    )
    st.markdown(f"<div style='font-size: 0.9rem; color: gray;'>훈련 정확도: {lr_train_acc*100:.2f}% &nbsp;|&nbsp; 테스트 정확도: {lr_test_acc*100:.2f}%</div>", unsafe_allow_html=True)

with col2:
    st.metric(
        label="의사결정트리 (질문으로 답하는 모델)",
        value=f"{dt_test_acc*100:.2f}%",
        delta=f"테스트 데이터: {len(y_test):,}명"
    )
    st.markdown(f"<div style='font-size: 0.9rem; color: gray;'>훈련 정확도: {dt_train_acc*100:.2f}% &nbsp;|&nbsp; 테스트 정확도: {dt_test_acc*100:.2f}%</div>", unsafe_allow_html=True)

with col3:
    st.metric(
        label="다수결 기준 모델 (많은 쪽으로만 답함)",
        value=f"{baseline_test_acc*100:.2f}%",
        delta=f"다수 클래스: {'뇌졸중' if majority_class==1 else '아님'}"
    )
    st.markdown(f"<div style='font-size: 0.9rem; color: gray;'>훈련 정확도: {baseline_train_acc*100:.2f}% &nbsp;|&nbsp; 테스트 정확도: {baseline_test_acc*100:.2f}%</div>", unsafe_allow_html=True)

st.markdown("---")

# 8. 산점도 및 결정 경계 시각화
st.subheader("📈 2차원 산점도 및 결정 경계 시각화")
st.write("선택한 속성 중 가로축과 세로축으로 사용할 두 가지 속성을 골라주세요.")

viz_col1, viz_col2 = st.columns(2)
with viz_col1:
    x_axis_kor = st.selectbox("가로축 속성", options=selected_kor, index=0)
with viz_col2:
    default_y_idx = 1 if len(selected_kor) > 1 else 0
    y_axis_kor = st.selectbox("세로축 속성", options=selected_kor, index=default_y_idx)

x_feat = reverse_feature_map[x_axis_kor]
y_feat = reverse_feature_map[y_axis_kor]

# 두 축이 아닌 나머지 속성들은 테스트 데이터의 중앙값으로 고정
fixed_values = {}
fixed_notes = []
for feat in selected_features:
    if feat != x_feat and feat != y_feat:
        med_val = test_df[feat].median()
        fixed_values[feat] = med_val
        fixed_notes.append(f"• **{feature_map[feat]}**: 테스트 데이터 중앙값({med_val:.2f})으로 고정됨")

st.info("📌 **나머지 속성 고정 값 안내**\n\n" + ("\n".join(fixed_notes) if fixed_notes else "선택된 속성이 2개뿐이므로 고정된 다른 속성은 없습니다."))

# Plotly Scatter Plot 생성
test_df_plot = test_df.copy()
test_df_plot['stroke_label'] = test_df_plot['stroke'].map({0: '아님 (0)', 1: '뇌졸중 (1)'})

fig = px.scatter(
    test_df_plot,
    x=x_feat,
    y=y_feat,
    color='stroke_label',
    color_discrete_map={'아님 (0)': '#1f77b4', '뇌졸중 (1)': '#d62728'},
    title=f"{x_axis_kor} vs {y_axis_kor} 테스트 데이터 산점도",
    labels={x_feat: x_axis_kor, y_feat: y_axis_kor, 'stroke_label': '실제 뇌졸중 여부'}
)

# 결정 경계 및 영역 계산용 그리드 생성
x_min, x_max = test_df[x_feat].min(), test_df[x_feat].max()
y_min, y_max = test_df[y_feat].min(), test_df[y_feat].max()
x_range = np.linspace(x_min - (x_max - x_min)*0.1, x_max + (x_max - x_min)*0.1, 150)
y_range = np.linspace(y_min - (y_max - y_min)*0.1, y_max + (y_max - y_min)*0.1, 150)
xx, yy = np.meshgrid(x_range, y_range)

grid_dict = {}
for feat in selected_features:
    if feat == x_feat:
        grid_dict[feat] = xx.ravel()
    elif feat == y_feat:
        grid_dict[feat] = yy.ravel()
    else:
        grid_dict[feat] = np.full(xx.ravel().shape, fixed_values[feat])

grid_df = pd.DataFrame(grid_dict)

# 1) 로지스틱 회귀 0.5 확률선 (contour)
lr_probs = lr_model.predict_proba(grid_df)[:, 1].reshape(xx.shape)
fig.add_trace(
    go.Contour(
        z=lr_probs,
        x=x_range,
        y=y_range,
        showscale=False,
        contours=dict(start=0.5, end=0.5, size=0, coloring='lines'),
        line=dict(color='green', width=3),
        name='로지스틱 경계 (0.5)'
    )
)

# 0.5 확률선이 현재 그림 범위 밖인지 확인
lr_pred_grid = lr_model.predict(grid_df)
if len(np.unique(lr_pred_grid)) == 1:
    st.warning("⚠️ 로지스틱 회귀 모델의 0.5 기준 경계선이 현재 그려진 산점도 범위 밖(Screen Bounds)에 위치해 있습니다.")

# 2) 의사결정트리 영역 색칠 (heatmap)
dt_preds = dt_model.predict(grid_df).reshape(xx.shape)
fig.add_trace(
    go.Heatmap(
        z=dt_preds,
        x=x_range,
        y=y_range,
        colorscale=[[0, 'rgba(31, 119, 180, 0.08)'], [1, 'rgba(214, 39, 40, 0.08)']],
        showscale=False,
        hoverinfo='skip',
        name='의사결정트리 분할 영역'
    )
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 9. 의사결정트리 가지 구조 시각화 (Graphviz 활용)
st.subheader("🌳 의사결정트리 질문 가지 구조")
st.write("의사결정트리가 데이터를 분류하기 위해 던진 질문과 마디별 샘플 수, 뇌졸중 환자 수 및 비율을 보여줍니다.")

# 훈련용 데이터를 기준으로 각 마디의 샘플 수와 뇌졸중(1) 개수 커스텀 레이블 생성을 위해 Graphviz 내보내기 활용
feature_names_kor = [feature_map[f] for f in selected_features]

dot_data = export_graphviz(
    dt_model,
    out_file=None,
    feature_names=feature_names_kor,
    class_names=['아님', '뇌졸중'],
    filled=True,
    rounded=True,
    special_characters=True,
    precision=2
)

st.graphviz_chart(dot_data)

# 트리 구조 통계 분석 및 요약
n_leaves = np.sum(dt_model.tree_.children_left == -1)
leaf_values = dt_model.tree_.value[dt_model.tree_.children_left == -1]
leaf_classes = [np.argmax(val) for val in leaf_values]
n_leaf_no_stroke = sum([1 for c in leaf_classes if c == 0])

node_feature_indices = dt_model.tree_.feature
used_feature_indices = set([f for f in node_feature_indices if f != -2])
asked_features_kor = [feature_names_kor[i] for i in used_feature_indices]

st.markdown("##### 📌 의사결정트리 분석 요약")
st.markdown(f"- **답을 내는 마디(리프 노드)의 총 개수**: {n_leaves}개")
st.markdown(f"- **그중 '아님'이라고 답하는 칸의 개수**: {n_leaf_no_stroke}개")
st.markdown(f"- **이 나무가 실제로 질문에 사용한 속성**: {', '.join(asked_features_kor) if asked_features_kor else '없음'}")
