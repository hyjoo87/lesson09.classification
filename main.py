import os
import json
import pandas as pd
import streamlit as st

# 1. 페이지 기본 설정 (브라우저 탭 제목 및 아이콘 설정)
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# 데이터 URL 정의
DATA_URL = "https://raw.githubusercontent.com/hyjoo87/data/refs/heads/main/stroke.csv"
SAVE_DIR = "내_기록"
SAVE_FILE = os.path.join(SAVE_DIR, "data_info_records.json")

# 2. 저장용 폴더 생성 및 안전장치 추가 (FileExistsError 방지)
def ensure_dir():
    if os.path.exists(SAVE_DIR):
        if not os.path.isdir(SAVE_DIR):
            # 만약 '내_기록'이 폴더가 아니라 파일로 존재한다면 삭제 후 폴더 생성
            os.remove(SAVE_DIR)
            os.makedirs(SAVE_DIR, exist_ok=True)
    else:
        os.makedirs(SAVE_DIR, exist_ok=True)

def load_user_records():
    ensure_dir()
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"korean_meanings": {}, "data_source": ""}

def save_user_records(meanings_dict, source_text):
    ensure_dir()
    data = {
        "korean_meanings": meanings_dict,
        "data_source": source_text
    }
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 사용자 기록 불러오기
saved_records = load_user_records()

# 3. 데이터 로드 함수 (캐싱 적용)
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

df = load_data()

# 4. 화면 타이틀
st.title("🧠 뇌졸중 예측 실습실")
st.markdown("### 데이터셋 소개 및 개요")
st.write("본 실습실은 뇌졸중 관련 데이터를 탐색하고 예측 모델을 구축하기 위한 학습 공간입니다.")
st.markdown("---")

# 5. 주요 지표 카드 (4개 대시보드 메트릭)
total_count = len(df)
col_count = len(df.columns)
stroke_count = int(df['stroke'].sum())
stroke_ratio = (stroke_count / total_count) * 100

m1, m2, m3, m4 = st.columns(4)
m1.metric(label="전체 사람 수", value=f"{total_count:,}명")
m2.metric(label="열 개수", value=f"{col_count}개")
m3.metric(label="뇌졸중 환자 수 (stroke=1)", value=f"{stroke_count:,}명")
m4.metric(label="뇌졸중 비율", value=f"{stroke_ratio:.2f}%")

st.markdown("---")

# 6. 열 정보 표 및 우리말 뜻 입력 섹션
st.subheader("📋 데이터 열(Column) 정보")
st.caption("우리말 뜻 칸은 교재를 참고하여 직접 입력할 수 있습니다. 작성 후 아래 '저장하기' 버튼을 누르면 자동 저장됩니다.")

# 열 정보 데이터프레임 구성
col_info_list = []
saved_meanings = saved_records.get("korean_meanings", {})

for col in df.columns:
    data_type = str(df[col].dtype)
    null_count = int(df[col].isnull().sum())
    korean_meaning = saved_meanings.get(col, "")
    
    col_info_list.append({
        "열 이름 (Column)": col,
        "우리말 뜻": korean_meaning,
        "값의 종류 (Type)": data_type,
        "빈 값 개수 (Null)": null_count
    })

col_info_df = pd.DataFrame(col_info_list)

# Data Editor로 편집 가능하게 제공
edited_df = st.data_editor(
    col_info_df,
    disabled=["열 이름 (Column)", "값의 종류 (Type)", "빈 값 개수 (Null)"],
    hide_index=True,
    use_container_width=True,
    key="col_info_editor"
)

st.markdown("---")

# 7. 데이터 미리보기 (상위 5줄)
st.subheader("🔍 데이터 미리보기 (첫 5줄)")
st.dataframe(df.head(5), use_container_width=True)

st.markdown("---")

# 8. 데이터 출처 메모 및 저장 버튼
st.subheader("📌 데이터 출처 및 메모")
data_source_input = st.text_area(
    "교재에 기재된 데이터 출처 및 관련 내용을 적어주세요.",
    value=saved_records.get("data_source", ""),
    height=120,
    placeholder="여기에 출처 정보를 입력하세요..."
)

# 저장 버튼 동작
if st.button("💾 작성 내용 저장하기", type="primary"):
    # 에디터에서 편집된 우리말 뜻 딕셔너리로 추출
    updated_meanings = dict(zip(edited_df["열 이름 (Column)"], edited_df["우리말 뜻"]))
    save_user_records(updated_meanings, data_source_input)
    st.success("내용이 '내_기록/data_info_records.json'에 성공적으로 저장되었습니다!")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text

# 1. 페이지 설정
st.set_page_config(
    page_title="뇌졸중 분류 모델 실습실",
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
st.title("🤖 뇌졸중 분류 모델 학습 및 평가 실습실")
st.markdown("입력 속성을 선택하고 로지스틱 회귀와 의사결정트리 모델을 학습하여 성능을 비교해 봅니다.")
st.markdown("---")

# 속성 매핑 사전 정의
feature_map = {
    'age': '나이',
    'avg_glucose_level': '평균 혈당',
    'bmi': '체질량지수',
    'hypertension': '고혈압',
    'heart_disease': '심장병'
}
reverse_feature_map = {v: k for k, v in feature_map.items()}

# 4. 입력 속성 선택 사이드바 / 상단 설정
st.subheader("🛠️ 모델 입력 속성 선택")
st.write("모델 학습에 사용할 속성을 선택해주세요 (기본값: 체질량지수 제외 4개)")

default_features_kor = ['나이', '평균 혈당', '고혈압', '심장병']
all_features_kor = list(feature_map.values())

selected_kor = st.multiselect(
    "사용할 속성을 선택하세요 (2개 이상)",
    options=all_features_kor,
    default=default_features_kor
)

# 2개 보다 적게 고르면 안내문 표시 후 멈춤
if len(selected_kor) < 2:
    st.warning("⚠️ 속성을 최소 2개 이상 선택해 주세요!")
    st.stop()

selected_features = [reverse_feature_map[k] for k in selected_kor]

# 5. 데이터 전처리 및 테스트/훈련 데이터 분할
# 번호(id) 순 정렬
df_sorted = df_raw.sort_values('id').reset_index(drop=True)

test_indices = []
train_indices = []

# 열 명씩 묶어서 앞 3명은 테스트, 나머지 7명은 훈련
for i in range(0, len(df_sorted), 10):
    chunk = df_sorted.iloc[i:i+10]
    if len(chunk) >= 3:
        test_indices.extend(chunk.index[:3].tolist())
        train_indices.extend(chunk.index[3:].tolist())
    else:
        test_indices.extend(chunk.index.tolist())

train_df = df_sorted.iloc[train_indices].copy()
test_df = df_sorted.iloc[test_indices].copy()

# 체질량지수(bmi) 결측치 처리: 훈련용의 중앙값으로 채우기
bmi_median = train_df['bmi'].median()
train_df['bmi'] = train_df['bmi'].fillna(bmi_median)
test_df['bmi'] = test_df['bmi'].fillna(bmi_median)

# X, y 분할
X_train = train_df[selected_features]
y_train = train_df['stroke']
X_test = test_df[selected_features]
y_test = test_df['stroke']

# 6. 모델 학습
# 6-1. 로지스틱 회귀 모델
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

lr_train_acc = lr_model.score(X_train, y_train)
lr_test_acc = lr_model.score(X_test, y_test)

# 6-2. 의사결정트리 모델 (질문 3번, 최소 리프 노드 5명 미만 분기 금지, 난수 고정)
dt_model = DecisionTreeClassifier(
    max_depth=3, 
    min_samples_leaf=5, 
    random_state=42
)
dt_model.fit(X_train, y_train)

dt_train_acc = dt_model.score(X_train, y_train)
dt_test_acc = dt_model.score(X_test, y_test)

# 6-3. 다수결 기준 모델 (훈련용에서 많은 쪽으로만 답하는 모델)
majority_class = y_train.mode()[0]
baseline_train_acc = (y_train == majority_class).mean()
baseline_test_acc = (y_test == majority_class).mean()

# 7. 정확도 카드 표시
st.markdown("---")
st.subheader("📊 모델별 정확도 비교")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="로지스틱 회귀 (확률로 답하는 모델)",
        value=f"{lr_test_acc*100:.2f}%",
        delta=f"테스트 인원: {len(y_test):,}명"
    )
    st.caption(f"• 훈련 정확도: {lr_train_acc*100:.2f}%<br>• 테스트 정확도: {lr_test_acc*100:.2f}%", unsafe_allow_html=True)

with col2:
    st.metric(
        label="의사결정트리 (질문으로 답하는 모델)",
        value=f"{dt_test_acc*100:.2f}%",
        delta=f"테스트 인원: {len(y_test):,}명"
    )
    st.caption(f"• 훈련 정확도: {dt_train_acc*100:.2f}%<br>• 테스트 정확도: {dt_test_acc*100:.2f}%", unsafe_allow_html=True)

with col3:
    st.metric(
        label="다수결 기준 모델 (많은 쪽으로만 답함)",
        value=f"{baseline_test_acc*100:.2f}%",
        delta=f"다수 클래스: {'뇌졸중' if majority_class==1 else '아님'}"
    )
    st.caption(f"• 훈련 정확도: {baseline_train_acc*100:.2f}%<br>• 테스트 정확도: {baseline_test_acc*100:.2f}%", unsafe_allow_html=True)

st.markdown("---")

# 8. 산점도 및 결정 경계 시각화
st.subheader("📈 2차원 결정 경계 및 테스트 데이터 시각화")
st.write("선택한 속성 중 시각화에 사용할 2개의 축을 골라주세요.")

viz_col1, viz_col2 = st.columns(2)
with viz_col1:
    x_axis_kor = st.selectbox("가로축 속성", options=selected_kor, index=0)
with viz_col2:
    # 두 번째 기본값 설정 (선택된 항목이 2개 이상이면 두 번째 것)
    default_y_idx = 1 if len(selected_kor) > 1 else 0
    y_axis_kor = st.selectbox("세로축 속성", options=selected_kor, index=default_y_idx)

x_feat = reverse_feature_map[x_axis_kor]
y_feat = reverse_feature_map[y_axis_kor]

# 나머지 속성들은 테스트 데이터의 중앙값으로 고정
fixed_values = {}
fixed_notes = []
for feat in selected_features:
    if feat != x_feat and feat != y_feat:
        med_val = test_df[feat].median()
        fixed_values[feat] = med_val
        fixed_notes.append(f"• **{feature_map[feat]}**: 테스트 데이터 중앙값({med_val:.2f})으로 고정")

st.info("📌 **나머지 속성 고정 값 안내**\n\n" + "\n".join(fixed_notes))

# Plotly Scatter Plot 생성
test_df_plot = test_df.copy()
test_df_plot['stroke_label'] = test_df_plot['stroke'].map({0: '아님 (0)', 1: '뇌졸중 (1)'})

fig = px.scatter(
    test_df_plot,
    x=x_feat,
    y=y_feat,
    color='stroke_label',
    color_discrete_map={'아님 (0)': '#1f77b4', '뇌졸중 (1)': '#d62728'},
    title=f"{x_axis_kor} vs {y_axis_kor} 테스트 데이터 산점도 및 결정 경계",
    labels={x_feat: x_axis_kor, y_feat: y_axis_kor, 'stroke_label': '실제 뇌졸중 여부'}
)

# 결정 경계(Decision Boundary) 및 의사결정트리 영역 그리기 위한 격자 생성
x_min, x_max = test_df[x_feat].min(), test_df[x_feat].max()
y_min, y_max = test_df[y_feat].min(), test_df[y_feat].max()
x_range = np.linspace(x_min - (x_max-x_min)*0.1, x_max + (x_max-x_min)*0.1, 150)
y_range = np.linspace(y_min - (y_max-y_min)*0.1, y_max + (y_max-y_min)*0.1, 150)
xx, yy = np.meshgrid(x_range, y_range)

# 그리드 포인트 구성 (나머지 속성은 중앙값 채우기)
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

# 경계가 그림 밖인지 확인
lr_pred_grid = lr_model.predict(grid_df)
if len(np.unique(lr_pred_grid)) == 1:
    st.warning("⚠️ 로지스틱 회귀 모델의 0.5 기준 경계선이 현재 그려진 그림 범위 밖에 존재합니다.")

# 2) 의사결정트리 영역 색칠 (contour / heatmap)
dt_preds = dt_model.predict(grid_df).reshape(xx.shape)
fig.add_trace(
    go.Heatmap(
        z=dt_preds,
        x=x_range,
        y=y_range,
        colorscale=[[0, 'rgba(31, 119, 180, 0.1)'], [1, 'rgba(214, 39, 40, 0.1)]'],
        showscale=False,
        hoverinfo='skip'
    )
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 9. 의사결정트리 규칙 시각화 (Graphviz DOT 문자열 활용)
st.subheader("🌳 의사결정트리 구조 및 질문 규칙")

# scikit-learn tree structure를 활용해 DOT 포맷 생성 및 우리말 속성명으로 변환
from sklearn.tree import export_graphviz

# 피처 이름을 우리말 이름으로 교체
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

# 그래프비즈 스타일 조정 (답을 내는 마디 색상 다르게 등)
st.graphviz_chart(dot_data)

# 트리 해석 요약 통계 계산
n_nodes = dt_model.tree_.node_count
is_leaf = dt_model.tree_.children_left == -1
n_leaves = np.sum(is_leaf)
leaf_values = dt_model.tree_.value[is_leaf]
leaf_classes = [np.argmax(val) for val in leaf_values]
n_leaf_no_stroke = sum([1 for c in leaf_classes if c == 0])

# 트리가 실제로 물은 속성들 파악
node_feature_indices = dt_model.tree_.feature
used_feature_indices = set([f for f in node_feature_indices if f != -2])
asked_features_kor = [feature_names_kor[i] for i in used_feature_indices]

st.markdown("##### 📌 의사결정트리 분석 요약")
st.markdown(f"- **답을 내는 마디(리프 노드)의 총 개수**: {n_leaves}개")
st.markdown(f"- **그중 '아님'이라고 답하는 칸의 개수**: {n_leaf_no_stroke}개")
st.markdown(f"- **이 나무가 실제로 질문에 사용한 속성**: {', '.join(asked_features_kor) if asked_features_kor else '없음'}")
