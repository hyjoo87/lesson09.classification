import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# 1. 페이지 설정
st.set_page_config(
    page_title="뇌졸중 모델 채점표",
    page_icon="📋",
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
st.title("📋 뇌졸중 분류 모델 채점표 (Evaluation Metrics & Error Analysis)")
st.markdown("테스트 데이터를 바탕으로 혼동행렬과 평가지표(정확도, 재현율, 정밀도, F1)를 비교하고 오분류 사례를 분석합니다.")
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

# 4. 입력 속성 선택 (분류 모델 페이지와 동일한 기본 설정: bmi 제외 4개)
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

# 6. 모델 학습 및 예측
# 6-1. 로지스틱 회귀 (확률로 답하는 모델)
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# 6-2. 의사결정트리 (질문으로 답하는 모델)
dt_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# 6-3. 다수결 기준 모델 (훈련용에서 많은 쪽으로만 답하는 모델)
majority_class = y_train.mode()[0]
y_pred_baseline = pd.Series([majority_class] * len(y_test), index=y_test.index)

# 테스트 데이터 요약 정보 표시
total_test_count = len(y_test)
actual_stroke_count = int(y_test.sum())
st.info(f"📌 **테스트 데이터 현황**: 총 **{total_test_count:,}명** 중 실제 뇌졸중 환자는 **{actual_stroke_count:,}명**입니다.")

st.markdown("---")

# 7. 혼동행렬(Confusion Matrix) 2x2 표 표시 (행: 실제, 열: 예측)
st.subheader("📊 모델별 혼동행렬 (Confusion Matrix)")
st.write("행은 실제값(실제 아님 / 실제 뇌졸중), 열은 예측값(예측 아님 / 예측 뇌졸중)을 나타냅니다.")

def get_confusion_matrix_df(y_true, y_pred):
    # sklearn confusion_matrix 순서: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    cm_df = pd.DataFrame(
        {
            "예측: 아님 (0)": [f"{tn:,} (TN)", f"{fn:,} (FN)"],
            "예측: 뇌졸중 (1)": [f"{fp:,} (FP)", f"{tp:,} (TP)"],
            "합계": [f"{tn+fp:,}", f"{fn+tp:,}"]
        },
        index=["실제: 아님 (0)", "실제: 뇌졸중 (1)"]
    )
    # 합계 행 추가
    cm_df.loc["합계"] = [f"{tn+fn:,}", f"{fp+tp:,}", f"{total_test_count:,}"]
    return cm_df, (tn, fp, fn, tp)

col_cm1, col_cm2, col_cm3 = st.columns(3)

with col_cm1:
    st.markdown("##### 로지스틱 회귀 (확률로 답하는 모델)")
    cm_lr_df, _ = get_confusion_matrix_df(y_test, y_pred_lr)
    st.dataframe(cm_lr_df, use_container_width=True)

with col_cm2:
    st.markdown("##### 의사결정트리 (질문으로 답하는 모델)")
    cm_dt_df, _ = get_confusion_matrix_df(y_test, y_pred_dt)
    st.dataframe(cm_dt_df, use_container_width=True)

with col_cm3:
    st.markdown("##### 다수결 기준 모델")
    cm_base_df, _ = get_confusion_matrix_df(y_test, y_pred_baseline)
    st.dataframe(cm_base_df, use_container_width=True)

st.markdown("---")

# 8. 네 가지 지표(정확도, 재현율, 정밀도, F1 Score) 계산 및 비교 표
st.subheader("📈 평가지표 비교 표")

def calc_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = tn + fp + fn + tp
    
    accuracy = (tp + tn) / total
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        precision_str = f"{precision:.4f}"
        f1_str = f"{f1:.4f}"
    else:
        precision_str = "계산할 수 없음"
        f1_str = "계산할 수 없음"
        
    return {
        "정확도 (Accuracy)": f"{accuracy:.4f}",
        "재현율 (Recall)": f"{recall:.4f}",
        "정밀도 (Precision)": precision_str,
        "F1 Score": f1_str
    }

metrics_lr = calc_metrics(y_test, y_pred_lr)
metrics_dt = calc_metrics(y_test, y_pred_dt)
metrics_base = calc_metrics(y_test, y_pred_baseline)

metrics_comparison_df = pd.DataFrame(
    [metrics_lr, metrics_dt, metrics_base],
    index=[
        "로지스틱 회귀 (확률로 답하는 모델)", 
        "의사결정트리 (질문으로 답하는 모델)", 
        "훈련용에서 많은 쪽으로만 답하는 모델"
    ]
).T

st.dataframe(metrics_comparison_df, use_container_width=True)

st.markdown("---")

# 9. 오분류 환자 목록 분석 (놓친 환자 FN, 헛짚은 사람 FP)
st.subheader("🔍 모델별 오분류 사례 분석 (놓친 환자 및 헛짚은 사람)")

model_choice = st.selectbox(
    "분석할 모델을 선택하세요",
    options=["로지스틱 회귀 (확률로 답하는 모델)", "의사결정트리 (질문으로 답하는 모델)", "다수결 기준 모델"]
)

if model_choice.startswith("로지스틱"):
    current_pred = y_pred_lr
elif model_choice.startswith("의사결정"):
    current_pred = y_pred_dt
else:
    current_pred = y_pred_baseline

# 오분류 마스크 생성
test_eval_df = test_df.copy()
test_eval_df['pred'] = current_pred

# 놓친 환자 (실제 stroke=1인데 pred=0으로 예측한 경우 -> FN)
fn_df = test_eval_df[(test_eval_df['stroke'] == 1) & (test_eval_df['pred'] == 0)].copy()
# 헛짚은 사람 (실제 stroke=0인데 pred=1로 예측한 경우 -> FP)
fp_df = test_eval_df[(test_eval_df['stroke'] == 0) & (test_eval_df['pred'] == 1)].copy()

# 나이가 많은 순서로 정렬
fn_df = fn_df.sort_values(by='age', ascending=False)
fp_df = fp_df.sort_values(by='age', ascending=False)

fn_count = len(fn_df)
fp_count = len(fp_df)

col_err1, col_err2 = st.columns(2)

with col_err1:
    st.markdown(f"##### 🔴 놓친 환자 (실제 뇌졸중이나 아니라고 예측) [전체 {fn_count}명]")
    display_fn = fn_df.head(20).copy()
    display_fn['고혈압'] = display_fn['hypertension'].map({0: '없음', 1: '있음'})
    display_fn['심장병'] = display_fn['heart_disease'].map({0: '없음', 1: '있음'})
    display_fn_table = display_fn[['age', 'avg_glucose_level', '고혈압', '심장병']].reset_index(drop=True)
    display_fn_table.columns = ['나이', '평균 혈당', '고혈압 여부', '심장병 여부']
    st.dataframe(display_fn_table, use_container_width=True)

with col_err2:
    st.markdown(f"##### 🔵 헛짚은 사람 (실제 아니나 뇌졸중이라고 예측) [전체 {fp_count}명]")
    display_fp = fp_df.head(20).copy()
    display_fp['고혈압'] = display_fp['hypertension'].map({0: '없음', 1: '있음'})
    display_fp['심장병'] = display_fp['heart_disease'].map({0: '없음', 1: '있음'})
    display_fp_table = display_fp[['age', 'avg_glucose_level', '고혈압', '심장병']].reset_index(drop=True)
    display_fp_table.columns = ['나이', '평균 혈당', '고혈압 여부', '심장병 여부']
    st.dataframe(display_fp_table, use_container_width=True)

# 혼동행렬의 FN, FP 값과 일치하는지 확인용 문구 표시
st.markdown(f"💡 **검증 확인**: 위 목록의 인원 수(놓친 환자 **{fn_count}명**, 헛짚은 사람 **{fp_count}명**)는 해당 모델의 혼동행렬 내 **FN** 및 **FP** 칸의 숫자와 정확히 일치합니다.")
