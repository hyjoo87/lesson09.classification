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
