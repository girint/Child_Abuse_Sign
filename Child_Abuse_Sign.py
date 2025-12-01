import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
import os
import time
import re
from PIL import Image
from collections import deque





#----# YOLO 탐지 실행 및 결과 이미지 반환 함수-----------
def detect_image(image_rgb):
    results = model(image_rgb)
    result_bgr = results[0].plot()
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    return result_rgb




# ----------- 모델 파일명 입력 → 이미지 자동 선택 ----------
BASE_MODEL_DIR = r"C:\girin\VS_code_windows\project\model"
BASE_IMAGE_DIR = os.path.join(BASE_MODEL_DIR, 'image')
model_file = st.sidebar.text_input("모델 파일명 입력", value="best18.pt")
model_path = os.path.join(BASE_MODEL_DIR, model_file)

@st.cache_resource
def load_model(path):
    return YOLO(path)

try:
    model = load_model(model_path)
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.sidebar.error(f"모델 로드 실패: {e}")

page = st.sidebar.radio("메뉴 선택", ["메인","학습결과 대시보드", "객체 탐지"])


# ----------- 프로젝트 소개 -----------
if page == "메인":
    st.set_page_config(page_title="YOLOv8 객체탐지", layout="centered")
    st.title("🛡️ 아동학대 감지 모델 학습 🛡️")

    with st.expander("프로젝트 소개", expanded=True):
        st.markdown(
            """
            <div style="background-color:#f0f8ff; padding:15px; border-radius:10px;">
                AI 영상 분석으로 어린이집 내 아동과 교사 행동을 실시간 모니터링하여 
                비정상적 행동을 탐지하면 즉시 관리자에 알림하는 안전 솔루션입니다.<br>
                <b>주요 특징:</b> 실시간 탐지, 관리자 알림, 정확도 개선
            </div>
            """, unsafe_allow_html=True
        )

    with st.expander("진행과정", expanded=True):
        st.markdown(
            """
            1. **아동학대와 관련된 주요 객체(행동·상황)를 정의**  
            학대의 위험신호가 될 수 있는 특정 행동이나 패턴을 선정함

            2. **관련 데이터 이미지 수집 및 서치**  
            정의된 객체(행동 등)를 포함한 이미지·영상 데이터를 광범위하게 확보하고, 품질 검증

            3. **YOLOv8 알고리즘을 활용한 딥러닝 모델 학습**  
            수집한 이미지 데이터를 객체별로 라벨링하여 YOLO 모델로 학습, 검증 작업 수행

            4. **지속적 객체 반복 행동 탐지 및 경고 시스템 구현**  
            실시간/배치 탐지 결과에서 동일한 학대 연관 객체가 일정 임계치 이상 반복 출현하면 관리자 혹은 보호자에게 자동 알림 메시지 발송  
            이상 행동이 누적될 경우 아동학대 의심 신호로 적극 대응 지원
            """
        )

    with st.expander("라벨링 클래스", expanded=True):
        st.markdown(
            """
            <div style="background-color:#fcf8e3; padding:15px; border-radius:10px; color:#856404; font-weight:bold;">
                <b>학대와 관련있는 11개의 객체 지정</b>
                <ul style="list-style-type: square; padding-left: 20px;">
                    <li>0. 성인 : adult</li>
                    <li>1. 아이 : child</li>
                    <li>2. 손들기 : hand_up</li>
                    <li>3. 주먹 : fist</li>
                    <li>4. 울음 : cry</li>
                    <li>5. 발길질 : foot_up</li>
                    <li>6. 고함 : scream</li>
                    <li>7. 목조르기 : choke</li>
                    <li>8. 평범표정 : normal</li>
                    <li>9. 움츠림 : crouch</li>
                    <li>10. 손가락질 : finger</li>
                </ul>
            </div>
            """, unsafe_allow_html=True
        )




#---------- 학습결과 대시보드 ------------------
elif page == "학습결과 대시보드":
    if os.path.exists(BASE_IMAGE_DIR):
        image_files = [
            f for f in os.listdir(BASE_IMAGE_DIR)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]
    else:
        image_files = []

    st.markdown("## <span style='color:#0072C6;'>📊 <b>YOLOv8 학습 결과 대시보드</b></span>", unsafe_allow_html=True)
    st.markdown("---")

    # 모델 파일명에서 번호 추출하여 기본 이미지 자동 선택
    import re
    match = re.search(r'(\d+)', model_file)
    model_num = match.group(1) if match else ""
    default_image = next((f for f in image_files if model_num and model_num in f), image_files[0] if image_files else None)

    selected_image = st.selectbox(
        "이미지를 선택하세요:",
        image_files,
        index=image_files.index(default_image) if default_image in image_files else 0
    )

    # 선택이미지 한 장만 표시
    if selected_image:
        image_path = os.path.join(BASE_IMAGE_DIR, selected_image)
        st.image(image_path, caption=selected_image, use_container_width=True)
        st.markdown("⭐ ** 학습·검증 손실(Metric) Plot**")
    elif not image_files:
        st.warning(f"'{BASE_IMAGE_DIR}' 폴더가 없거나 이미지 파일이 없습니다.")




#--------------- 객체 탐지 부분--------------
else:
    if not model_loaded:
        st.error("모델을 정상적으로 로드하지 못했습니다.")
    else:
        # 공통 제목
        st.title("🛡️ YOLOv8 아동학대 감지 모델 🛡️")

        # 라디오 버튼 왼쪽 정렬
        st.markdown(
            """
            <style>
            div[role="radiogroup"] > label {
                display: block;
                text-align: left;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        mode = st.radio("탐지 모드 선택", ["이미지 업로드", "웹캠", "동영상 업로드"])

        def detect_image(image_rgb):
            results = model(image_rgb)
            result_bgr = results[0].plot()
            return result_bgr

        if mode == "이미지 업로드":
            uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image_bgr = cv2.imdecode(file_bytes, 1)

                st.subheader("탐지 결과")
                input_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                result_bgr = detect_image(input_rgb)
                result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, caption="탐지된 이미지", use_container_width=True)

        elif mode == "웹캠":
            OBJECT_LIST = [
                "adult", "child", "hand_up", "fist", "cry", "foot_up", "scream",
                "choke", "normal", "crouch", "finger"
            ]
            DETECT_RULES = {
                "crouch": {"window_sec": 20, "min_count": 20, "msg": "'움츠림' 20초 내 자주 감지 → 아동학대 의심"},
                "choke": {"window_sec": 5, "min_count": 5, "msg": "'목조르기' 5초 내 반복 감지 → 위험 경고"},
                "cry": {"window_sec": 10, "min_count": 10, "msg": "'울음' 10초 내 반복 감지 → 주의 필요"},
            }

            run = st.checkbox("웹캠 실시간 탐지 시작")
            stframe = st.empty()
            chart_area = st.empty()
            table_area = st.empty()
            object_logs = {obj: deque(maxlen=1000) for obj in OBJECT_LIST}

            alert_sidebar = st.sidebar.empty()  # 알림 오른쪽 사이드바 출력

            if run:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("웹캠을 열 수 없습니다.")
                else:
                    while run:
                        ret, frame_bgr = cap.read()
                        if not ret:
                            st.warning("웹캠 프레임을 가져올 수 없습니다.")
                            break

                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        results = model(frame_rgb)
                        now_sec = time.time()

                        for result in results:
                            classes = result.boxes.cls.cpu().numpy().astype(int)
                            for cls_id in classes:
                                cls_name = result.names[cls_id]
                                if cls_name in OBJECT_LIST:
                                    object_logs[cls_name].append(now_sec)

                        alert_list = []
                        for obj, rule in DETECT_RULES.items():
                            times = object_logs[obj]
                            cnt = sum([t > now_sec - rule['window_sec'] for t in times])
                            if cnt >= rule['min_count']:
                                alert_list.append(rule['msg'])

                        result_bgr = results[0].plot()
                        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                        stframe.image(result_rgb, channels="RGB", use_container_width=True)

                        # 알림 메시지를 사이드바에 표시
                        alert_sidebar.empty()
                        if alert_list:
                            for msg in alert_list:
                                alert_sidebar.warning(msg)

                        counts = {obj: sum([t > now_sec - 10 for t in object_logs[obj]]) for obj in OBJECT_LIST}

                        # 숫자가 그래프보다 먼저 보이도록 표 위에 표시
                        table_area.dataframe(
                            [{"객체명": obj, "최근 10초 감지수": counts[obj]} for obj in OBJECT_LIST]
                        )

                        # y축 최대값 고정 (예: 최대 40)
                        chart_area.bar_chart(counts, use_container_width=True, height=250)  # Streamlit 기본 bar_chart는 y축 고정 옵션 제한적이라 차트 내 데이터 범위 참고

                        time.sleep(0.5)
                    cap.release()

        else:  # 동영상 업로드
            uploaded_video = st.file_uploader("동영상을 업로드하세요", type=["mp4", "mov", "avi"])
            if uploaded_video:
                st.success("동영상 업로드 완료! 자동 객체탐지 진행 중입니다.")
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_video.read())
                cap = cv2.VideoCapture(tfile.name)

                frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_video = cap.get(cv2.CAP_PROP_FPS)
                duration = frame_total / fps_video if fps_video > 0 else 0

                st.write(f"총 프레임: {frame_total} | FPS: {fps_video:.2f} | 영상 길이: {duration:.1f}초")

                stframe = st.empty()
                frame_idx = 0
                start_time = time.time()

                target_fps = 5
                frame_interval = int(fps_video // target_fps) if fps_video > 0 else 6

                # 재생/정지 상태 토글
                play = st.button("재생 / 정지 ",key="play")
                paused = False

                while cap.isOpened():
                    if paused:
                        time.sleep(0.1)
                        continue

                    ret, frame_bgr = cap.read()
                    if not ret or frame_idx >= frame_total:
                        break
                    if frame_idx % frame_interval == 0:
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        result_bgr = detect_image(frame_rgb)
                        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                        stframe.image(result_rgb, channels="RGB", use_container_width=True)
                        time.sleep(1.0 / target_fps)
                    frame_idx += 1

                    # 토글 버튼 눌림 감지 (재생/정지 상태 토글)
                    if st.button("재생 / 정지 토글"):
                        paused = not paused

                cap.release()
                total_play_time = time.time() - start_time
                st.success(f"객체 탐지 완료! 실제 재생 시간: {total_play_time:.1f}초 (원본 길이: {duration:.1f}초)")

            else:
                st.info("동영상 업로드 후 자동으로 객체탐지를 시작합니다.")
