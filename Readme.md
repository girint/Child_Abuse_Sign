# 👨‍👩‍👦 어린이집 가정 내 아동학대 의심 아동 위험도 예측 및 신고 자동화 솔루션  
### yolo 기반 영상 분석 프로젝트

---

## 📌 INDEX
- [📍프로젝트 소개](#프로젝트-소개)
- [🌞프로젝트 구조](#프로젝트-구조)
- [🌝모델 개발 과정](#모델-개발-과정)
- [⭐향후 개선 방향](#향후-발전-방향)
- [🌈팀원 마지막 한마디](#팀원-마지막-한마디)


---

## 📍프로젝트 소개

### 👨‍🏫 아동학대 위험도 예측 및 신고 자동화 서비스

- 어린이집 아이들의 행동 객체을 **객체 학습 모델**로 분석
- 반복도를 통한 아동학대 **위험도 예측** 및 자동 **알림 전송**
- 교사의 부담 감소 및 **조기 대응** 목표

### 🔍 주요 특징
- 실시간 위험 행동 감지
- 관리자 알림 기반 즉각 대응

---

## 🌞프로젝트 구조

```
아동학대감지프로젝트/
├── Model/
│ ├── best03.pt             #초기 학습 모델 파라미터
│ ├── best09.pt             #개별학습모델(손가락)
│ ├── best13.pt             #개별학습모델(손들기)
│ ├── best18.pt             #통합학습모델(best)
│ ├── best19.pt             #통합학습모델(vaild지정)
│ └── images/
│   ├── test03.png          #학습그래프
│   ├── test09.png
│   ├── test13.png
│   ├── test18.png
│   └── test19.png
│
│
├── Child_Abuse_Sign.py   #stramit화면구현
├── README.md             #readme
├── memo                  #모델별 소개
```

---

## 🌝모델 개발 과정

### Phase 1. 초기 시도

**🛑방식** 
<details><summary>16개 객체 동시 학습</summary>
  <br>
  1. 성인 : adult  <br>
  2. 아이 : child  <br>
  3. 손들기 : handup   <br>
  4. 주먹 : fist       <br>
  5. 무기 : weapon  <br>
  6. 엎드려뻗쳐 : updoun <br>
  7. 체벌 : scold <br>
  8. 목조르기 : choke   <br>
  9. 벽보기  : stand_side <br>
  10. 움츠림 : crouch   <br>
  11. 손가락질 : finger <br>
  12. 넘어짐 : falling down  <br>
  13. 울음 : cry  <br>
  14. 발길질 : foot_up <br>
  15. 고함 : scream  <br>
  16. 평범 : nomal 
</details>  


**🚨결과**  
<details><summary>결과그래프 </summary>
<img width="3018" height="1509" alt="image" src="https://github.com/user-attachments/assets/224850f4-1240-40ca-9ce9-57c1ff7dcb86" />
</details>

- mAP50: 20% 
- Precision: 30% 
- Recall: 50%

**🚥객체 탐지 실패 원인**
1. 클래스별 데이터 부족(50~200개 수준)
2. 라벨링 편차
3. 하이퍼파라미터 최적화 미흡


**🪛문제점 해결 전략**
1. 객체 수 축소 및 객체 명확화
2. 개별 객체별 단일 클래스 학습 -> 통합학습

---

### Phase 2. 단계별 개선 전략-개별

**🛑방식**
<details><summary>11개 객체 동시 학습</summary>
  <br>
  
  ```

15개 → 11개 객체로 축소
  
주체(2개)            성인, 아이
신체활동(6개)        손들기 주먹 발길질
                     손가락질 웅크림 목 졸림
표정(3개)            울음 고함 정상표정
                     
```

</details>  

**🚨결과**
<details><summary>✔ 성능 개선 결과 그래프 </summary>
<img width="3049" height="1524" alt="image" src="https://github.com/user-attachments/assets/8ff063d2-38f5-46f4-97f0-cd01bf16bffd" />
</details>

- mAP50: 20% → 75%📈
- Precision: 30% → 75%📈
- Recall: 50% → 80%📈


→ 총 9개 모델 생성완료(best03~best13)

---

### Phase 3. 단계별 개선 전략-통합

**🛑방식**
통합 객체 학습

**🚨결과**
<details><summary>✔ 성능 개선 결과 그래프 </summary>
<img width="2942" height="1471" alt="image" src="https://github.com/user-attachments/assets/0ce6ecbf-a2da-4b50-8856-d775e0fd3b99" />
</details>

- mAP50: 75% → 71% 📉
- Precision: 75% → 60% 📉
- Recall: 80% → 48% 📉


**🚥객체 탐지 실패 원인**
1. 클래스 불균형
2. 특징 공간 불일치
3. 예측 충돌
4. 편향된 데이터 수 
5. 모델 간 학습 차이
6. NMS 미흡

**🪛문제점 해결 전략**
1. Oversampling, Augmentation
2. 공통 Backbone, Transfer Learning
3. Soft-NMS, 임계값 조정


---


## ⭐향후 발전 방향

1. 클래스 가중치 조정
2. 데이터 증강
3. OVER/UNDER-SAMPLING
4. FOCAL LOSS 적용
5. 정규화 및 배치 정규화

---

## 🌈팀원 마지막 한마디

| 팀원 | 한마디 |
|---|---|
| 백기림 | 프로젝트를 통해 AI가 아동 보호 분야에서 실질적인 역할을 할 수 있다는 가능성을 확인했습니다. 실제 모델 설계와 검증 과정에서 기술적 역량이 크게 향상되어 의미 있는 경험이었습니다.|
| 진성희 |데이터 불균형과 모델 성능 저하 문제를 해결하는 과정은 도전적이었지만 성장의 기회가 되었습니다. 실시간 탐지 모델을 구현하며 AI 기술의 실제 적용 가능성을 체감할 수 있었습니다.|
| 이기화 | 데이터 라벨링부터 모델링까지 전 과정에서 많은 것을 배우며 협업의 중요성을 깊이 느꼈습니다. 팀과 함께 문제를 해결하며 완성도 있는 결과를 만들어낸 점이 큰 성취였습니다.|

---

