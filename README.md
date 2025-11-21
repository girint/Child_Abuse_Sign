# 👨‍👩‍👦 '어린이집의 가정 내 아동학대 의심 아동 조기 발견 및 신고 자동화 솔루션' JSP 웹 개발 프로젝트

![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/3054c13b-8c4c-4a5d-a99f-41775fd7ef54)

### INDEX
- [프로젝트 소개](#프로젝트-소개)   
- [팀원 구성과 역할](#팀원-구성과-역할)   
1. [개발 환경](#1-개발-환경)
2. [개발 기간 및 작업 관리](#2-개발-기간-및-작업-관리)
3. [요구사항 명세 및 다이어그램](#3-요구사항-명세-및-다이어그램)     
4. [구현 기능](#4-구현-기능)
5. [개인 회고](#5-개인-회고)      

<br>

## 프로젝트 소개

### 👨‍🏫 아동학대 의심 아동 조기 발견 및 신고 자동화 솔루션
- 어린이집 가정 내에서 발생할 수 있는 아동학대를 사전에 감지하고 즉각적인 대응을 지원하기 위한 서비스
  AI 기반 영상 분석 기술을 활용하여 교사와 아이들이 상호작용 및 행동 패턴을 분석하고, 학대로 의심되는
  상황을 탐지하면 자동으로 관리자 또는 관련 기관에 알림을 전달하여 피해를 최소화 하는 것이 목표.
  ### 🔍 특징
  - 교사와 아동의 움직임 및 비정상적 행동 패턴을 학습하여 학대 가능성을 실시간으로 감지
  - 위험 상황 발생 시 관리자와 담당 기관에 즉시 알림전달 및 적시에 대응 가능
  - 공지사항 등록 및 관리 기능 제공
  - 문의하기 게시판을 통해 사용자 질문 확인 및 답변 작성 가능
  - 실제 환경 기반 데이터 축적을 통해 탐지 정확도 향상
  ### 👉 프로젝트 바로가기
  <div align="center">
  
  | <img src="https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/267e789a-5f97-445e-832e-27c5734c111f"  width="145" height="145"/> |<img src="https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/564563d2-47e6-4cfa-b744-b11dc31e0ce9"  width="145" height="145"/> |
  | :---: | :---: |
  | [GitHub](https://github.com/ChunjaeMomCh/MomChannel) | [Notion 팀회의록](https://geee1-kim.notion.site/JSP-Project-2061031e0e3b45148829eac978ca199d?pvs=4) |
 </div>

<br><br>


## 팀원 구성과 역할

<div align="center">

| **백기림** | **진성희** | **이기화** |
| :------: | :------: | :------: |
| [<img src="https://avatars.githubusercontent.com/u/40616792?v=4" height=150 width=150> <br/> @kimg1623](https://github.com/kimg1623) | [<img src="https://avatars.githubusercontent.com/u/145963633?v=4" height=150 width=150> <br/> @lmh9999](https://github.com/LMH9999) | [<img src="https://avatars.githubusercontent.com/u/145524819?v=4" height=150 width=150> <br/> @Lzynee](https://github.com/Lzynee) |
| 코드입력 <br> 머신러닝 모델링 학습 <br> 모델링  <br> 프로젝트 | 모델링  <br> 학습진행 <br> 프로젝트 <br> 프로젝트 | 모델링 <br> 학습진행 <br> 프로젝트 <br> 프로젝트 |


</div>

<br>

## 1. 개발 환경

**Back-End** 
<div>
<img src="https://img.shields.io/badge/Java-007396?style=for-the-badge&logo=Java&logoColor=white">
<img src="https://img.shields.io/badge/JSP-de6c1e?style=for-the-badge&logo=jsp&logoColor=white">
<img src="https://img.shields.io/badge/Servlet-e42c2e?style=for-the-badge&logo=servlet&logoColor=white"><br>
<img src="https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white">
<img src="https://img.shields.io/badge/MariaDB-003545?style=for-the-badge&logo=mariadb&logoColor=white"> 
<img src="https://img.shields.io/badge/Mybatis-000000?style=for-the-badge&logo=Fluentd&logoColor=fff"/><br>
<img src="https://img.shields.io/badge/SHA256+SALT-ecd31d?style=for-the-badge&logo=sha256&logoColor=000"/>
</div>
<br>

**Front-End** 
<div>
<img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=HTML5&logoColor=fff"/>
<img src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=CSS3&logoColor=fff"/>
<img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=JavaScript&logoColor=fff"/><br>
<img src="https://img.shields.io/badge/bootstrap-7952B3?style=for-the-badge&logo=bootstrap&logoColor=fff"/>
<img src="https://img.shields.io/badge/jquery-0769AD?style=for-the-badge&logo=jquery&logoColor=fff"/>
</div> 
<br>

**Tools** 
<div>
<img src="https://img.shields.io/badge/IntelliJIDEA-000000?style=for-the-badge&logo=intellijidea&logoColor=white">
<img src="https://img.shields.io/badge/DBeaver-362822?style=for-the-badge&logo=dbeaver&logoColor=white"><br>  
<img src="https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazonaws&logoColor=white">
<img src="https://img.shields.io/badge/RDS-527FFF?style=for-the-badge&logo=amazonrds&logoColor=white">
<img src="https://img.shields.io/badge/EC2-FF9900?style=for-the-badge&logo=amazonec2&logoColor=white">
</div>  
<br>

**Collaboration** 
<div>
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white">
<img src="https://img.shields.io/badge/gitkraken-179287?style=for-the-badge&logo=gitkraken&logoColor=white">
<img src="https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=Notion&logoColor=white">
</div>
<br>

## 2. 개발 기간 및 작업 관리

### 개발 기간

<details>
  <summary>전체 개발 기간 : 2025-11-03 ~ 2025-11-28</summary>
  ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/40616792/c2b695a4-c8fe-425e-9a10-8558fd2e7f1b)
</details>

<br>

### 작업 관리

- 상황별 모델링을 github에 업로드하여 팀원들에게 모델링 쉽게 다운로드 허용 
- 기타등등
- 기타등등

<br>


## 3. 요구사항 명세 및 다이어그램
👉 각 이미지를 클릭하면 확대되어 자세한 내요을 확인하실 수 있습니다.
<details>
  <summary>요구사항 명세서</summary>

  1. 공통 및 메인화면
  ![image](https://github.com/LMH9999/MomChannel/assets/145963633/a37e28d0-d9cf-4776-9f90-8a23de130f9c)

  2. 학부모 정보 공유 게시판
   ![image](https://github.com/LMH9999/MomChannel/assets/145963633/d770a942-9245-4023-9480-1905d4c6914d)

  3. 채널
    ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/0a3c0d8b-dcbf-43c6-adcc-79277738343e)

  4. 공지사항 게시판
   ![image](https://github.com/LMH9999/MomChannel/assets/145963633/eb2138f7-a46e-4dc7-93e4-09dad58f5f22)

  5. 문의하기 게시판
   ![image](https://github.com/LMH9999/MomChannel/assets/145963633/16f39925-86b9-4b60-acbe-aea63d44a965)
  
  6. 로그인
  ![image](https://github.com/LMH9999/MomChannel/assets/145963633/9a3547b4-8289-40a1-b465-74fcb4307b44)

  7. 회원가입 
  ![image](https://github.com/LMH9999/MomChannel/assets/145963633/aeb9bcad-2557-4e08-8a00-deee7c378537)


</details>

  
<br>


## 4. 구현 기능

### [ 상단 고정 메뉴 ( Header ) ]
- 누르면 메인 페이지로 이동시켜주는 로고
- 게시물 검색 기능
- 로그인이 되어 있지 않은 경우
  - 회원가입
  - 로그인
- 로그인이 되어 있는 경우
  - 내 정보 보기로 이동가능한 프로필 사진
  - 로그인 한 회원님의 아이디 표시
  - 로그아웃 
- 게시판의 종류와 공지사항 문의하기 게시판을 갈 수 있는 풀다운 메뉴

| 비회원 & 메뉴 |
| --- |
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/13121206-b032-4434-ac04-89ce0e749d0e) | 

| 회원 & 풀다운 메뉴 |
| --- |
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/277d38f9-b958-45a1-a9b8-8722eb52f59e) |

<br>

### [ Main Menu ]
- 맘채널에 관한 소개글 홍보글 등이 보이는 슬라이드 쇼
- 다양한 기준에 대한 게시물 추천

| 슬라이드 쇼 |
| --- |
| ![메인화면_슬라이드쇼](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/9ebf0dbe-4d5d-4d3f-8445-47979eb0d922) |




<br>


### [ 회원 가입 ]
- 회원 가입을 하면 DB의 member 테이블에 입력 받은 정보를 저장합니다.
  - 다음 Open API 를 활용한 주소 입력
  - 아이디 중복체크
  - 이메일 양식 확인
  - 비밀번호 확인
- 회원가입 성공 후 로그인 화면으로 이동

| 회원가입 폼 | 주소 API |
| --- | --- |
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/e5e6a6cf-bcf3-46c6-a63f-c3398527323f) | ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/dc0eb020-d02d-4327-8da7-64af1185b878) |

<br>

| 회원 중복 확인 | 알림 |
| --- | --- |
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/68243652-c894-47a7-9205-972805defbf1)  | ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/d0984377-3bf5-4da0-99a8-aeaabca1d6f8) | 

<br>

| 이메일 양식 | 비밀번호 확인 |
| --- | --- |
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/b92a92a8-e357-4000-b8d9-7e9d003b6007) | ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/1fb77b35-7504-4ada-bda5-9ec7a35b7268) |

<br>






<br>

### [ 로그인 ]
- 회원 로그인은 아이디와 비밀번호를 입력하면 DB의 데이터와 비교하여 일치한 경우 로그인에 성공하고 불일치 시 경고창 표시 후 다시 로그인 창으로 이동

| 로그인 폼| 비밀번호 검증 & 정상 로그인 |
| --- | --- |
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/91688597-693c-4ab7-b2f0-be7a282753e2) | ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/06510711-94de-4b2e-8e7e-b880d5dff4e7) <br> <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/cbab9546-bc10-40b1-aeaf-812efd4d7be8) |

<br>


<br>

### [ 내 정보 ]
- 내 정보를 볼 수 있는 내 정보 보기 페이지
- 내 정보 수정
  - 내 정보들과 회원가입 시 기본 이미지 였던 프로필 사진 변경 가능

| 내 정보 보기 | 내 정보 수정 |
| --- | --- |
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/ede1d5ef-098c-472d-ba92-70708bdf534f) | ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/2142b8fe-5c08-4120-b785-15181ceece6d) |


<br>


<br>

#### [ 게시판 ]
- 학부모 회원들이 공유하는 게시글들을 분류별로 모아서 보는 게시판 
  - 회원인 경우  공유하고 싶은 게시글을 작성, 상세보기, 수정, 삭제
  - 게시글 작성 및 수정 시 게시 내용에 관한 지역별, 학년별 분류를 선택
  - 게시글 작성 또는 수정 시 썸네일 이미지를 임의로 선택하지 않을 경우 디폴트 이미지를 자동 선택하여 게시판에 노출시킵니다.


| 게시판 |
|----------|
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/74b9bc50-2b2a-407f-a587-e1f3942f9692) |
| 1. 게시글 작성 |
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/eca5318f-9717-45e3-a187-54b7b8532796) |
| 2. 상세보기 |
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/2c2a40f4-1116-4664-9fee-66e0e340c9b5) |
| 3. 수정하기 |
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/f0de1480-e1e0-49fb-b2f2-0e946ffcb33f) |

<br>

<br>

#### [ 채널 ]
- 회원가입 시 자동 생성되는 채널
  - 회원이 작성한 게시물들을 채널에서 확인할 수 있음
  - 구독 기능 구현 : 회원 당 한 채널은 한번만 구독 가능
  - 채널 정보 편집


| 채널 |
|----------|
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/feee35a7-a23a-46b5-aaf4-3f15f5510f3d) |
| 1. 채널 상세보기 |
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/75705a60-9c41-452c-a226-d405d22263db) |
| 2. 구독 기능 |
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/516df8bc-9260-440e-a23e-f77d1733ce2b) ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/cca13a58-9e0d-4e70-a937-231da379b105) |
| 3. 채널 정보 편집 |
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/e15c9eb9-0b32-479a-a362-2de514ebf905) |

<br>




<br>

#### [ 서비스 안내 ]
- 맘채널 소개
  - 맘채널을 소개하는 포스터
- 공지사항
  - 관리자가 '이벤트 / 공지 / 중요' 카테고리로 나뉜 공지사항을 등록, 수정 및 삭제
- 문의하기
  - 회원이 질문글을 올리고 관리자가 답변을 작성하는 문의하기 게시판

| 맘채널 소개 |
|----------|
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/08df4e92-60de-496b-8f84-5c078ff142f3) |

<br>


<br>

| 공지사항 |
|----------|
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/89495355-a6e7-4078-8364-0158ad400d92) |
| 1. 공지 작성 |
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/4393bacf-f01c-4f09-b75d-9c8bbab26c79) |
| 2. 상세보기 |
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/47fb1fe8-6d0e-4230-8eae-820b161ac5a7) |
| 3. 수정하기 |
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/202c6e3f-fbfb-4362-8900-e89f1eb1c432) |

<br>

| 문의하기 |
|----------|
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/3b89f9e1-4694-4508-a782-38ee7b0af1ec) |
| 1. 문의글 작성 |
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/b75d1c1d-ed5d-4d88-ab50-054d08ca97bd) |
| 2. 상세보기 |
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/1616099f-bcb8-429c-bd66-fc68d2254f8c) |
| 3. 수정하기 |
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/62c14227-1d43-4e0d-aa30-2f9109324d99) |

<br>

| 관리자 답변 작성 |
|----------|
| ![image](https://github.com/ChunjaeMomCh/MomChannel/assets/145963633/85469769-30c6-4dff-994f-594e2d20b9d0) |

<br>







#### [ 하단 고정 메뉴 ( Footer ) ]
- 팀원들의 GitHub와 회의 내용을 기록한 Notion을 표시

| 하단 고정 메뉴 |
|----------|
| ![image](https://github.com/LMH9999/MomChannel/assets/145963633/871a3c12-bd13-429c-87d0-473bc784c109) |


<br>



## 5. 개인 회고

### 🍊 백기림
아동학대라는 중요한 문제를 다루며 기술의 책임감을 느꼈던 프로젝트였습니다. 기능 구현 과정에서 어려움도 있었지만 팀과 함께 해결하며 성장할 수 있었고, 다음에는 탐지 정확도를 높이는 방향으로 더 발전시키고 싶습니다.

<br>

### 👻 진성희
AI 기반 분석 기능을 구현하면서 기술이 실제로 도움이 될 수 있다는 점을 실감했습니다. 개발 과정의 어려움도 있었지만 팀 협업을 통해 해결하며 많은 경험을 쌓을 수 있었습니다.

<br>

### 😎 이기화
AI 기술을 활용해 사회적 문제를 해결할 수 있다는 점이 의미 있었던 프로젝트였습니다. 민감한 데이터를 다루며 윤리적 고려의 중요성을 느꼈고, 팀과 협력하며 기능을 개선해 나가는 과정에서 많은 것을 배웠습니다.

<br>

