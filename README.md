# Emotion_Recognition
AI+X 딥러닝 Final Project


[월간 데이콘 음성 감정 인식 AI 경진대회](https://dacon.io/competitions/official/236105/overview/description)

-----------------------------

### Members
* 정재희 | 융합전자공학부 | jayjeong7364@outlook.kr

* 이창민 | 융합전자공학부 | lcentimeter@hanyang.ac.kr

* 정지훈 | 정책학과 | wond1216@naver.com

----------------------------

### 주제
음향 데이터 기반 감정 인식 AI 모델 개발



### 설명
음향 데이터들로부터의 감정을 인식(Emotion Recognition)하는 모델을 만들어 주세요!

### Proposal (Option 1)

### Dataset Info.

* train [폴더]
학습을 위한 소리 샘플
TRAIN_0000.wav ~ TRAIN_5000.wav


* test [폴더]
추론을 위한 소리 샘플
TEST_0000.wav ~ TEST_1880.wav


* train.csv [파일]
id : 샘플 별 고유 ID
path : 음향 샘플 파일 경로
label : 감정의 종류
0: angry
1: fear
2: sad
3: disgust
4: neutral
5: happy


* test.csv [파일]
id : 샘플 별 고유 ID
path : 음향 샘플 파일 경로


* sample_submission.csv [파일] - 제출 양식
id : 샘플 별 고유 ID
label : 예측한 감정의 종류

-------------------------
## License & Rights
This Crowd-sourced Emotional Mutimodal Actors Dataset (CREMA-D) is made available under the Open Database License: 

http://opendatacommons.org/licenses/odbl/1.0/. 

Any rights in individual contents of the database are licensed under the Database Contents License: 

http://opendatacommons.org/licenses/dbcl/1.0/
