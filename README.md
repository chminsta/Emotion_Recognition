# Emotion_Recognition
AI+X 딥러닝 Final Project



-----------------------------
### Title: 음향 데이터 기반 감정 인식 AI 모델 개발
>음향 데이터들로부터의 감정을 인식(Emotion Recognition)하는 모델
### Members
* 정재희 | 융합전자공학부 | jayjeong7364@outlook.kr

* 이창민 | 융합전자공학부 | lcentimeter@hanyang.ac.kr

* 정지훈 | 정책학과 | wond1216@naver.com

----------------------------



## I. Proposal (Option 1)
### Motivation : 
음성 감정 인식은 사람들의 심리적 상태를 이해하고 문제를 해결하는 데에 도움을 줄 수 있습니다. 

가령 우울증이나 불안장애를 가진 사람들에게 도움을 주고 스트레스나 분노로 인한 갈등을 미리 예방하는 데에도 도움을 줄 수 있습니다. 

또한, 상업적인 관점에서도 고객의 음성을 통해 감정과 반응을 파악함으로써 기업의 제품 및 서비스의 개선방안을 설정할 수 있습니다. 

음성 데이터를 다루는 것이 비교적 생소하지만, 수업 시간에 배운 딥러닝 기법을 이용하여 음성 데이터를 분석해보고 싶어 해당 주제를 선택하게 되었습니다.


### What do you want to see at the end?
어떤 음향 데이터든 감정을 정확하게 인식하는 모델을 만들고 싶습니다.

## II. Datasets

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

------------------------------------
## III. Methodology
### Wav2vec 2.0
-	2020년에 Facebook에서 발표한 트렌스포머를 사용한 자기지도학습 모델
-	자기지도학습이란 라벨이 없는 데이터의 집합에서 특성을 배우는 학습 방법
-	라벨링 되어있지 않은 데이터로 표현학습을 한 후 소량의 라벨링 된 데이터를 사용하여 fine-tuning함

<p align="center">
  <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbqP9vg%2FbtrqAd1cWv6%2FZ0NExFFGkxOVZKrkmne3d1%2Fimg.png">
</p>
이미지 출처: https://sooftware.io/wav2vec2/)


#### Feature Encoder

-	Feature Encoder는 입력 데이터에서 중요한 특징이나 표현을 추출하는 계층인데, 보통 입력데이터를 저차원의 특성벡터로 변환하여 정보를 의미 있는 형태로 표현한다.
	  1.	Raw waveform(음성 데이터)를 CNN 인코더에 넣어 25ms의 표현벡터로 변환한다.
    2. 변환된 데이터를 transformer encoder와 quantizer에 공급한다.

#### Quantization Module
1.	양자화는 신호를 작은 구간으로 분할하여 각 구간에 대한 표현을 선택하는 과정이다. wav2vec에서는 신호를 일련의 구간으로 분할하고, 각 구간에 대해 가장 가까운 벡터를 선택하여 양자화한다. 이러한 양자화된 벡터들은 신호의 저차원 표현으로 사용된다.

2.	양자화 모듈은 wav2vec의 학습 중에 사용된다. wav2vec는 자기 지도 학습(self-supervised learning) 방식으로 학습되는데, 입력 신호의 일부 구간을 양자화한 뒤, 양자화된 벡터를 예측하는 작업을 수행한다. 이러한 자기 지도 학습은 대규모 음성 데이터를 사용하여 모델을 사전 훈련할 수 있게 해준다.

3.	양자화 모듈은 wav2vec 모델의 성능을 향상시키는 데 도움을 준다. 양자화는 웨이브폼 신호를 저차원 벡터로 변환하여 메모리 사용량을 줄이고, 학습 및 추론 속도를 향상시킬 수 있다. 또한, 양자화 모듈은 wav2vec 모델의 로버스트성(robustness)을 향상시키는 데 도움을 줄 수 있다.

#### Transformer Module
1.	Transformer 모듈은 전체 오디오 시퀀스의 정보를 추가한다.
2.	입력인 z(음성 표현 벡터)를 마스킹 트랜스포머에 넣으면, 주변 정보를 이용하여 복원된 context representations인 c를 생성한다.
3.	트랜스포머의 출력은 contrastive task(대조 작업)를 푸는데 사용되며, 모델은 마스크된 위치에 대한 정확한 양자화된 음성 단위를 식별해야 한다.
4.	목표는 context representations과 해당 위치의 latent speech representations이 유사하도록 학습하는 것입니다. 이를 위해 contrastive loss를 사용한다.
5.	contrastive loss를 최소화하면 음성 데이터 안에 공통적으로 가지고 있는 상호 정보를 최대화할 수 있다.
6.	wav2vec 2.0을 학습한 후 fine-tuning을 수행하면 적은 데이터로도 좋은 성능을 보장할 수 있다.


-------------------------

## IV. Evaluation & Analysis
- Graphs, tables, any statistics (if any)

-------------------------

## V. Related Work
- https://nongnongai.tistory.com/34
- [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/pdf/2006.11477.pdf)
- [Audio Data EDA](https://www.kaggle.com/code/psycon/audio-data-eda-processing-modeling-recommend#EDA)
- [wav2vec 2.0: learning the structure of speech from raw audio](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/)
- https://smilegate.ai/2020/08/05/wav2vec-2/
- https://kaen2891.tistory.com/83  
- https://zerojsh00.github.io/posts/Wav2Vec2/
- https://zerojsh00.github.io/posts/Vector-Quantization/
- [월간 데이콘 음성 감정 인식 AI 경진대회](https://dacon.io/competitions/official/236105/overview/description)

--------------------------------------

## VI. Conclusion: Discussion


-------------------------
## License & Rights
This Crowd-sourced Emotional Mutimodal Actors Dataset (CREMA-D) is made available under the Open Database License: 

http://opendatacommons.org/licenses/odbl/1.0/. 

Any rights in individual contents of the database are licensed under the Database Contents License: 

http://opendatacommons.org/licenses/dbcl/1.0/
