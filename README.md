# Emotion_Recognition
AI+X 딥러닝 Final Project



-----------------------------
### Title: 음향 데이터 기반 감정 인식 AI 모델 개발
>음향 데이터들로부터의 감정을 인식(Emotion Recognition)하는 모델
### Members
* 정재희 | 융합전자공학부 | jayjeong7364@outlook.kr | wav2vec 을 이용한 데이터 전처리 및 학습

* 이창민 | 융합전자공학부 | lcentimeter@hanyang.ac.kr | Librosa 를 이용한 데이터 전처리 및 학습

* 정지훈 | 정책학과 | wond1216@naver.com | 데이터 셋 수집, 관련 알고리즘 조사, 데이터 학습, 설명 발표 [영상](https://youtu.be/i53mOiCMns4)

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
> 저희는 크게 두가지 방식으로 진행하였습니다. 첫번째로 WAV2VEC을 이용하였고, 두번째로 LIBROSA를 이용하였습니다.
### [1] Wav2vec 2.0
-	2020년에 Facebook에서 발표한 트랜스포머를 사용한 자기지도학습 모델
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

#### Code 설명
```python
def speech_file_to_array_fn(df):
    feature = []
    for path in tqdm(df['path']):
        speech_array, _ = librosa.load(path, sr=CFG['SR'])
        feature.append(speech_array)
    return feature
```
>위 함수는 먼저 빈 리스트인 feature를 초기화하고 df['path']에서 각 경로를 반복하면서 librosa.load 함수를 사용하여 해당 경로의 음성 파일을 불러온다. 이때, 샘플링 주파수는 CFG['SR']로 설정된다. 로드된 음성 파일은 speech_array에 할당되고 이를 feature 리스트에 추가한다. 모든 음성 파일에 대해 위의 작업을 반복한 후, 최종적으로 feature 리스트를 반환한다. 이 리스트는 DataFrame의 'path' 열에 지정된 모든 음성 파일들이 배열로 변환된 것을 포함하고 있다.
```python
def create_data_loader(dataset, batch_size, shuffle, collate_fn, num_workers=0):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      collate_fn=collate_fn,
                      num_workers=num_workers
                      )

train_dataset = CustomDataSet(train_x, train_df['label'], processor)
valid_dataset = CustomDataSet(valid_x, valid_df['label'], processor)

train_loader = create_data_loader(train_dataset, CFG['BATCH_SIZE'], False, collate_fn, 16)
valid_loader = create_data_loader(valid_dataset, CFG['BATCH_SIZE'], False, collate_fn, 16)
```
>create_data_loader 함수는 주어진 데이터셋과 매개변수를 사용하여 DataLoader 객체를 생성하는 함수이다. 먼저 함수는 주어진 dataset, batch_size, shuffle, collate_fn, num_workers를 매개변수로 받는다. 이 함수는 torch.utils.data.DataLoader 클래스를 사용하여 데이터로더 객체를 생성하는데 주어진 데이터셋과 매개변수를 사용하여 데이터 로딩 및 배치 처리 설정을 구성한다. 배치 크기, 셔플 여부, 데이터 배치 처리 함수, 워커 개수 등의 매개변수를 설정한다. 이렇게 생성된 데이터 로더를 사용하면 학습과 검증 과정에서 데이터를 배치 단위로 로딩하고 필요한 전처리 작업 등을 수행할 수 있게 된다.
```python
def validation(model, valid_loader, creterion):
    model.eval()
    val_loss = []

    total, correct = 0, 0
    test_loss = 0

    with torch.no_grad():
        for x, y in tqdm(iter(valid_loader)):
            x = x.to(device)
            y = y.flatten().to(device)

            output = model(x)
            loss = creterion(output, y)

            val_loss.append(loss.item())

            test_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += y.size(0)
            correct += predicted.eq(y).cpu().sum()

    accuracy = correct / total

    avg_loss = np.mean(val_loss)

    return avg_loss, accuracy
```
>validation 함수는 모델을 평가하기 위한 함수이다. 이 함수는 주어진 모델을 평가 모드로 설정하고 검증 데이터로더를 사용하여 모델의 성능을 평가한다. 이를 위해 검증 데이터로더에서 배치별로 데이터를 가져온다. 그리고 이동시킨 데이터를 모델에 입력으로 전달하여 출력을 얻는다. 출력과 정답 간의 손실을 계산하기 위해 손실 함수를 사용하고 계산된 손실은 검증 손실 리스트에 추가된다. 정확도를 계산하기 위해 예측된 출력과 정답을 비교하는데 예측된 출력에서 최대값을 찾아 정답과 비교하여 정확한 예측 개수를 계산한다. 모든 배치에 대해 손실과 정확도를 계산한 후 평균 검증 손실과 정확도를 한다.
```python
class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.model = audio_model
        self.model.classifier = nn.Identity()
        self.classifier = nn.Linear(256, 8)

    def forward(self, x):
        output = self.model(x)
        output = self.classifier(output.logits)
        return output
```
>BaseModel은 음성 분류를 위한 모델이다. 해당 모델은 audio_model (facebook/wav2vec2-base, wav2vec2모델)이라는 사전 학습된 음성 분류 모델을 기반으로 구성되어 있다. 모델의 구조를 설정할 때 기존 모델의 분류기를 nn.Identity()로 대체하였고 nn.Linear(256, 8)을 분류기로 설정했다. 이 모델은 입력 데이터를 받아 모델을 통과시킨 후 출력값을 반환하는 모델이다.
```python
def train(model, train_loader, valid_loader, optimizer, scheduler):
    accumulation_step = int(CFG['TOTAL_BATCH_SIZE'] / CFG['BATCH_SIZE'])
    model.to(device)
    creterion = nn.CrossEntropyLoss().to(device)

    best_model = None
    best_acc = 0

    for epoch in range(1, CFG['EPOCHS']+1):
        train_loss = []
        model.train()
        for i, (x, y) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            y = y.flatten().to(device)

            optimizer.zero_grad()
            
            output = model(x)
            loss = creterion(output, y)
            loss.backward()

            if (i+1) % accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss.append(loss.item())

        avg_loss = np.mean(train_loss)
        valid_loss, valid_acc = validation(model, valid_loader, creterion)

        if scheduler is not None:
            scheduler.step(valid_acc)

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model

        print(f'epoch:[{epoch}] train loss:[{avg_loss:.5f}] valid_loss:[{valid_loss:.5f}] valid_acc:[{valid_acc:.5f}]')
    
    print(f'best_acc:{best_acc:.5f}')

    return best_model
```
>train 함수는 모델을 학습시키는 역할을 한다. 주어진 학습 데이터로더를 사용하여 모델을 반복적으로 학습하고 옵티마이저를 활용하여 가중치를 하게되는데 학습 중에는 모델을 train 모드로 설정하여 드롭아웃 및 배치 정규화와 같은 기법들을 적용하였다. 또한 배치 크기에 따른 그래디언트 누적을 처리하기 위해 accumulation_step을 설정했고 학습 과정에서 발생한 손실값들을 기록하고 주기적으로 검증 데이터를 사용하여 모델의 성능을 평가한다. 성능이 개선될 때마다 최적의 모델을 저장하고 학습이 완료된 후에는 최적의 모델을 반환한다.
```python
model = BaseModel()
optimizer = torch.optim.Adam(model.parameters(), lr=CFG['LR'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
infer_model = train(model, train_loader, valid_loader, optimizer, scheduler)
test_df = pd.read_csv('./test.csv')
def collate_fn_test(batch):
    x = pad_sequence([torch.tensor(xi) for xi in batch], batch_first=True)
    return x
test_x = speech_file_to_array_fn(test_df)
test_dataset = CustomDataSet(test_x, y=None, processor=processor)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, collate_fn=collate_fn_test)
def inference(model, test_loader):
    model.eval()
    preds = []

    with torch.no_grad():
        for x in tqdm(iter(test_loader)):
            x = x.to(device)

            output = model(x)

            preds += output.argmax(-1).detach().cpu().numpy().tolist()

    return preds
preds = inference(infer_model, test_loader)
submission = pd.read_csv('./sample_submission.csv')
submission['label'] = preds
submission.to_csv('./baseline_submission.csv', index=False)
```
>위 코드는 위에서 설계한 모델과 전처리된 데이터를 train 함수를 사용하여 훈련시키고 생성된 infer_model을 inference 함수를 사용하여 추론을 수행하고 예측결과를 preds에 담는다. 그리고 예측결과를 baseline_submission.csv라는 파일에 출력하게 된다.

-------------------------
### [2] Librosa
음성파일에 감정과 상관관계에 있는 요소들이 무엇이 있을까 생각을 해보았습니다. 음높이, 강도, 템포, 음색 등등을 생각해 내었고 이 네 가지를 이용하여 모델을 만들면 좋을 것 같다고 판단하였습니다. 데이터 셋으로부터 이 특징들을 추출하기 위하여 numpy와 librosa를 이용하였습니다.
```python
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
features_dir = 'extracted_features'
os.makedirs(features_dir, exist_ok=True)
```
> 데이터셋을 로드합니다. 데이터셋 전처리 결과를 저장할 폴더를 만듭니다. 전처리 과정이 너무 오래 걸려서 최초실행시에만 추출하고 그 뒤로는 저장된 값을 사용하게 했습니다. 
```python
def preprocess_audio(file_path):
    features_file = os.path.join(features_dir, os.path.splitext(os.path.basename(file_path))[0] + '.npy')
    if os.path.exists(features_file):
        print(f"Loading features from {features_file}")
        return np.load(features_file)
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', mono=True)
    pitch = librosa.yin(audio, fmin=100, fmax=1000)
    energy = np.mean(audio ** 2)
    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sample_rate)[0]
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
```
> 전처리를 담당하는 함수입니다. 먼저 추출된 특징이 이미 있으면 백업파일을 사용하게 했습니다.
> Librosa와 Numpy를 이용하여 pitch, intensity 를 추출합니다.
> 각 시점에 발생할 수 있는 ‘onset’, 오디오 신호의 특별한 변화점들을 추출한 뒤, 이를 이용하여 tempo를 추출합니다.
> 이번엔 톤을 추출하기 위해 mfccs를 추출합니다. Mfcc는 Mel-frequency cepstral coefficients로 노이즈를 제거하고 중요한 음색, 톤을 추출하는 기능을 합니다. 이렇게 추출한 톤의 변화율과 가속률을 또 추출합니다.
```python
    max_len = 1000
    pitch = np.pad(pitch, (0, max_len - len(pitch)))
    mfccs = np.pad(mfccs, ((0, 0), (0, max_len - mfccs.shape[1])))
    delta_mfccs = np.pad(delta_mfccs, ((0, 0), (0, max_len - delta_mfccs.shape[1])))
    delta2_mfccs = np.pad(delta2_mfccs, ((0, 0), (0, max_len - delta2_mfccs.shape[1])))
    features = np.concatenate((pitch, [energy], [tempo], mfccs.flatten(), delta_mfccs.flatten(), delta2_mfccs.flatten())
    np.save(features_file, features)    
    print(f"Extracted features saved to {features_file}")
    return features
```
> 이후 행렬계산을 위해 max_len으로 패딩을 하고 추출값을 백업합니다.
```python
train_df['audio_features'] = train_df['path'].apply(preprocess_audio)
test_df['audio_features'] = test_df['path'].apply(preprocess_audio)

X = np.stack(train_df['audio_features'].values)
y = train_df['label'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

X_test = np.stack(test_df['audio_features'].values)
test_predictions = model.predict(X_test)
test_df['predicted_label'] = label_encoder.inverse_transform(test_predictions)

test_df.to_csv('test_predictions.csv', index=False)
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'label': test_df['predicted_label']
})
submission_df.to_csv('submission.csv', index=False)
```
> 데이터 셋을 전처리합니다. 
> 추출한 특징들을 X에 정리하고 y에 label을 정리합니다. 그후 validation을 위해 0.2의 비율로 나눕니다. 그후 라벨을 숫자로 encode를 합니다.
> 학습을 시킵니다. SVC모델을 사용하였고 이후 Random Forest, Logistic Regression 으로도 해보았습니다. 
> 학습된 모델을 바탕으로 test 데이터에 테스트를 합니다. submission.csv로 저장하였습니다.

## IV. Evaluation & Analysis	
| id        | label |
|:---------:|:-----:|
| TEST_0000 | 5     |
| TEST_0001 | 0     |
| TEST_0002 | 2     |
| TEST_0003 | 2     |
| TEST_0004 | 5     |
| TEST_0005 | 4     |
| TEST_0006 | 2     |
| TEST_0007 | 3     |
| TEST_0008 | 1     |
| TEST_0009 | 5     |
| TEST_0010 | 1     |
| TEST_0011 | 4     |
| TEST_0012 | 2     |
| TEST_0013 | 0     |
| TEST_0014 | 0     |
| TEST_0015 | 3     |
| TEST_0016 | 0     |
| TEST_0017 | 1     |
| TEST_0018 | 2     |
| TEST_0019 | 4     |
| TEST_0020 | 2     |
| TEST_0021 | 2     |
| TEST_0022 | 5     |
| TEST_0023 | 5     |
| TEST_0024 | 4     |
| TEST_0025 | 1     |
| TEST_0026 | 2     |
| TEST_0027 | 2     |
| TEST_0028 | 5     |
........


#### submission.csv가 위와 같이 1880개의 데이터셋을 테스트 하였고 데이콘에 제출한 결과 wav2vec을 이용한 정확도는 약 0.32, librosa를 이용한 정확도는 약 0.45으로 나왔습니다. 찍었을때 확률이 0.16161..임을 감안하면 약 2.8배 정확한 걸 알 수 있습니다. 그래도 많이 부족한 결과임을 알 수 있었습니다.
	
<img width="100" alt="화면 캡처 2023-06-09 133935" src="https://github.com/chminsta/Emotion_Recognition/assets/119744076/e6064874-5881-456e-a45d-b6284ebb4d0d">

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
