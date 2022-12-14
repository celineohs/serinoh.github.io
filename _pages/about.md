---
layout: about
title: main
permalink: /
subtitle: "2022학년도 2학기, 인간 뇌이미징의 데이터사이언스"

profile:
  align: right,
  address: >
    <p>08826</p>
    <p>1, Gwanak-ro, Gwanak-gu,</p>
    <p>Seoul, Republic of Korea</p>

news: false  # includes a list of news items
selected_papers: false
# includes a list of papers marked as "selected={true}"
social: true  # includes social icons at the bottom of the page
---

Hi, I'm Serin Oh. <br>
I'm a senior at Seoul National University, majoring in Psychology and Linguistics. <br>
<br>
<br>
<br>
<h1> Final Article </h1> <br>
<h1> 아동ꞏ청소년 대상 자살 예방 프로젝트 </h1>
<h2> 머신러닝 모델을 기반으로 </h2> <br>

우리나라에서 자살률을 낮추기 위한 시도는 줄곧 이어져 왔지만, 우리나라는 아직도 OECD 국가 중 자살률 1위라는 오명을 벗지 못하고 있습니다. 이처럼 지금까지의 자살 예방이 효과적이지 않은 이유는 무엇일까요? 어쩌면, 우리는 자살 예방의 방향성을 잘못 설정하고 있지는 않나요? <br>

이 문제에 답하기 위해, 저희는 <**자살 사고**>라는 변인에 집중했습니다. 자살 사고를 하는 사람들이 일반 사람들보다 자살을 시도할 확률이 높다는 예측은 쉽게 할 수 있습니다. 물론 사람들이 자살 사고를 하지 않게끔 유도하는 것도 중요하지만, 저희는 사람들이 어떠한 요인으로 인해 자살 사고를 하는지 그 상관성을 밝혀내는 일이 시급하다고 느꼈습니다. 지금 이 순간에도 자살 사고를 하는 사람들이 있고, 자살 사고를 예방하는 것보다 이미 자살 사고를 하는 사람들에 대해 집중적인 지원을 하는 것이 우선이라고 생각했기 때문입니다. <br>

자살 사고에 대해 좀 더 알아보자면, 자살 사고는 심리적 요인, 생물학적 요인, 유전적 요인, 사회적 요인 등 여러 요인에 의해 유발됩니다. 저희는 다방면에서 자살 사고를 예측하기 위해 최대한 다양한 요인을 활용하고자 하였고, 그를 위한 데이터로 The Adolescent Brain Cognitive Development, 즉 아동/청소년의 육체적/심리적 건강과 뇌 발달을 알아보기 위하여 수집한 최대의 데이터 셋인 **ABCD data**를 채택하였습니다. <br>

**하지만,** 몇 가지 선행 연구를 살핀 결과 ABCD의 전체 데이터를 활용하는 것이 도리어 자살 사고에 대한 예측력을 낮춘다고 보고된 바 있었습니다. 또한, 저희는 무턱대고 낙관적인 전망이 아닌, 현실적인 방안을 제안하고자 하였기 때문에 정해진 예산 안에서 해결책을 모색하고자 하였습니다. 이에 따라 주어진 자료에서 몇 개 변인을 추출하는 작업을 먼저 거쳤습니다. <br>

ABCD에서 주어진 뇌 이미징 데이터는 fMRI, DTI, structural MRI의 3가지 종류였습니다. 예산의 한계로 모든 아동들에게서 세 데이터를 모두 수집하는 것은 어려우리라고 예상했고, 저희는 선택적으로 데이터를 사용하기로 결정하였습니다. <br/>

먼저, fMRI는 BOLD signal을 통해 뇌의 neural firing을 관찰하는 기법으로 functional connectivty와 activation pattern을 파악할 수 있다는 뚜렷한 장점을 지니고 있습니다. 그러나 이 과정에서 높은 비용이 필요할 뿐더러 검사 시간 동안 기기 안에서 자세를 고정해야 하므로 어린 아동에게 부적합한 방법입니다. 또한, 자살 사고를 예측하는 데 있어 유의미한 연관성을 가지는 것은 Emotional N-Back task에 대한 fMRI라고 알려져 있습니다. 그러나 주어진 데이터에서는 SST에 참여한 참가자들의 뇌 영상만 있었기 때문에, 저희의 목적에는 부합하지 않는다고 여겨 배제하였습니다. 다음으로 DTI은 물 분자 확산 기법을 통해 WM/GM Connectivity를 살피는 방법입니다. 이 기법에 대해서는, 선행 연구에서 DTI를 활용해 자살 사고를 유의미하게 예측했다는 연구는 타 뇌 이미징 기법에 비해 적었으며 fMRI와 마찬가지로 적지 않은 비용이 소모된다는 점을 들어 배제하였습니다. <br/>

마지막으로, **sMRI**는 말 그대로 뇌의 구조적 이미지를 살피는 방안입니다. 최근의 한 선행 연구에서는 Cortical thickness가 자살 사고와 연관돼 있고(Colic et al., 2022), 특히나 주요 우울 장애를 가진 아동에 대해서는 자살 시도 여부를 분류하는 데 뛰어난 성능을 가졌다고 보고된 바 있습니다(Hong et al., 2020). 이를 기반으로 저희는 경제성까지 함께 고려했을 때 sMRI야말로 개인의 선천적인 이유, 혹은 오랜 기간의 역기능이 반영된 자료이자 객관적으로 확인 가능한 바이오마커의 기능을 수행할 수 있으리라고 판단하였습니다. <br/>

그 다음으로는 사회적/심리적 요인을 파악하기 위해 **CBCL 데이터**를 선택했습니다. CBCL 데이터는 부모 대상으로 아동의 행동을 관찰한 결과를 답변해 얻은 자료이므로 경제적으로도 효율적이며, 수많은 연구에서 설명력이 가장 높은 변인으로 꼽혔기 때문입니다(Harman et al., 2021). 더구나 앞서 언급한 바이오마커를 포함해 유전적/선천적 취약성이 외현화되는 지표라고도 볼 수 있었습니다. 이 외에도, demographic 변인 중 선행연구에서 자살 사고에 유의미한 영향을 미친다고 보고된 부모의 교육 수준과 가정 소득을 포함해(Assari, 2020) **Psychosocial Factor**라는 명칭으로 분류하였습니다. <br/>

유전적 요인에 대해서는 **PRS,** 즉 polygenic risk score 자료를 선정하였습니다. 이는 개인이 가진 유전자가 특정 변인에 대해 얼마나 취약한지 계산한 점수로, 여러 선행 연구에서 자살 사고에 큰 부분을 차지하는 정신 장애나 가족력을 예측하는 데도 유의미한 성능을 거둔 것으로 알려져 있었습니다(Hubers et al., 2022; Otsuka et al., 2021). <br/>

따라서 저희는 이 세 변인을 적절히 조합하여 만든 데이터를 통해, 머신러닝 기법을 적용해 자살 사고를 예측하고 예방하기 위해 가장 중요한 데이터가 무엇일지 알아보고자 하였습니다. <br/>
<br>

<h2> 머신러닝 모델링 </h2>
이제부터 저희가 자살 사고를 예측하기 위해 선택한 방법인 **머신러닝 과정**에 대해 소개할 예정입니다. 모든 단계 R studio를 통해 이루어졌음을 미리 밝힙니다. <br/>
<br/>
<h3> 데이터 전처리 </h3>
ABCD 데이터를 분석에 그대로 사용할 수는 없으므로 **데이터 전처리** 과정에서는 총 5개 단계를 거쳤습니다. 우선, 파일로 분리된 여러 ABCD 데이터를 하나로 종합하는 Data Merging, 값이 없는 데이터를 처리하는 Data Handling, 특정 기준에 따라 데이터를 솎아내는 Data Filtering, 데이터의 값을 표준화하는 Data Scaling, 마지막으로 Categorical Variable에 대해 변수를 조작하는 Dummy Coding 단계를 거쳤습니다. 해당 과정을 도식적으로 표현한 파이프라인은 다음의 [Link](https://drive.google.com/file/d/1OzHm-drP79zVN63hHjxE4R0TtrDGONIB/view?usp=sharing)에서 확인하실 수 있습니다.
<br/>
<br/>
<h3> 데이터 분석 <h3/>
저희가 가장 먼저 분석한 데이터는 단일 변인 중 가장 높은 예측력을 지닌다고 알려진 CBCL 자료였습니다. <br>

본격적으로 데이터 분석을 시작하기 위해, 모델을 학습할 Train Data Set과 생성된 모델의 일반화 가능성을 예측할 Test Data Set은 7:3의 비율로 주어진 데이터를 무작위하게 분리하였습니다. 예측하고자 하는 변수는 Healthy Condition과 Suicide Ideation의 그룹 간 대조 결과를 0, 1로 표현한 HCvsSI였고, p-value가 0.05 이하인 GLM 모델을 돌렸습니다. 앞선 선행 연구에서 참가자들의 나이와 성별을 주요한 covariate로 설정하였기 때문에, 본 연구에서도 age와 sex 값을 covariate로 지정해주었습니다. <br/>
<br>
"Random Data Split" <br/>
set.seed(1) <br/>
train.index <- createDataPartition(data_cbcl$HCvsSI, p = .7, list = FALSE) <br/>
train <- data_cbcl[ train.index,] <br/>
test  <- data_cbcl[-train.index,] <br/>

<br/>

"Feature Selection" <br/>
y <- "HCvsSI" <br/>
x <- setdiff(names(train), y) <br/>
<br/>
train[, y] <- as.factor(train[, y]) <br/>
test[, y] <- as.factor(test[, y]) <br/>
<br/>
x_significant <- c() <br/>
for (i in x){
  form <- paste(y,'~',i)
  fit <- glm(as.formula(form), family='binomial', data=train)
  p_value <- coef(summary(fit))[,4][2]
  if(p_value < 0.05){
    x_significant <- append(x_significant, names(p_value))
  }
} <br/>
<br/>
<Add Covariate> <br/>
x_significant <- append('age', x_significant) <br/>
x_significant <- append('sex.1', x_significant) <br/>

<br/>

저희가 머신러닝 모델링에 사용한 h2o는 java 기반의 머신러닝 플랫폼으로, h2o에서 요구하는 버전의 java jdk를 다운받고, h2o.init() 함수를 사용하면 R과 손쉽게 연동이 가능하다는 장점을 가지고 있습니다. <br/>
<br/>

"Prepare h2o" <br/>

install.packages(”h2o”) <br/>

library(h2o) <br/>

h2o.init() <br/>

<br/>

또한, 5-fold cross validation을 활용해 모델을 구성했습니다. Cross-validation 기법은 모델을 학습할 때, training dataset을 k등분하여 그 안에서 validation set을 여러 번 바꿔가며 학습하는 것을 의미합니다. Training set 안에서 validation set을 따로 구분하는 이유는 쉽게 말하자면 하이퍼 파라미터를 설정하기 위해서입니다. Validation set을 따로 두지 않으면, Test set에 의해 하이퍼 파라미터가 설정되기도 하므로 그 가능성을 차단하기 위해 시행하는 것이죠. 여기서는 k의 자리에 5를 넣었으니 각기 다른 validation set에 활용한 학습을 5회에 걸쳐 하게 됩니다.  <br/>

<br/>

"5-fold cross validation" <br/>

fold_column <- rep(0, nrow(train)) <br/>
train <- data.frame(fold_column, train) <br/>

k_folds <- 5 <br/>
set.seed(1) <br/>
train$fold_column <- createFolds(train$HCvsSI, k=k_folds, list=FALSE) <br/>

<br/>

데이터셋의 기초 작업을 해두고, train dataset을 학습한 모델링 방식은 아래와 같습니다. <br/>
<br/>

"ML Modeling" <br/>
train_h2o <- as.h2o(train, use_datatable = TRUE) <br/>
aml <- h2o.automl(x = x_significant, y = y,
training_frame = train_h2o,
max_models = 20,
seed = 1,
fold_column = 'fold_column') <br/>

<br/> 이때 max_models는 한 번에 비교할 모델의 수로, 여기서는 20을 기재하였으나 더 높은 수도 얼마든지 지정할 수 있습니다.
<br>
<br/>

"Checking the LeaderBoard" <br/>
lb <- aml@leaderboard <br/>
h2o.clusterStatus() <br/>
as.data.frame(lb) %>%  # use print(lb, n = nrow(lb)) to print all rows instead of default (6 rows)
kbl() %>%
kable_styling(bootstrap_options=c('condensed','striped')) <br/>

<br/>

이렇게 학습된 모델들을 바탕으로, 저희는 리더보드 모델만 골라 미리 만들어둔 Test Dataset을 예측해보고자 하였습니다. 나아가 주어진 데이터 중 어떤 변인이 종속변수를 설명하는 데 가장 유의미하였는지 Important Variable 또한 살펴보았고요. <br/>
<br>

"Making Prediction" <br/>
test_h2o <- as.h2o(test, use_datatable = TRUE) <br/>
perf <- h2o.performance(aml@leader, test_h2o) <br/>
perf <br/>

"Checking Important Variables" <br/>
df <- as.data.frame(h2o.varimp(aml@leader)) <br/>
df[1:20,] %>%
kbl() %>%
kable_styling(bootstrap_options=c('condensed','striped')) <br/>

<br/>

마지막으로는 모델의 explainability를 확인하였습니다. <br/>
<br>
"Estimating Explainability" <br/>
exp <- h2o.explain(aml, test_h2o) <br/>
exp <br/>

<br/>

<h3> 데이터 선정 </h3>
모델을 학습시키기 위해 사용한 데이터셋에 대해 다시 돌아오자면, 데이터 선정 과정은 크게 두 단계로 이루어졌습니다. 먼저, Psychosocial 데이터를 만들기 위해서는 CBCL 데이터와 demographic 데이터의 통합이 필요했습니다. 그런데 이때, 한 선행 연구에서는 CBCL의 전체 데이터를 활용하는 것보다는 일부 영역으로 자살 사고를 예측하는 것이 더욱 효과적이라는 주장이 있었습니다. <br>

그에 따라 저희는 Psychosocial 모델의 예측력을 높이기 위해 CBCL 전체 모델과 일부 영역만 추출한 CBCL Region 모델의 예측력을 비교하고, 둘 중 더 좋은 성능을 보인 모델의 데이터를 Psychosocial 데이터셋에 포함하고자 하였습니다. 그 결과, 선행 연구와는 달리 앞선 예시로 들었던 CBCL 전체 데이터가 더 높은 설명력을 보여주었기 때문에 Psychosocial 데이터셋을 구성할 때는 CBCL의 일부 데이터가 아닌, 전체 데이터를 사용하였습니다. <br/>

다음으로, 이렇게 생성된 Psychosocial 모델이 과연 CBCL 모델보다 높은 예측력을 보이는지 확인하였습니다. 또한, 이 Psychosocial 데이터를 토대로 앞서 선정했던 두 변인인 sMRI와 PRS 각각을 조합해 각 모델의 성능까지 검토하였습니다. 모든 과정은 위에서 포함한 코드를 따라 이루어졌습니다. <br/>

이 모든 데이터 분석 과정을 요약한 사진은, [Link2](drive.google.com/file/d/1EgPBiYVe3Lk1MS0uM9S_p4myndc_6v6W/view?usp=sharing)에서 확인하실 수 있습니다.
<br>
<br>
<br>
<h2> 머신러닝 모델링 결과 </h2>
결론부터 이야기하자면, 저희가 Psychosocial 변수를 만들고 후에 비교한 여러 모델 중 <Psychosocial + PRS 모델>이 가장 뛰어난 성능을 보여주었습니다. 저희가 확인한 종속변수는 AUC, Recall, Accuracy의 총 3개 변인으로, 예측력과 더불어 Recall을 계산한 이유는 자살 사고를 예측하는 데 있어 가장 치명적인 실수는 자살 사고가 있는데도 없다고 판단한 경우라고 여겼기 때문입니다. 모델링 순서에 따른 결과는 아래와 같습니다. <br/>
<br>
<1.> 벤치마킹 모델 (CBCL) <br/>
    1) AUC: 0.94 <br/>
    2) Accuracy: 0.94 <br/>
    3) Sensitivity(Recall): 0.81 <br/>
    <br>
<2.> Psychosocial 모델 (CBCL+Demographic info.) <br/>
    1) AUC: 0.96 <br/>
    2) Accuracy: 0.95 <br/>
    3) Sensitivity(Recall): 0.81 <br/>
    <br>
<3.> Psychosocial + PRS 모델 <br/>
    1) AUC: 0.97 <br/>
    2) Accuracy: 0.94 <br/>
    3) Sensitivity(Recall): 0.82 <br/>
    <br>
<4.> Psychosocial + sMRI 모델 <br/>
    1) AUC: 0.96 <br/>
    2) Accuracy: 0.94 <br/>
    3) Sensitivity(Recall): 0.83 <br/>
    <br>

정리하자면, Psychosocial 모델은 AUC가 0.96, accuracy가 0.94, sensitivity(recall)가 0.81로 다른 수치는 유사하지만 벤치마킹 모델에 비해 AUC가 소폭 상승한 결과를 보였습니다. <br/>

이후 Psychosocial + PRS 모델은 AUC가 무려 0.97까지 상승하는 결과를 보여주었는데, 다른 결과 또한 다른 모델이 비해 뒤처지지 않았습니다. <br/>

Psychosocial + sMRI 모델은 벤치마킹 모델보다는 성능이 좋았으나 AUC가 Psychosocial + PRS 모델보다는 낮았습니다. 물론 sensitivity(recall)가 Psychosocial + PRS 모델보다 0.01 가량 높기는 하였으나, 아이들의 sMRI 데이터를 수집하는 비용이 PRS 검사 비용보다 높다는 것을 고려하자면 0.01만큼의 차이로 인해 자살 사고를 설명하는 데 Psychosocial + sMRI 모델을 선정하는 것은 비효율적이라고 판단하였습니다. <br/>
<br>
<h2> 논의 </h2>
저희가 비교한 모델들은 공통적으로 Psychosocial 데이터를 포함하고 있었습니다. Psychosocial+PRS 모델이 가장 높은 성능을 보여주기는 하였으나 Psychosocial 모델과 비교하였을 때 독보적으로 유의미한 결과라고 보기는 어려웠던 만큼, 모든 데이터의 리더모델에서도 Psychosocial 데이터의 중요도를 무시할 수 없었습니다. 때문에 저희는 해결책을 고안할 때도 Psychosocial한 변인을 조절하기 위한 방책을 우선적으로 고려하였습니다. 특히나 앞선 데이터 분석에서 모델 대부분에서 Important Variable로 등장한 정신 장애 양상 등을 완화하기 위한 해결책 위주를 고안하였습니다. 이때 Important Variable은 대개 리커트 척도에 따라 보고된 정신 장애 행동 양상 점수가 높을수록 자살 사고와의 연관성이 높은 것으로 나타났습니다. 관련된 그래프의 예시는 [Link3](drive.google.com/drive/folders/1YUouEZNnurE9P2sA8PDklUeUbKYBoQ1V?usp=sharing)에서 확인하실 수 있습니다.  <br/>

덧붙여 PRS의 경우 현존하는 데이터는 유럽 계통의 유전자 검사에 가장 유의미하다고 보고한 선행 연구가 있었어서, PRS를 곧바로 자살 사고 예측 및 해결에 적용하기보다는 한국형 데이터 수집에 초점을 맞추고자 하였습니다. <br/>

<br/>

<h2> 해결책 및 예산안 제안 </h2>
저희는 위험군에 대한 선별적 개입과 보편적 예방을 동시에 하는 것을 제1목적으로 두었습니다. 세부 목표로는 자살 사고를 정확하게 예측하고 효과적으로 예방할 것, 개인/가정/학교를 아울러 접근성이 좋을 것, 현실적으로 가능할 것, 나아가 Psychosocial 데이터에 추가하여 PRS를 합한 모델이 성능이 좋았던 것을 토대로 한국형 Bio-bank의 초석이 될 것을 지정하였습니다. <br/>

이에 따라, 저희가 목표로 한 대상은 초등학교 재학 중인 아동 및 부모님으로, 아동 대상의 자살 사고 예방 해결책은 향후 아동이 청소년이 되었을 때 자살 시도의 낮은 빈도로 이어질 것이라고 기대하였기 때문입니다. 조사한 바, 전국 초등학교의 수는 6000개 안팎, 초등학생 수는 264만 명 안팎이었습니다. 현 예산의 경우, 보건복지부에서 지정한 2020/2021 ‘아동/청소년 정신건강 증진사업’을 위한 예산은 약 44억원이었으므로 이를 기준으로 해결책을 고안하였습니다. <br/>
<br>
Treatment는 개인/가정/학교의 3개 영역으로 구분하였습니다. 먼저, 개인 차원의 해결책으로는 자연 자극으로 Trait를 조절하기 위하여 '게임 기반 디지털 치료제 개발'을 고려해보았습니다. 간단히 설명하자면 모델의 주요 변수인 대다수의 정신 장애에 공통적으로 작용하는 강한 충동성 등과 같은 특질을, 게임을 통해 훈련하는 것입니다. 이처럼 딱딱하지 않고 말랑말랑한 접근법을 선정한 까닭은 초등학생을 대상으로 한 treatment이므로 흥미를 유도할 수 없다면 효과가 반감될 것이라고 예상했기 때문입니다. <br/>

이처럼 정신 장애에 영향을 미치는 특질을 게임 기반 서비스로 완화하고자 하는 시도로는 ADHD의 주의력 부족을 치료하기 위한 Endeavor RX와 같은 선례도 존재합니다. 이처럼 여타 정신 장애에 유의미한 특질을 조절하는 서비스 개발을 위해 필요한 구성은 크게 연구와 홍보의 두 갈래로, 전자는 서울대 내 심리학과/뇌인지과학부/정보문화학 연계전공/컴퓨터공학부 등 다양한 학문적 배경이 존재하는 project team을 꾸릴 것을 제안합니다. 이후 만들어진 서비스를 홍보하고 모니터링을 하는 일까지는 연구와 합해 약 3억 원이 소요될 것으로 예상합니다. <br/>

다음으로, 가정 차원의 해결책입니다. 대다수 모델에서 CBCL의 데이터가 Important Variable로 도출되었던 점에 착안한 것으로, 부모가 자녀에 대해 CBCL의 답을 보고할 수 있다는 것은 최소한 부모가 자녀에게 그만큼의 관심을 가지고 있다는 사실을 의미합니다. 그에 따라 저희는 부모와 아이 간의 유대를 강화하면서도 자살 사고를 완화할 방법을 모색하였는데, 그것이 바로 'PCIT'입니다. PCIT는 부모까지 포함하는 치료 방법으로 가족력이 있는 아이들을 도울 수 있어, K-CBCL이 높게 나온 아이들의 가정을 대상으로 PCIT 교육을 필수로 제공하는 것을 제안합니다. K-CBCL의 장점 중 하나는 비교적 적은 비용이 소모된다는 것으로, 검사 비용과 교육 비용까지 합해 약 14억 원을 책정하였니다. <br/>

마지막으로, 학교 차원에서는 'Mindfulness(마음챙김)'를 공식 수업 과정으로 지정하기를 제안합니다. 명상을 통한 보호요인의 개발을 지향하는 것인데, 이는 여러 정신 장애의 원인으로 작용하는 neuroticism을 감소시키고 default mode network를 높여 보호요인으로 작용할 수 있습니다. 이처럼 state mindfulness를 증가하는 훈련은, trait mindfulness를 증가하는 일로도 이어지기까지 합니다. 이에 대해, 모든 학교에 도입하기는 어려우므로 전국의 5% 시범 초등학교에 대해 명상 영상 콘텐츠를 제작하고 강사 인건비까지 합하여 약 10억 원이 소모될 것으로 추정됩니다. <br/>
<br>
예방책으로 넘어가서, 개인적 차원에서는 'Mental Health Care App for Child'가 적합한 방안으로 생각됩니다. 종합적인 정신 건강 보호 앱의 개발을 통해 daily self-report, 위급상황 알림 등을 사용할 수 있도록 유도하고, 데이터를 기반으로 하는 personalized treatment 제공 또한 기대할 수 있습니다. 이 데이터를 바탕으로 고위험군은 지역별 정신건강사회복지사와 사례관리 및 전문적인 치료를 받을 수 있도록 연계한다면 유의미한 효과를 창출할 것입니다. APP 제작부터 사례 관리 등까지 포함해, 약 25억 원이 예상됩니다. <br/>

사회적인 차원에서는 'MHC Data Center'를 운영하는 일을 고려해볼 수 있습니다. 즉, 한국형 ABCD 데이터를 구축하는 것입니다. 앞선 여러 해결책들의 결과를 관리하는 동시에, 앞서 언급했듯 자살 사고 예측에 성능 증가를 가져왔던 PRS 또한 한국형으로 새로이 수집될 필요가 있습니다. 이 과정에서 고위험군에 대해서는 fMRI 등 추가로 뇌 이미징 영상을 촬영할 수 있을 것이고, DTx 결과부터 APP Data 분석, PRS 검사까지 합하여 약 3억 원의 예산이 예상됩니다. <br/>
<br>
<h2> 결론 </h2>
자살 사고에 대한 이와 같은 다각도의 접근은 개인적인 차원에서 치료 및 예방의 접근성을 높이는 효과부터, 가정이나 학교 차원의 도움으로 타인과의 건강한 의사소통 방식과 애착 관계를 배우고, 회복 탄력성을 높이는 등 개인의 정신적 성장까지도 기대할 수 있습니다. 과학적 검증을 바탕으로 하는 Mental Health Care APP이나 MHC Data Center 등의 경우 데이터에 기반해 위기아동을 이원분류가 아닌 다층 분류를 하는 동시에, 개입까지 시도하고, 앞으로 치료 대상 확대를 위한 기반을 마련한다는 의의가 있습니다. <br>
요약하자면, 사회가 나서서 아이들이 자살 사고에 취약한 특질을 조절할 수 있게끔 돕고, 객관적인 증거를 바탕으로 시범 학교 아동들의 자살 사고률을 유의미하게 낮출 것을 기대합니다. 이처럼 우리의 미래르 이끌어나갈 아이들을 위해, 그들이 행복한 어른이 될 수 있도록 고민하고 노력하는 것은 현재 어른인 우리들의 몫일 것입니다. <br/>
