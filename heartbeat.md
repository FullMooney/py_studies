---
layout: default
title: heartbeat Classification
description: heartbeat classification 을 통한 심장질환 예측 
---


<p align="center">
    <img src="" />
</p>


## 제공사항

1. 분류결과가 포함된 트레이닝 데이터 00000건 - excel형식
1. 분류결과가 포함되지 않은 테스트 데이터 0000건
1. 테스트 데이터를 예측하여 업로드하였을때 10%의 샘플링에 대한 정확도 - kaggle 활용


<p align="center">
    <img src="" />
</p>

###### 목표

주어진 feature와 분류라벨을 활용하여 가장 정확도가 높은 heartbeat 분류모델을 만드는 것


### 1차 ML 을 통한 분류 시도 

```python

model...

```

몇 줄 안되는 소스로 구현하였음에도 97%라는 상당히 높은 정확도를 보임. (트레이닝/테스트 스플릿 20%)
이러다 99% 찍는것 아닌지 행복한 상상을 시작함

### 2차 CNN 을 통한 분류 시도 - LeNet

```python

model...

```

feature중에 의미가 없을 것이라 생각한 데이터를 일부 제외하고 12 X 12 형태의 이미지로 데이터를 변경 (트레이닝/테스트 스플릿 20%)
우여곡절이 있었으나 98% 정확도에 도달
이제는 곧 완전한 모델이 나올것이라는 기대를 갖게됨

### 3차 CNN 을 통한 분류 시도 - LeNet

```python

model...

```

제외하였던 데이터를 포함하고 부족한 부분은 zero-padding하여 13 X 13 형태의 이미지로 데이터를 변경 (트레이닝/테스트 스플릿 20%)
여전히 97~98%를 벗어나지 못함
뭔가 심상치 않아지기 시작


### 4차 CNN 을 통한 분류 시도 - LeNet

```python

model...

```

이미지 사이즈는 다시 12 X 12로 복귀. 별 차이를 못느꼈기 때문이었음 (트레이닝/테스트 스플릿 20%)

1. 트레이닝 시킬때 배치 사이즈 조절 100 -> 30
1. optimizer 변경 적용 시작  (loss function: categorical_crossentropy)
   optimizer:
   - rmsprop
   - Adam
   - Adhelta
   
### 5차 CNN 을 통한 분류 시도 - LeNet

```python

model...


```
1. loss function 변경: binary_crossentropy  
   optimizer:
   - SGD

갑작스런 99%를 로컬에서 경험하였으나, 실제 적용시 97%에도 못미치는 정확도를 보임

### 6차 CNN 을 통한 분류 시도 - LeNet 

```python

model...

```
테스트 데이터들간에 비율이 다른것을 확인
가중치를 부여하기 시작함=> 큰차이는 나지 않음

### 7차 CNN 을 통한 분류 시도 - LeNet 

```python

model...

```
validation 데이터를 각 100건씩으로 동일수로 추출하고 
나머지 데이터들의 비율 고려하여 가중치 재 적용 => 오히려 학습률이 감소

### 8차 CNN 을 통한 분류 시도 - Vgg16 , GoogleNet

```python

model...


```

네트워크 모델을 변경하여 가며 테스트 시작=> 여전히 변화를 보이지 않음. 정확도는 95~98%까지 떨어짐

### 9차 ML을 통한 분류 시도

```python

model...


```

Randomforest 등 ML 로 분류시도 = > 98은 넘어서지 못함


그러다 어느새 종료..


#### 패착

1. 데이터를 임의로 제거하고 12 x 12로 테스트 한점
1. validation 데이터를 너무 많이 잡은것
1. 네트워크를 너무 복잡하게 설계한것




[실습git](https://github.com/FullMooney/py_studies/blob/master/)



[메인으로 돌아가기](./)
