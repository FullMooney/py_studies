---
layout: default
title: Another page
description: This is just another page
---


# DiscoGAN 

** 논문 Learning to Discover Cross-Domain Relations with Generative Adversarial Networks **

[LINK](https://arxiv.org/pdf/1703.05192.pdf).

## 논문 요약 

다른 두 도메인에서 관계를 찾는 것은 사람에게는 자연스러운 일이다. 예를 들어 영어와 이를 번역한 프랑스어의 관계를 인식하거나 바지나 신발과 같은 스타일의 수트 자켓을 고르기도 한다. 
과연 이러한 유사성에 대해 학습하는 능력을 가질 수 있을까의 문제는 특정 조건의 이미지의 생성 문제로 바꿔말할 수 있다. 또 바꿔 말하면 하나의 도메인과 다른 도메인의 매핑을 위한 함수를 찾는 문제로 생각할 수 있다.

GAN(Generative Adversarial network, 생성적 적대 신경망)의 최근 트레이닝 방식은 대부분 쌍을 이루는 명시적인 데이터를 사람이나 다른 알고리즘을 제공하는 방식으로 접근되고 있다.
이러한 명시적인 데이터는 라벨링 하는데에 많은 노동력을 필요로 할 수 있고 하나의 이미지안에 많은 베스트 후보군이 있어서 작업이 어려울 수 있다.

discoGAN은 두 가지 시각적 도메인이 명시적인 데이터 없이 관계를 발견해 낼 수 있도록 하고자 했다.

2014년 Goodfellow 의 Standard GAN 에서는 랜덤한 가우시안 노이즈 z 를 hidden features h 에 인코딩하고 MNIST와 같은 숫자 이미지를 생성하였으나 DiscoGAN에서는 노이즈 대신 이미지를 인풋값으로 사용하였다. 
그리고 기존에는 도메인A에서 B로의 매핑만 배울수 있는 구조였기 때문에 하나의 제너레이터를 더 추가하였다. 

![Branching](./images/discoGAN.PNG)




```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```




_yay_

[back](./)
