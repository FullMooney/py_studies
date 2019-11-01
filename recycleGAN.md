---
layout: default
title: RecycleGAN
description: RecycleGAN 논문을 개인적으로 정리해 보겠습니다
---

** 논문 Recycle-GAN: Unsupervised Video Retargeting **

[Paper/Code LINK](http://www.cs.cmu.edu/~aayushb/Recycle-GAN/).

CycleGAN에서 예상보다 성능이 나오지않아 자료를 더 조사하던중 '18년도 8월에 나온 아직 따끈한 recycleGAN에 대해 알게되었습니다.
아래 이미지에서 볼 수 있듯이 비디오 입력에 대해서 cycleGAN 보다 우수한 성능을 보여주는 것 같은데 
얼마나 어려운지, 학습시간은 어느정도 걸릴지는 해보면서 더 정리하겠습니다.

<p align="center">
    <img src="http://5b0988e595225.cdn.sohucs.com/images/20180912/6edfe8819f7b467f8060b9e83fc2e031.gif" />
</p>


## 논문 분석

recycleGan은 타겟 도메인의 스타일을 유지하면서 하나의 도메인의 순서가 있는 컨텐츠를 타겟 도메인으로 전송하는 비지도 data-driven video retargeting 접근방법이다. 이를 통해 사람의 움직임과 얼굴을 다른 사람에게 전송하거나 로봇에게 사람의 동작을 따라하게 하거나, 흑백비디오를 컬러로 만들수 있다. 
cyclegan과 다른점은 sequential 한 데이터를 트레이닝에 이용하므로써 (cyclegan에서는 randome하게 학습시킴) prediction에 대한 loss까지 고려가 되었다는 부분으로 보인다



[메인으로 돌아가기](./)
