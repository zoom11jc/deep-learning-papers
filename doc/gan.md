
[//]: # (Image References)

[image1]: ./images/gan/cgan_loss2.png "cgan_loss2.png"
[image2]: ./images/gan/cgan_structure.png "cgan_structure.png"
[image3]: ./images/gan/cgan_generator.png "cgan_generator.png"
[image4]: ./images/gan/cgan_discriminator.png "cgan_discriminator.png"
[image5]: ./images/gan/cgan_mnist.png "cgan_mnist.png"
[image6]: ./images/gan/cgan_txt2img.png "cgan_txt2img.png"
[image7]: ./images/gan/cgan_pix2pix.png "cgan_pix2pix.png"

# GAN

## GAN
최윤제님 정리 자료
 - 원본: https://www.youtube.com/watch?v=uQT464Ms6y8
 - 네이버 버전: https://www.youtube.com/watch?v=odpjk7_tGY0
 - 슬라이드: https://www.slideshare.net/NaverEngineering/1-gangenerative-adversarial-network

AtoZ: 
 - http://nbviewer.jupyter.org/github/metamath1/ml-simple-works/blob/master/GAN/GANs.ipynb

GAN tutorial 
 - 2016 한글 정리: https://kakalabblog.wordpress.com/2017/07/27/gan-tutorial-2016/
 - 2017: https://nips.cc/Conferences/2016/Schedule?showEvent=6202

블로그
 - 컨셉과 원리: http://learnai.tistory.com/ 
 - 구현: http://jaynewho.com/post/2 
 - 라온피플 자료: http://laonple.blog.me/221190581073


## DCGAN
DCGAN 논문 리뷰 한글: 
 - http://laonple.blog.me/221201915691
 - http://artoria.us/19
 - https://jaejunyoo.blogspot.com/2017/02/deep-convolutional-gan-dcgan-1.html
 - 소개 동영상: https://www.youtube.com/watch?v=7btUjE2y4NA
 - https://kakalabblog.wordpress.com/2017/06/04/unsupervised-representation-learning-with-dcgan-2016-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0/
 - 독창적인 소개: https://dev-strender.github.io/articles/2017-07/decan-introduction

참조하기 좋은 자료들
 - http://www.khshim.com/archives/218
 - https://kakalabblog.wordpress.com/2017/06/10/gandcgan-%EB%A6%AC%EB%B7%B0-%EB%B0%9C%ED%91%9C/


## cGAN

### Introduction
- GAN
  - 주목받는 이유
    - 까다로운 확률 계산을 approximating하는 것은 어려운데, 이것을 회피할 수 있는 generative model 학습 framework가 대안으로 뜨고 있다. 
  - 장점
    - Markov chain도 필요없고
    - gradient를 얻기 위해 backprop만 사용되고
    - 학습 과정에서 inference도 필요없고
    - 다양한 factor와 interation을 모델에 쉽게 포함시킬 수 있고
    - S.O.T.A log-likelihood estimate와 진짜같은 sample을 만들 수 있다.
- cGAN의 특징
  - 기존 GAN과는 달리 추가적인 정보를 사용하여 data generation process를 제어하는 것이 가능함
  - 이러한 conditioning은 아래와 같은 것들이 될 수 있다. 
    - class label
    - some part of data for inpainting ([inpainting](https://www.slideshare.net/PulkitGoyal1/image-inpainting), [참고논문 Fig. 3](https://papers.nips.cc/paper/5024-multi-prediction-deep-boltzmann-machines.pdf))
    - data from different modality
- 이 논문에서는 두 가지 데이터셋으로 실험을 진행했다. 
  - MNIST: condition이 class label
  - MIR Flickr: condition이 multi-modal 정보

### Related Work

기존 supervised neural networks의 2가지 challenge
1. 아주 많은 수의 output category를 예측하는 모델로의 확장이 어렵다.
    - 해결책: 다른 modality의 정보를 활용한다. ([참고논문 Fig. 1](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/41869.pdf)) 
    - => prediction error라도 ground truth에 가까울 수 있고(e.g. table 대신 chair를 예측), 
    - => 학습 과정에서 unseen인 label에까지 generalized prediction이 가능하다
2. input에서 output으로의 매핑이 one-to-one mapping을 학습하는데 많은 연구가 집중되어 있지만, 많은 실제 문제들은 probabilistic one-to-many mapping이다. (e.g. 이미지 하나에 대해 다양한 태깅이 존재할 수 있음)
    - 해결책: conditional probabilistic generative model을 사용한다.
    - => multi-modal Deep Boltzmann Machine을 학습하는 방법 (cGAN 방법론과 유사한 접근법)
    - => multi-modal neural language model 학습해서 이미지에 대한 다양한(= one-to-many) description을 생성

### Conditional Adversarial Nets
generator와 discriminator에 추가적인 정보 y로 condition을 주면 GAN을 conditional model로 확장할 수 있다. 

- Generator: input noise p(z)와 y가 joint hidden representation으로 결합된다. (참고로 adversarial training framework는 hidden representation을 구성하는 방식에 상당한 유연성을 제공한다. 반면 전통적인 generative framework에서는 이게 엄청 어려웠음)
- Discriminator: x와 y가 입력으로 discriminator function에 제공된다. 
- Objective function:  
![alt text][image1]
- Structure:  
![alt text][image2]

### Experimental Results
#### 1) Unimodal (MNIST)
* conditional adversarial net on MNIST images: 
  * (one-hot vector로 인코딩된) class labels로 conditioned된 MNIST 이미지로 conditional adversarial net을 학습함.
* Generator 구현부 설명
  - 100차원 unit hypercube에서 uniform distribution으로부터 z 추출함. 
  - z와 y는 각각 size 200, 1000 짜리 hidden layer(w/ ReLU)로 매핑됨. 그러고나서 양쪽 모두 size 1200짜리 두 번째 combined hidden ReLU 레이어로 매핑됨.
  - 마지막으로 784 차원의 MNIST 샘플을 output으로 생성하는 sigmoid unit layer가 있음.
  - ![alt text][image3]
* Discriminator 구현부 설명
  - x, y를 maxout layer(각각 240/5, 50/5)로 매핑한다. 
  - 두 hidden layer는 joint maxout layer(240/4)로 매핑된 후 sigmoid layer로 들어간다. 
  - discriminator의 아키텍쳐는 충분한 capacity만 있다면 별로 중요하지 않고, maxout이 이 task에 궁합이 좋다.
  - ![alt text][image4]
  - mini-batch size: 100
  - learning rate: 0.1 -> 0.000001
  - momentum: 0.5 -> 0.7
  - dropout keepprob: 0.5
  - stopping point: best estimate of log-likelihood on the validation set
* 성능 평가 결과
  - ![alt text][image5]
  - Gaussian Parzen window log-likelihood estimate로 성능 평가
  - Parzen window distribution을 사용해서 test set의 log-likelihood를 추정했다.([참고 논문 5장](https://github.com/bt22dr/deep-learning-papers/blob/master/paper/Generative%20Adversarial%20Networks.pdf))
    - 막상 결과를 수치로 따져보면 다른 모델이 좀 더 낫다. 근데 우리 모델은 그냥 PoC라서 그런것이고 좀 더 고도화하면 나아질 것이다. 

#### 2) Multimodal (Flickr)
* UGM(user-generated metadata)
  * Flickr같은 사진 사이트에는 labeled data가 많다. (e.g. 사진에 연관된 사용자 태그들)
  * UGM은 canonical image labelling schems랑은 많이 달라서 좀 더 descriptive하다
  * UGM은 사람마다 다른 단어를 이용해서 같은 컨셉을 기술하기 때문에 동의어가 많다. 고로 이런 labels를 잘 normalize하는 방법이 중요하다. => word embedding을 사용한다. 
* 자동화된 이미지 태깅
  * 이미지 feature로 condition된 tag-vector의 분포를 생성하기 위해 conditional adversarial nets를 사용
  * feature
    * image representation: ImageNet으로 pre-train된 network에서 마지막 FC layer의 4096 차원 output을 image representation으로 사용
    * word representation: YFCC100M으로 skip-gram model을 학습하여 word representation으로 사용
  * 샘플 생성 방법:
    1. image feature vector를 condition으로 사용하여 word feature vector를 생성한다. 
    2. 해당 word representation과 cosine 유사도가 높은 word들을 선별
    3. 개중에 10개의 most common words를 선택해서 evaluation에 사용함.
 * 자세한 내용은 논문을 참조

### 연관된 연구들 ([참고 이미지](https://github.com/bt22dr/deep-learning-papers/blob/master/paper/Conditional%20Generative%20Adversarial%20Nets.pdf))
cGAN은 아래 연구들로 발전된다.
- generative adversarial text to image synthesis  
![alt text][image6]
- pix2pix: Image-to-Image Translation with Conditional Adversarial Networks  
![alt text][image7]

### 같이 보면 좋은 자료 
- 김승일 소장님 발표: https://www.youtube.com/watch?v=iCgT8G4PkqI
- cgan 정리 블로그:
  - http://t-lab.tistory.com/29
  - https://kangbk0120.github.io/articles/2017-08/conditional-gan







![](https://latex.codecogs.com/svg.latex?y%3Dx%5E2)
- test
  - testtest1 ![](http://latex.codecogs.com/gif.latex?%5Cint%20p_%5Ctheta%20%28z%29%20p_%5Ctheta%28x%7Cz%29) test
  - testtest2 ![](http://latex.codecogs.com/svg.latex?%5Csmall%5Cint%20p_%5Ctheta%20%28z%29%20p_%5Ctheta%28x%7Cz%29) test
  - testtest3 ![](http://latex.codecogs.com/svg.latex?%5Ctiny%5Cint%20p_%5Ctheta%20%28z%29%20p_%5Ctheta%28x%7Cz%29) test
  - testtest4 ![](http://latex.codecogs.com/svg.latex?%5Csmall%20%5Cint%20p_%5Ctheta%28z%29p_%5Ctheta%28x%7Cz%29) test
  - testtest5 ![](http://latex.codecogs.com/svg.latex?%5Ctiny%20%5Cint%20p_%5Ctheta%28z%29p_%5Ctheta%28x%7Cz%29) test
