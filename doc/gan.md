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

Introduction
- GAN
 - 주목받는 이유
  - 까다로운 확률 계산을 approximating하는 것은 어려운데, 이것을 회피할 수 있는 generative model 학습 framework로 떠오르고 있다. 
 - 장점
  - Markov chain도 필요없고
  - gradient를 얻기 위해 backprop만 사용되고
  - 학습 과정에서 inference도 필요없고
  - 다양한 factor와 interation을 모델에 쉽게 포함시킬 수 있고
  - S.O.T.A log-likelihood estimate와 진짜같은 sample을 만들 수 있다.  

- 김승일 소장님 발표: https://www.youtube.com/watch?v=iCgT8G4PkqI
- cgan 정리 블로그:
  - http://t-lab.tistory.com/29
  - https://kangbk0120.github.io/articles/2017-08/conditional-gan
