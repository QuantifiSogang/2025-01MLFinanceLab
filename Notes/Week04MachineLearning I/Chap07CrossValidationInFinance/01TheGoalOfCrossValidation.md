## The Goal of Cross Validation

Machine Learning의 목적 중 하나는 데이터의 일반 구조를 학습해 미래의 관측되지 않은 특성에 대해 예측하느 것이다.
Machine Learning Algorithm 학습에 사용한 것과 동일한 데이터세으로 테스트하면 대체로 환상적인 결과를 얻는다.
Machine Learning Algorithm이 이런 식으로 오용되면 파일 손실 압축 알고리즘과 차이가 없다.
이들은 놀라운 충실도로 데이터를 요약하지만, 예측력인 전혀 없는 것과 마찬가지이다.
Cross Validation은 IID Process를 통해 추출한 관측 데이터를 두 가지 집합으로 나눈다.
하나는 훈련셋, 나머지 하나는 테스트셋이다. 전체 데이터셋의 각 관측값은 오직 하나의 집합에만 소속된다. 
이는 하나의 집합에 있는 정보가 다른 집합으로 흘러 들어가는 것을 방지하는데, 그렇지 않다면 미지의 데이터로 테스트하는 목적이 무의미해진다.

여러 가지 다양한 Cross Validation이 가능한데, 그 중 가장 유명한 것은 K-Fold Cross Validation이다. 아래의 그림은 Cross Validation이 수행하는 $k$개의 train/test 분할을 보여 준다.
여기서 $k=5$로 보여준다. 이 방법은 다음과 같이 진행된다.

1. 데이터셋은 $k$개의 부분 집합으로 분할된다.
2. $i=1,\dots,k$에 대해 Machine Learning Algorithm이 $i$를 제외한 모든 부분 집합에 대해 학습되며, 적합화된 Machine Learning Algorithm은 $i$에 테스트한다.

K-Fold Cross Validation의 결과는 $k \times 1$ 배열의 Cross Validation 성과 척도이다.
예를 들어, 이진 분류기의 경우 Cross Validation의 정확도가 $\frac{1}{2}$을 넘으면 모델이 무엇인가 배운 것으로 간주한다.
$\frac{1}{2}$의 정확도란 공평한 동전을 던지면 얻을 수 있는 확률이기 때문이다.

![cross_validation.png](images%2Fcross_validation.png)

금융에 있어서 Cross Validation은 두 가지 상황에서 주로 활용된다.
하나는 모델의 개발이고, 다른 하나는 Backtesting이다.
Backtesting는 별도의 주제로 심도 있게 다뤄지는 복잡한 주제이다. 이번 장에서는 Cross Validation에 집중하기로 한다.

## Why K-Fold Cross Validation fails in finance

지금까지 아마도 Machine Learning Algorithm이 금융에도 잘 작동하는 근거로서 K-Fold Cross Validation을 설명하는 몇 가지 논문을 보았을 가능성이 높다.
불행히도 이 결과는 잘못됐을 가능성이 크다.
K-Fold Cross Validation이 금융에서 실패하는 원인은 다음과 같다.

1. 관측값을 IID로 추출하는 가정을 할 수 없다
2. 테스트셋이 모델 개발 과정 프로세스에 여러 번 사용된다. 이는 다중 테스트와 선택 편향을 초래한다.

정보 누출은 훈련 데이터셋이 테스트 데이터셋에도 등장하는 정보를 포함하는 경우에 발생한다. Auto Correlation이 존재하는 Feature $X$가 중첩된 데이터에서 형성된 label $Y$와 관계가 있다고 가정해 보자.

- Auto Correlation 떄문에 $X_t \approx X_{t+1}$이다
- label이 중첩된 데이터 포인터에서 유도되었으므로 $Y_t \approx Y_{t+1}$이다

training set과 test set을 $t$와 $t+1$로 분리하면 정보가 누출된다. 분류기가 우선 $X_t, Y_t$를 학습한 후 관측된 $X_{t+1}$에 대해 $E[Y_{t+1} | X_{t+1}]$을 예측한다. 이 분류기는 $X$가 상관없는 특성이라 하더라도, $Y_{t+1}=E[Y_{t+1}|X_{t+1}]$을 달성할 것이다.

$X$가 예측 변수일 경우 정보누출은 이미 잘 작동하는 전략의 성능을 좀 더 향상시킨다. 이 문제는 잘못된 발견을 야기하는 상관없는 특성의 존재에 따른 정보 누출이다. 이러한 정보 누출을 줄이는 두 가지 이상의 방법이 있다.

1. $Y_i$가 $Y_j$를 결정하는 데 사용된 정보의 함수이고, $j$가 테스트셋에 속하면 모든 관측값 $i$를 훈련 데이터셋에서 제거한다. 예를 들어, $Y_i$와 $Y_j$는 중첩된 기간이 없어야 한다.
2. 분류기 과적합을 피해야 한다. 이를 통해 약간의 누수가 발생하더라도 분류기는 이로부터 혜택을 받지 않는다. 다음 방법을 이용하자. 

    a. `base_estimator`의 조기 종료(Early Stopping) 
    
    b. 중복된 예제에서 발생하는 과표본 문제를 통제하면서 분류기를 bagging하여 개별 분류기가 최대한 다양해질 수 있도록 한다. 이를 위해 `max_samples`를 평균 고유도로 설정하고, Sequential Bootstrapping을 적용한다.

중첩된 정보에서 형성된 $X_i$와 $X_j$를 생각해 보자. $i$는 훈련 데이터셋, $j$는 테스트 데이터셋에 속한다. 이 경우, 정보 누출에 해당하는가? $Y_i$가 $Y_j$와 서로 독립적이라면 반드시 그렇지는 않다.
누출이 발생하려면 $(X_i, Y_i) \approx (X_j, Y_j)$여야 하며, $X_i \approx X_j$나 $Y_i \approx Y_j$만으로는 충분하지 않다.