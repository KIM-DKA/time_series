#!/usr/bin/env python
# coding: utf-8

# In[1]:


############################################################################# 
# PART1 
#############################################################################
#############################################################################
# Xgboost, RandomForest, LTSM, prophet, auto_arima 등 다양한 모델링 방법으로 접근했으나
# 서비스별, 지역별 세분화 후 모델 생성시 시계열적인 패턴이 잘 파악되지 않는 한계점이 있었음
# 특히,Y, X에 lag변수를 활용하는 등 추가 파생변수 생성 및 많은 변수들을 활용했으나 기존 X 변수들간 다중공선성이 커서
# 실질적으로 모델을 만드는데 성능이 떨어지는 아쉬움 존재
# 다양한 Machine Learning, Deeplearning 방법으로 접근하였으나
# 과거 데이터가 적고, 데이터가 simple하여 오히려 정확도가 떨어지는 것으로 파악되어
# PART2에서 통계적인 모델로 접근 방법 변경함
# 예선 진행 중반부에 통계적인 모델로 뱡향을 변경하여 다양한 시도를 하지 못한 점이 아쉬움 
#############################################################################


# In[2]:


############################################################################## 
## PART2 
##############################################################################
## - PART1에서 진행한 다양한 방법론들에 대해서 한계점이 발견되어
## - 지역별 세분화 후 통계적인 모델로 접근하여 
##   seasonality가 명확한 autoregressive한 모델을 생성하는 방법론 활용
## - 특히, 지역별 sum값을 기반으로 EDA 및 Time Series Decomposition 진행하니 요일간 1주일 단위로 
##   seasonality가 두드러짐. 요일별 특징 뚜렷함을 발견.
##   (서비스별, 지역별 sum 값을 기반으로는 주기성을 파악하기 어려웠음)
## - 실질적으로 rmse기반 모델 성능 비교시 PART2 접근방법이 더 정확도가 높아
##   PART2 접근방법으로 모델링 진행


# In[3]:


# AIDU 내부 연동을 위한 라이브러리
from aicentro.session import Session
from aicentro.framework.keras import Keras as AiduFrm

# AIDU와 연동을 위한 변수
aidu_session = Session(verify=False)
aidu_framework = AiduFrm(session=aidu_session)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import datetime


# In[4]:


##############################################################################
# 1) 데이터 전처리 (EDA)
##############################################################################
# 데이터 로딩
df0 = pd.read_csv(aidu_framework.config.data_dir + '/rev_episode2_train_new.csv')
df0.head()

# 날짜 컬럼 수정 
df0['base_date'] = pd.to_datetime(df0['base_date'].astype(str), format='%Y-%m-%d') #기준일을 string -> datetime 포맷 변환 
df0['base_day'] =  [d.weekday() for d in df0['base_date']] #요일 컬럼 생성 (월=0, 화=1, ... , 일=6)

# 지역/기준일별로 패킷량 합침 
df1 = df0.groupby(['gun_gu_nm','base_date','base_day'])['sum_packt_trm_qnt'].agg('sum').reset_index(name ='y')
df1


# In[5]:


# (주말 - 주중) 패킷량 순으로 지역 나열 
df_diff = pd.DataFrame(columns=['gun_gu_nm','diff'])
df_diff['gun_gu_nm'] = df1.gun_gu_nm.unique()

group = df1.gun_gu_nm.unique()
for g in group:
    t = df1[df1.gun_gu_nm==g]
    diff = np.mean(t.loc[t.base_day.isin([3,4]),'y']) - np.mean(t.loc[t.base_day.isin([0,1,2,5,6]),'y'])
    df_diff.loc[df_diff.gun_gu_nm==g, 'diff'] =  diff
    
sort_group = df_diff.sort_values('diff', ascending=False)['gun_gu_nm'].tolist()


# In[6]:


# 전체 지역에 대한 시계열 그래프
# 한 그래프에 10개 지역씩 그림
plt.figure(figsize=(25,20))
for i in range(1, 26):
    plt.subplot(5, 5, i)
    for k in range(10):
        ys = df1.loc[df1.gun_gu_nm==sort_group[k+10*(i-1)]]['y']
        ys = (ys-np.mean(ys))/np.std(ys) # 지역별  패킷량 표준화 
        plt.plot(range(21),ys)
        plt.axvline(x=0.5, color='r')
        plt.axvline(x=7.5, color='r')
        plt.axvline(x=14.5, color='r')


# In[7]:


##############################################################################
# 2) 모델링
##############################################################################
# AIDU 내부 연동을 위한 라이브러리
from aicentro.session import Session
from aicentro.framework.keras import Keras as AiduFrm

# AIDU와 연동을 위한 변수
aidu_session = Session(verify=False)
aidu_framework = AiduFrm(session=aidu_session)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# ### Time Series Decomposition(시계열 분해)

# In[23]:


# Time Series Decomposition(시계열 분해)
# 시계열 분해 후 요일별 계절성이 명확하게 확인
# 지역/기준일별로 패킷량 합침 
df1 = df0.groupby(['gun_gu_nm','base_date'])['sum_packt_trm_qnt'].agg('sum').reset_index(name ='y')
df1.set_index('base_date',inplace=True)
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().system('pip install statsmodels')
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[24]:


# 지역별로 seasonality 확인 가능
series =df1['y'][df1['gun_gu_nm']=='G001'] 
df = pd.DataFrame({'timeseries': series})


# In[25]:


from statsmodels.tsa.seasonal import seasonal_decompose
ts = df.timeseries
result = seasonal_decompose(ts, model='additive')
plt.rcParams['figure.figsize'] = [12, 8]
result.plot()
plt.show()


# #### EDA의 결과와 비슷하게, 시계열 분해를 진행했을 때도 지역별로 확인했을 떄 seasonality 가 명확하게 보이는 것을 확인

# #### Time Series Decomposition(시계열 분해)결과, 1주일을 주기로 주기성이 뚜렷하게 파악되어 아래와 같은 모형을 적용 후 가장 optimal한 $\psi_i$를 찾는데 주력
# ## $$ X_t = c + \Sigma_{i=1}^{3} \psi_i X_{t-7*i} + \epsilon_t^*$$

# #### 시계열 분해 결과, Trend에 있어서도 가장 최근 1주일 전의 데이터에 영향을 많이 받을 것으로 예상하여 heuristic하게 $\psi_{i}$를 선정하여 위의 식을 직접 구현하여 모델링 진행

# In[11]:


# 데이터 로딩
df0 = pd.read_csv(aidu_framework.config.data_dir + '/rev_episode2_train_new.csv')
df0.head() #svc : null 제외안함

# 컬럼 수정 
df0['base_date'] = pd.to_datetime(df0['base_date'].astype(str), format='%Y-%m-%d') #기준일을 string -> datetime 포맷 변환 
df0['base_day'] =  [d.weekday() for d in df0['base_date']] #요일 컬럼 생성 (월=0, 화=1, ... , 일=6)

# 지역/기준일별로 패킷량 합침 
df1 = df0.groupby(['gun_gu_nm','base_date','base_day'])['sum_packt_trm_qnt'].agg('sum').reset_index(name ='y')
df1

# 5차 제출 예측치
df = df1.copy()
result5=pd.DataFrame(columns=['ID','1022','1023','1024'])
result5['ID']=df.gun_gu_nm.unique()


# In[12]:


for g in df.gun_gu_nm.unique():
    ylist1 = df[(df.gun_gu_nm==g)&(df.base_day==3)]['y'].tolist() #목
    ylist2 = df[(df.gun_gu_nm==g)&(df.base_day==4)]['y'].tolist() #금
    ylist3 = df[(df.gun_gu_nm==g)&(df.base_day==5)]['y'].tolist() #토
   
    result5.loc[result5.ID==g,'1022'] = np.average(df[(df.gun_gu_nm==g)&(df.base_day==3)]['y'], weights=[.0,.1,.9])   # 가중치
    result5.loc[result5.ID==g,'1023'] = np.average(df[(df.gun_gu_nm==g)&(df.base_day==4)]['y'], weights=[.0,.1,.9])   # 가중치
    result5.loc[result5.ID==g,'1024'] = np.average(df[(df.gun_gu_nm==g)&(df.base_day==5)]['y'], weights=[.0,.1,.9])   # 가중치
result5


# In[13]:


# 5차 중간결과 제출(final 결과 제출)
# result5.to_csv(aidu_framework.config.data_dir + '/5차_aiplay1115_result_ep2.csv', header=False, index=False)

# 최종 예선 결과 제출
# result5.to_csv(aidu_framework.config.data_dir + '/aiplay1115_result_ep2.csv', header=False, index=False)


# In[14]:


# plot 그리기
g='G001'
plt.plot(range(24),pd.concat([df.loc[df.gun_gu_nm==g,'y'],pd.DataFrame(result5[result5.ID==g].drop('ID', axis=1).values.reshape(3,1))])) # 보라색 (5차)


# In[15]:


##############################################################################
# 3) RMSE 평가
##############################################################################
# RMSE 평가
## test set : 직전 1주
## 직전 2주간 train 후 진전 1주 test set으로 error rmse 구하기
## 직전 2주만 데이터가 있어서 실제 rmse와 차이가 있을 것으로 예상됨
### 10.15~21 요일별로 평가

df_train = df1.loc[df1.base_date < '2020-10-15']

df_1015 = df1.loc[df1.base_date == '2020-10-15']
df_1016 = df1.loc[df1.base_date == '2020-10-16']
df_1017 = df1.loc[df1.base_date == '2020-10-17']
df_1018 = df1.loc[df1.base_date == '2020-10-18']
df_1019 = df1.loc[df1.base_date == '2020-10-19']
df_1020 = df1.loc[df1.base_date == '2020-10-20']
df_1021 = df1.loc[df1.base_date == '2020-10-21']


# In[16]:


df = df_train.copy()
result1=pd.DataFrame(columns=['ID','1015','1016','1017','1018','1019','1020','1021'])
result1['ID']=df.gun_gu_nm.unique()

for g in df.gun_gu_nm.unique():
    weights = [.1,.9]
    result1.loc[result1.ID==g,'1015'] = np.average(df[(df.gun_gu_nm==g)&(df.base_day==3)]['y'], weights=weights)   # 가중치
    result1.loc[result1.ID==g,'1016'] = np.average(df[(df.gun_gu_nm==g)&(df.base_day==4)]['y'], weights=weights)   # 가중치
    result1.loc[result1.ID==g,'1017'] = np.average(df[(df.gun_gu_nm==g)&(df.base_day==5)]['y'], weights=weights)   # 가중치   
    result1.loc[result1.ID==g,'1018'] = np.average(df[(df.gun_gu_nm==g)&(df.base_day==6)]['y'], weights=weights)   # 가중치
    result1.loc[result1.ID==g,'1019'] = np.average(df[(df.gun_gu_nm==g)&(df.base_day==0)]['y'], weights=weights)   # 가중치
    result1.loc[result1.ID==g,'1020'] = np.average(df[(df.gun_gu_nm==g)&(df.base_day==1)]['y'], weights=weights)   # 가중치
    result1.loc[result1.ID==g,'1021'] = np.average(df[(df.gun_gu_nm==g)&(df.base_day==2)]['y'], weights=weights)   # 가중치

result1


# In[17]:


# 오차 계산
rmse1 = np.sqrt(mean_squared_error(df_1015['y'], result1['1015']))
rmse2 = np.sqrt(mean_squared_error(df_1016['y'], result1['1016']))
rmse3 = np.sqrt(mean_squared_error(df_1017['y'], result1['1017']))
rmse4 = np.sqrt(mean_squared_error(df_1018['y'], result1['1018']))
rmse5 = np.sqrt(mean_squared_error(df_1019['y'], result1['1019']))
rmse6 = np.sqrt(mean_squared_error(df_1020['y'], result1['1020']))
rmse7 = np.sqrt(mean_squared_error(df_1021['y'], result1['1021']))

print( rmse1)
print( rmse2)
print( rmse3)


# #### 예선 진행 중반부에 통계적인 모델로 접근 방향을 변경함으로써 optimal한 $\psi_{i}$를 찾는데 아쉬움이 존재.
# #### 예선 진행 중 중간 평가 결과 예측을 위해 heuristic하게 $\psi_{i}$를 선정했으나, 추후 정교화 및 고도화 할 수 있을것으로 판단됨
# #### 데이터가 과거일수가 적고, simple하여 Machine Learning, Deep Learning 보다 통계적인 모델이 성능이 좋아 간단한 모델로 접근하였으나,
# #### 과거 데이터가 더 많다면 다양한 Machine Learning, Deep Learning 방법론을 적용해 볼 수 있을것으로 생각됨.

# In[ ]:



