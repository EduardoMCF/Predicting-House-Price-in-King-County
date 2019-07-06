
# Relatório de Estatística

**Autores:** CAVALCANTI, Eduardo; ALVES, Gustavo; MONTE, Wesley.

### Introdução ao problema

O quão relacionados estão a área da sala de estar, o número de pisos, a latitude e a presença ou não de um corpo d'água próximo com o preço das casas em uma região? Utilizando-se de um modelo de regressão linear múltipla podemos inferir conclusões sobre o questionamento acima. Como estudo de caso, escolheu-se o condado de King, localizado no noroeste dos Estados Unidos. A região administrativa engloba a capital do estado de Washington, Seattle e tem uma população estimada de 2.223.163 habitantes, tornando o 12º condado mais populoso do país, de acordo com dados do censo de 2018.


```python
import gmaps
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.stats
import statsmodels.api as sm
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt

from matplotlib import cm,colors
from sklearn import linear_model
from statsmodels.formula.api import ols
from scipy import stats

from math import log, ceil

dataframe = pd.read_csv('datasets/kc_house_data.csv')
dataframe.dropna(inplace = True)
variables = ['sqft_living', 'lat', 'waterfront', 'floors']

GMAPS_KEY = 'AIzaSyCmanG0fu4D7t02Hfb72h_MLrpi9gBi1Ps'
coords = dataframe[['lat','long']]
gmaps.configure(api_key=GMAPS_KEY)

figure_layout = {
    'width': '800px',
    'height': '800px',
    'border': '1px solid black',
    'padding': '1px'
}

fig_maps = gmaps.figure(layout=figure_layout,center=[47.605305, -122.207489],zoom_level=11)
fig_maps

```


    Figure(layout=FigureLayout(border='1px solid black', height='800px', padding='1px', width='800px'))

![png](output_1_0.png)
Condado de King (fonte: Google Maps)

### Sobre o conjunto de dados utilizado

Para realização da análise, utilizou-se um *dataset* com informações sobre vendas de imóveis na região durante o período de maio de 2014 a maio de 2015. O *dataset* possui informações coletadas sobre 21.613 imóveis em relação às seguintes características: número de quartos, banheiros, área construída, da sala de estar, presença ou não de amenidades ao redor, número de andares, preço e outras informações. O arquivo pode ser encontrado no seguinte link: https://www.kaggle.com/harlfoxem/housesalesprediction.

### Análise descritiva do domínio do problema

Antes de partir para a montagem do MRLM, analisaremos algumas das informações dos dados:

#### Histogramas e gráficos em barra das variáveis


```python
def chart_title(variable):
    titles = {
        'sqft_living': 'Histograma da variável \"Área da sala de estar (ft²)\"',
        'lat': 'Histograma da variável \"latitude"',
        'waterfront': 'Gráfico em barra da variável \"Presença de corpo d\'água próximo ao imóvel\"',
        'floors': 'Gráfico em barra da variável \"Número de pisos\""'
    }

    return titles[variable]

def chart_x_label(variable):
    names = {
        'sqft_living': 'Área da sala de estar (ft²)',
        'lat': 'Latitude',
        'waterfront': 'Presença de corpo d\'água próximo',
        'floors': 'Número de pisos'
    }

    return names[variable]

def get_count_variable(variable):
    ret = dataframe.groupby([variable]).size()
    return (ret.index,ret)

charts = plt.figure(figsize=(15,10))
plt.subplots_adjust(wspace=0.2, hspace=0.3)

for idx, variable in enumerate(['sqft_living', 'lat']):
    plot = charts.add_subplot(2, 2, idx + 1)
    plot.set(xlabel=chart_x_label(variable),
             ylabel='Quantidade',
             title=chart_title(variable))
    plot.hist(dataframe[variable])

for idx, variable in enumerate(['waterfront','floors']):
    plot = charts.add_subplot(2, 2, idx + 3)
    plot.set(xlabel=chart_x_label(variable),
             ylabel='Quantidade',
             title=chart_title(variable))
    ticks,values = get_count_variable(variable)
    plot.bar(ticks, values, 0.4)
    plot.set_xticks(ticks)

    if variable == 'waterfront':
        plot.set_xticklabels(['Não Possui','Possui'])
```


![png](output_3_0.png)


#### Gráficos de dispersão das variáveis


```python
def disp_chart_title(variable):
    titles = {
        'sqft_living': 'Gráfico de dispersão da variável \"Área da sala de estar (ft²)\"',
        'lat': 'Gráfico de dispersão da variável \"latitude"',
        'waterfront': 'Gráfico de dispersão da variável \"Presença de corpo d\'água próximo ao imóvel\"',
        'floors': 'Gráfico de dispersão da variável \"Número de pisos\""'
    }

    return titles[variable]

scatters = plt.figure(figsize=(15, 10))
plt.subplots_adjust(wspace=0.2, hspace=0.3)

for idx, variable in enumerate(variables):
    plot = scatters.add_subplot(2, 2, idx + 1)
    plot.set(ylabel='Price (Mi)',
             xlabel=chart_x_label(variable),
             title=disp_chart_title(variable))
    plot.scatter(dataframe[variable], dataframe.price / 10**6)

    if variable == 'waterfront':
        plot.set_xticks([0,1])
        plot.set_xticklabels(['Não Possui','Possui'])

```


![png](output_5_0.png)


Com isso, podemos ver que o gráfico de dispersão da variável waterfront é um tanto quanto peculiar. Isso de dá porque essa variável é qualitativa nominal (assume os valores: possui ou não possui) e o preço é uma variável quantitativa contínua. Logo, um gráfico de dispersão ,que representa a associação entre duas variáveis quantitativas, não é apropriado nesse caso.

Dito isso, uma forma apropriada de representar a associação entre essas duas variáveis é através de um **boxplot**.


```python
df = dataframe['price waterfront'.split()] / [10**6,1]
melted_df = pd.melt(df,id_vars=['waterfront'],value_vars=['price'],var_name="Presença de corpo d'água")
sns.set_style(style = 'white')
fig = plt.figure(figsize=(11, 9))
ax = sns.boxplot(x='waterfront',y='value',data=melted_df)
ax.set(xlabel='Presença de corpo d\'água próximo', ylabel='Preço (Mi)', xticklabels=['Não Possui','Possui'])
plt.show()
```


![png](output_7_0.png)


Com isso, percebemos que na categoria "Não possui" :
- A mediana encontra-se aproximadamente na faixa de meio milhão.
- Há pouca dispersão no preço ,evidenciado pela baixa distancia interquartílica.
- Os dados são assimétricos positivos, pois a mediana aproxima-se do primeiro quartil.
- Há vários outliers, que podem ser visualizados mais a frente.

Já na categoria "Possui":
- A mediana encontra-se aproximadamente na faixa do 1.5 milhão.
- Há uma alta dispersão no preço, denotada pela alta distancia interquartílica.
- Os dados são levemente assimétricos positicos, pois a mediana aproxima-se do primeiro quartil.
- Há alguns outliers.

Em uma análise geral, é possível perceber uma tendencia do preço de um imóvel aumentar caso o mesmo esteja próximo de um corpo d'água (praia, rio, lago, etc).


#### Coeficiente de correlação das variáveis com a variável preço

O coeficiente de correlação de Pearson será utilizado para calcular a correlação entre as variáveis explicativas e a variável resposta, com excessão do waterfront.
Não é adequado calcular a correlação entre waterfront e preço com o coeficiente de pearson, pois ele considera que ambas as variáveis são quantitativas. Nesse caso, um outro coeficiente de correlação será utilizado: o **ponto bisserial**.
O coeficiente **Ponto Bisserial** é baseado no coeficiente pearson (r), porém ele é adequado para correlação entre uma variável dicotômica e uma variável contínua.
Mais informações sobre a correlação Ponto Bisserial podem ser encontradas aqui:https://www.statisticshowto.datasciencecentral.com/point-biserial-correlation/


```python
correlationList = sorted([(chart_x_label(variable),np.corrcoef(dataframe.price, dataframe[variable])[1,0] if variable != "waterfront" else  stats.pointbiserialr(dataframe.price, dataframe.waterfront).correlation) for variable in variables],
                         key=lambda x: x[1],
                         reverse=True)

pd.DataFrame(correlationList, columns=['Variável', 'Coeficiente'], index=['']*4)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variável</th>
      <th>Coeficiente</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th></th>
      <td>Área da sala de estar (ft²)</td>
      <td>0.702035</td>
    </tr>
    <tr>
      <th></th>
      <td>Latitude</td>
      <td>0.307003</td>
    </tr>
    <tr>
      <th></th>
      <td>Presença de corpo d'água próximo</td>
      <td>0.266369</td>
    </tr>
    <tr>
      <th></th>
      <td>Número de pisos</td>
      <td>0.256794</td>
    </tr>
  </tbody>
</table>
</div>



#### Média, mediana, desvio padrão, quartis, etc. das variáveis


```python
dataframe[variables].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sqft_living</th>
      <th>lat</th>
      <th>waterfront</th>
      <th>floors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>21613.000000</td>
      <td>21613.000000</td>
      <td>21613.000000</td>
      <td>21613.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2079.899736</td>
      <td>47.560053</td>
      <td>0.007542</td>
      <td>1.494309</td>
    </tr>
    <tr>
      <th>std</th>
      <td>918.440897</td>
      <td>0.138564</td>
      <td>0.086517</td>
      <td>0.539989</td>
    </tr>
    <tr>
      <th>min</th>
      <td>290.000000</td>
      <td>47.155900</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1427.000000</td>
      <td>47.471000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1910.000000</td>
      <td>47.571800</td>
      <td>0.000000</td>
      <td>1.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2550.000000</td>
      <td>47.678000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>13540.000000</td>
      <td>47.777600</td>
      <td>1.000000</td>
      <td>3.500000</td>
    </tr>
  </tbody>
</table>
</div>



## Construindo a matrix de correlação


```python
def zip_escalar(l,e):
    return [(e,i) for i in l]

corr = dataframe[['price']+variables].corr()
corr['waterfront'] = list(map(lambda x: stats.pointbiserialr(dataframe[x[0]],dataframe[x[1]]).correlation, zip_escalar(['price']+variables,'waterfront')))

def CorrMtx(df, dropDuplicates = True, vmin = -1, vmax = 1):
    labels = df
    if dropDuplicates:
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

    sns.set_style(style = 'white')
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    if dropDuplicates:
        sns.heatmap(df, mask=mask, cmap=cmap, annot = labels,
                square=True,
                linewidth=.5, cbar_kws={"shrink": .5}, vmin = vmin, vmax = vmax, ax=ax)
    else:
        sns.heatmap(df, cmap=cmap, annot = labels,
                square=True,
                linewidth=.5, cbar_kws={"shrink": .5}, vmin = vmin, vmax = vmax, ax=ax)

CorrMtx(corr)
```


![png](output_14_0.png)


## Modelo de Regressão Múltipla Linear e teste de regressão (ANOVA)


```python
def get_groups(df, dependent_variable, independent_variables, factor = 0.9, adjust = False):
    model_variables = [dependent_variable] + independent_variables
    values = df[model_variables].max().apply(lambda x: 10**ceil(log(x,10))).to_numpy()
    slice_point = int(len(df)*factor)
    x = df[variables]
    y = df[dependent_variable]
    model_df = df[model_variables] / (values if adjust else 1)
    return list(map(lambda DF: (DF[:slice_point],DF[:slice_point]),[x,y,model_df])) + [values]


factor = int(len(dataframe)*0.2)
(train,test),(train_y,test_y),(model_train,model_test),values = get_groups(dataframe, 'price', variables)
model = ols('price ~ sqft_living + waterfront + floors + lat', data=model_train).fit()
table = sm.stats.anova_lm(model, typ=1)
table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>df</th>
      <th>sum_sq</th>
      <th>mean_sq</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sqft_living</th>
      <td>1.0</td>
      <td>1.283279e+15</td>
      <td>1.283279e+15</td>
      <td>24085.927566</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>waterfront</th>
      <td>1.0</td>
      <td>9.846525e+13</td>
      <td>9.846525e+13</td>
      <td>1848.099562</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>floors</th>
      <td>1.0</td>
      <td>7.413925e+10</td>
      <td>7.413925e+10</td>
      <td>1.391524</td>
      <td>0.238162</td>
    </tr>
    <tr>
      <th>lat</th>
      <td>1.0</td>
      <td>1.887954e+14</td>
      <td>1.887954e+14</td>
      <td>3543.510952</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>19446.0</td>
      <td>1.036067e+15</td>
      <td>5.327919e+10</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Aqui nós temos a tabela ANOVA com a informação das somas e médias dos quadrados e valor F para cada uma das variáveis explicativas e para os resíduos.

## Escolhendo o modelo
Para a escolha do melhor modelo utilizaremos o método Backward.
Fixamos o **nível de significância em 5%** e agora iremos ajustar o modelo com todas as variáveis citadas.


```python
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.603</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.602</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   7370.</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 06 Jul 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>
</tr>
<tr>
  <th>Time:</th>                 <td>17:41:25</td>     <th>  Log-Likelihood:    </th> <td>-2.6781e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 19451</td>      <th>  AIC:               </th>  <td>5.356e+05</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 19446</td>      <th>  BIC:               </th>  <td>5.357e+05</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>      <td> </td>
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>
</tr>
<tr>
  <th>Intercept</th>   <td>-3.368e+07</td> <td> 5.65e+05</td> <td>  -59.599</td> <td> 0.000</td> <td>-3.48e+07</td> <td>-3.26e+07</td>
</tr>
<tr>
  <th>sqft_living</th> <td>  268.0332</td> <td>    2.015</td> <td>  133.013</td> <td> 0.000</td> <td>  264.083</td> <td>  271.983</td>
</tr>
<tr>
  <th>waterfront</th>  <td> 8.311e+05</td> <td> 1.87e+04</td> <td>   44.358</td> <td> 0.000</td> <td> 7.94e+05</td> <td> 8.68e+05</td>
</tr>
<tr>
  <th>floors</th>      <td> 1357.4631</td> <td> 3542.321</td> <td>    0.383</td> <td> 0.702</td> <td>-5585.791</td> <td> 8300.718</td>
</tr>
<tr>
  <th>lat</th>         <td> 7.077e+05</td> <td> 1.19e+04</td> <td>   59.527</td> <td> 0.000</td> <td> 6.84e+05</td> <td> 7.31e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>14575.675</td> <th>  Durbin-Watson:     </th>  <td>   1.990</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>820664.950</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 3.079</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td>34.220</td>   <th>  Cond. No.          </th>  <td>7.66e+05</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 7.66e+05. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



É possível ver que o **valor p** da variável independente floors é maior do que o nível de significância estabelecido, diante disso iremos removê-la do modelo.


```python
(train_,test_),(train_y_,test_y_),(model_train,model_test),values = get_groups(dataframe, 'price', ['sqft_living','waterfront','lat'])
model_ = ols('price ~ sqft_living + waterfront + lat', data=model_train).fit()
model_.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.603</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.602</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   9827.</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 06 Jul 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>
</tr>
<tr>
  <th>Time:</th>                 <td>17:41:25</td>     <th>  Log-Likelihood:    </th> <td>-2.6781e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 19451</td>      <th>  AIC:               </th>  <td>5.356e+05</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 19447</td>      <th>  BIC:               </th>  <td>5.357e+05</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>      <td> </td>
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>
</tr>
<tr>
  <th>Intercept</th>   <td>-3.368e+07</td> <td> 5.65e+05</td> <td>  -59.604</td> <td> 0.000</td> <td>-3.48e+07</td> <td>-3.26e+07</td>
</tr>
<tr>
  <th>sqft_living</th> <td>  268.3408</td> <td>    1.848</td> <td>  145.188</td> <td> 0.000</td> <td>  264.718</td> <td>  271.963</td>
</tr>
<tr>
  <th>waterfront</th>  <td>  8.31e+05</td> <td> 1.87e+04</td> <td>   44.358</td> <td> 0.000</td> <td> 7.94e+05</td> <td> 8.68e+05</td>
</tr>
<tr>
  <th>lat</th>         <td> 7.077e+05</td> <td> 1.19e+04</td> <td>   59.539</td> <td> 0.000</td> <td> 6.84e+05</td> <td> 7.31e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>14565.091</td> <th>  Durbin-Watson:     </th>  <td>   1.990</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>818608.990</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 3.076</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td>34.180</td>   <th>  Cond. No.          </th>  <td>7.66e+05</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 7.66e+05. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



Já que todos os **valores p** estão **abaixo do nível de significância** estabelecido, podemos parar o método backward e utilizar esse modelo.


Também possível perceber que o **condition number** é muito elevado, mas isso é devido à uma grande distância entre os valores do modelo. Por exemplo, o maior valor que a variável preço assume é da ordem de 10^7 enquanto que o maior valor da variável waterfront é 1. Essa distância causa erros numéricos, por isso iremos colocar todas as variáveis do modelo em um intervalo entre 0 e 1.


```python
model_train.max()
```




    price          7.700000e+06
    sqft_living    1.354000e+04
    waterfront     1.000000e+00
    lat            4.777760e+01
    dtype: float64




```python
(train,test),(train_y,test_y),(model_train,model_test),values = get_groups(dataframe, 'price', ['sqft_living','waterfront','lat'], adjust = True)
model = ols('price ~ sqft_living + waterfront + lat', data=model_train).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.603</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.602</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   9827.</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 06 Jul 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>
</tr>
<tr>
  <th>Time:</th>                 <td>17:41:26</td>     <th>  Log-Likelihood:    </th>  <td>  45707.</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 19451</td>      <th>  AIC:               </th> <td>-9.141e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 19447</td>      <th>  BIC:               </th> <td>-9.138e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>      <td> </td>
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>
</tr>
<tr>
  <th>Intercept</th>   <td>   -3.3683</td> <td>    0.057</td> <td>  -59.604</td> <td> 0.000</td> <td>   -3.479</td> <td>   -3.257</td>
</tr>
<tr>
  <th>sqft_living</th> <td>    2.6834</td> <td>    0.018</td> <td>  145.188</td> <td> 0.000</td> <td>    2.647</td> <td>    2.720</td>
</tr>
<tr>
  <th>waterfront</th>  <td>    0.0831</td> <td>    0.002</td> <td>   44.358</td> <td> 0.000</td> <td>    0.079</td> <td>    0.087</td>
</tr>
<tr>
  <th>lat</th>         <td>    7.0772</td> <td>    0.119</td> <td>   59.539</td> <td> 0.000</td> <td>    6.844</td> <td>    7.310</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>14565.091</td> <th>  Durbin-Watson:     </th>  <td>   1.990</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>818608.990</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 3.076</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td>34.180</td>   <th>  Cond. No.          </th>  <td>    881.</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



É possível observar que o condition number diminui e agora está em níveis aceitaveis, isso endossa que há pouca colinearidade entre as variáveis explicativas.
Também é possível observar os valores das variáveis depois do ajuste.


```python
model_train.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>sqft_living</th>
      <th>waterfront</th>
      <th>lat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.02219</td>
      <td>0.0118</td>
      <td>0.0</td>
      <td>0.475112</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.05380</td>
      <td>0.0257</td>
      <td>0.0</td>
      <td>0.477210</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.01800</td>
      <td>0.0077</td>
      <td>0.0</td>
      <td>0.477379</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.06040</td>
      <td>0.0196</td>
      <td>0.0</td>
      <td>0.475208</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.05100</td>
      <td>0.0168</td>
      <td>0.0</td>
      <td>0.476168</td>
    </tr>
  </tbody>
</table>
</div>




```python
model_train.max()
```




    price          0.770000
    sqft_living    0.135400
    waterfront     1.000000
    lat            0.477776
    dtype: float64




```python
model_train.min()
```




    price          0.007500
    sqft_living    0.003700
    waterfront     0.000000
    lat            0.471559
    dtype: float64



# Análise de resíduos
Nessa análise iremos avaliar 3 tópicos:
- Linearidade
- Normalidade
- Homocedasticidade

## Linearidade.
Podemos analisar a linearidade a partir de um gráfico entre os valores preditos pela regressão e os valores reais.


```python
def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


y_hat = model.fittedvalues.copy()
y_np = model_train['price'].values.copy()
residual = y_np - y_hat

fig = plt.figure(figsize=(10,7))
plt.plot(y_hat,y_np,'o')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Predito vs. Atual: Teste de Linearidade Visual')
plt.tick_params(axis='x', colors='white')
plt.tick_params(axis='y', colors='white')
abline(1,0)
plt.show()
```


![png](output_30_0.png)


O gráfico mostra que há uma certa linearidade, porém baixa.

## Homocedasticidade
Na análise de homocedasticidade utilizamos um gráfico de dispersão entre os valores preditos (y_chapeu) e os residuos (y - y_chapeu).


```python
plt.figure(figsize=(10,7))
plt.plot(y_hat,y_np-y_hat,'o')
plt.xlabel('Predito y_chapeu')
plt.ylabel('Residuos y - y_chapeu')
plt.title('Predito vs Residuos')
plt.tick_params(axis='x')
plt.tick_params(axis='y')
```


![png](output_33_0.png)


Não há como tirar grandes conclusões com o gráfico, então utilizaremos um teste de hipótese.


```python
_, pval, __, _ = statsmodels.stats.diagnostic.het_breuschpagan(residual, model_train[variables[:-1]])
pval
```




    0.0



O teste de Breusch-Pagan avalia a homocedasticidade, um p-valor menor que o nível de significância nos faz rejeitar a hipótese nula (Há homocedasticidade) e dá indicios de Heterodasticidade.
Esse é o caso dessa regressão, o **p-valor (0.0) é menor que o nível de significância estabelecido (0.05)**. Logo, é um indicio de **Heterodasticidade**.

## Normalidade



```python
fig, ax = plt.subplots(figsize=(10,7))
_, (__, ___, r) = stats.probplot(residual, plot=ax, fit=True)
r**2
```




    0.8175649375233982




![png](output_38_1.png)


A boa adequação dos valores dos residuos à linha vermelha denota que a distribuição dos mesmos aproxima-se de uma distribuição normal.

## Predição


```python
reg = linear_model.LinearRegression()
reg.fit(train_,train_y_)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
reg.score(test_,test_y_)
```




    0.6025339196671944




```python
predictions = pd.DataFrame(np.array([test_y[110:121],reg.predict(test[110:121])]).transpose(),columns=['Valor Real','Valor Previsto'],index=range(110,121))
predictions['Valor Previsto'] = predictions['Valor Previsto'].apply(lambda x: float(int(x)))
predictions
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Valor Real</th>
      <th>Valor Previsto</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>110</th>
      <td>597750.0</td>
      <td>582151.0</td>
    </tr>
    <tr>
      <th>111</th>
      <td>570000.0</td>
      <td>401050.0</td>
    </tr>
    <tr>
      <th>112</th>
      <td>272500.0</td>
      <td>249718.0</td>
    </tr>
    <tr>
      <th>113</th>
      <td>329950.0</td>
      <td>525373.0</td>
    </tr>
    <tr>
      <th>114</th>
      <td>480000.0</td>
      <td>672844.0</td>
    </tr>
    <tr>
      <th>115</th>
      <td>740500.0</td>
      <td>1249852.0</td>
    </tr>
    <tr>
      <th>116</th>
      <td>518500.0</td>
      <td>492286.0</td>
    </tr>
    <tr>
      <th>117</th>
      <td>205425.0</td>
      <td>170827.0</td>
    </tr>
    <tr>
      <th>118</th>
      <td>171800.0</td>
      <td>352657.0</td>
    </tr>
    <tr>
      <th>119</th>
      <td>535000.0</td>
      <td>428100.0</td>
    </tr>
    <tr>
      <th>120</th>
      <td>660000.0</td>
      <td>770651.0</td>
    </tr>
  </tbody>
</table>
</div>



# Considerações finais


```python
heatmap_layer = gmaps.heatmap_layer(coords,weights=dataframe['price'])
fig_maps.add_layer(heatmap_layer)
fig_maps
```


    Figure(layout=FigureLayout(border='1px solid black', height='800px', padding='1px', width='800px'))

![png](output_42_0.png)

Esse é um mapa de calor onde quanto mais quente for a cor maior é o preço dos imóveis naquela localidade.
É interessante salientar algumas coisas:
- Waterfront é de fato importante na regressão, pois é visível que áreas perto de praias possuem imóveis mais caros.
- Os outliers no boxplot no caso do "Não Possui" podem ser observados, eles são as áreas em que não há corpos d'água porém mesmo assim o preço é elevado (e.g. Issaquah Highlands)
- A latitude também tem sua contribuição, pois é possível perceber que há uma faixa horizontal de altos preços no nível do capitólio.
