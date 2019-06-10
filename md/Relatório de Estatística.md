
# Relatório de Estatística

**Autores:** CAVALCANTI, Eduardo; ALVES, Gustavo; MONTE, Wesley.

### Introdução ao problema

O quão relacionados estão a área da sala de estar, o número de pisos, a latitude e a presença ou não de um corpo d'água próximo com o preço das casas em uma região? Utilizando-se de um modelo de regressão linear múltipla podemos inferir conclusões sobre o questionamento acima. Como estudo de caso, escolheu-se o condado de King, localizado no noroeste dos Estados Unidos. A região administrativa engloba a capital do estado de Washington, Seattle e tem uma população estimada de 2.223.163 habitantes, tornando o 12º condado mais populoso do país, de acordo com dados do censo de 2018.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import gmaps
import statsmodels.api as sm
from statsmodels.formula.api import ols

dataframe = pd.read_csv('datasets/kc_house_data.csv')
dataframe.dropna(inplace = True)
variables = ['sqft_living', 'lat', 'waterfront', 'floors']

GMAPS_KEY = 'AIzaSyBdtu_lK68b64TqL3-Fybk_x2D5kyScToA'
coords = dataframe[['lat','long']]
gmaps.configure(api_key=GMAPS_KEY)

figure_layout = {
    'width': '800px',
    'height': '800px',
    'border': '1px solid black',
    'padding': '1px'
}

fig = gmaps.figure(layout=figure_layout,center=[47.605305, -122.207489],zoom_level=11)
fig

```


    Figure(layout=FigureLayout(border='1px solid black', height='800px', padding='1px', width='800px'))


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
```


![png](output_5_0.png)


#### Coeficiente de correlação das variáveis com a variável preço


```python
correlationList = sorted([(chart_x_label(variable), numpy.corrcoef(dataframe.price, dataframe[variable])[1,0]) for variable in variables],
                         key=lambda x: x[1],
                         reverse=True)

pandas.DataFrame(correlationList, columns=['Variável', 'Coeficiente'], index=['']*4)
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



## Modelo de Regressão Múltipla Linear e teste de regressão (ANOVA)


```python
factor = int(len(dataframe)*0.9)
x = dataframe[variables]
x1 = dataframe[['price']+variables]
y = dataframe['price']
train = x[:factor]
train_y = y[:factor]
test = x[factor:]
test_y = y[factor:]

model = ols('price ~ sqft_living + lat + waterfront + floors', data=x1[:factor]).fit()
table = sm.stats.anova_lm(model, typ=1)
table[table.columns[:-1]]
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sqft_living</th>
      <td>1.0</td>
      <td>1.283279e+15</td>
      <td>1.283279e+15</td>
      <td>24085.927566</td>
    </tr>
    <tr>
      <th>lat</th>
      <td>1.0</td>
      <td>1.824972e+14</td>
      <td>1.824972e+14</td>
      <td>3425.299199</td>
    </tr>
    <tr>
      <th>waterfront</th>
      <td>1.0</td>
      <td>1.048298e+14</td>
      <td>1.048298e+14</td>
      <td>1967.555987</td>
    </tr>
    <tr>
      <th>floors</th>
      <td>1.0</td>
      <td>7.824162e+09</td>
      <td>7.824162e+09</td>
      <td>0.146852</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>19446.0</td>
      <td>1.036067e+15</td>
      <td>5.327919e+10</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




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
  <th>Date:</th>             <td>Mon, 10 Jun 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>07:55:39</td>     <th>  Log-Likelihood:    </th> <td>-2.6781e+05</td>
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
  <th>lat</th>         <td> 7.077e+05</td> <td> 1.19e+04</td> <td>   59.527</td> <td> 0.000</td> <td> 6.84e+05</td> <td> 7.31e+05</td>
</tr>
<tr>
  <th>waterfront</th>  <td> 8.311e+05</td> <td> 1.87e+04</td> <td>   44.358</td> <td> 0.000</td> <td> 7.94e+05</td> <td> 8.68e+05</td>
</tr>
<tr>
  <th>floors</th>      <td> 1357.4631</td> <td> 3542.321</td> <td>    0.383</td> <td> 0.702</td> <td>-5585.791</td> <td> 8300.718</td>
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




```python
reg = linear_model.LinearRegression()
reg.fit(train,train_y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
reg.intercept_
```




    -33680920.8109299




```python
reg.coef_
```




    array([2.68033152e+02, 7.07660741e+05, 8.31113848e+05, 1.35746313e+03])




```python
reg.score(test,test_y)
```




    0.6267720881507608




```python
pd.DataFrame(np.array([test_y[775:778],reg.predict(test[775:778])]).transpose(),columns=['Valor Real','Valor Previsto'],index=[775,776,777])
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
      <th>775</th>
      <td>585000.0</td>
      <td>607168.208808</td>
    </tr>
    <tr>
      <th>776</th>
      <td>380000.0</td>
      <td>373519.121336</td>
    </tr>
    <tr>
      <th>777</th>
      <td>565997.0</td>
      <td>657219.508581</td>
    </tr>
  </tbody>
</table>
</div>



# Explicações

> * Área da Sala de estar (ft²)
> * Latitude
> * Presença de corpo d\'água próximo ao imóvel
> * Número de pisos


```python
heatmap_layer = gmaps.heatmap_layer(coords,weights=dataframe['price'])
fig.add_layer(heatmap_layer)
fig
```


    Figure(layout=FigureLayout(border='1px solid black', height='800px', padding='1px', width='800px'))



```python

```
