import os
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import datetime as dt
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#########################################
#--------------- setting ----------------
#########################################

dir = os.path.dirname(os.path.abspath(__file__)) + "\\"

#推計に使用するデータの初期と終期
sdate = dt.datetime.strptime("2013-01-01", "%Y-%m-%d")
edate = dt.datetime.strptime("2020-03-01", "%Y-%m-%d")

est_sdate = dt.datetime.strptime("2019-08-01", "%Y-%m-%d")
est_edate = dt.datetime.strptime("2019-11-01", "%Y-%m-%d")

#データ格納してるファイルの名前
file_name = "data"

#説明変数の名前のリスト
#x_name = ["const", "d_wg_k", "d_c_index"]
#x_name = ["const", "d_wg_k"]
x_name = ["const", "d_c_index"]

#被説明変数の名前のリスト
y_name = ["d_ita"]

#データ読み込み
data = pd.read_excel(dir+file_name+".xlsx", sheet_name="data", index_col=0, parse_dates=True)

#使用データの分離
data_x =  data.loc[sdate:edate, x_name]
data_y =  data.loc[sdate:edate, y_name]

#カーネル密度を推計したい時期
data_est =  data.loc[est_sdate:est_edate, x_name]

##################
### 分位点回帰

#model
model = QuantReg(data_y, data_x)

#分位点回帰の刻み幅
step = 0.01
n = int(1/step) - 1

#係数行列
coeff = np.ones((n, len(data_x.T)))
for i in range(n):
  res = model.fit(q=step*(i+1))
  coeff[i,:] = np.array(list(res.params)).reshape(1,-1)

#########################
### カーネル密度分布の推計

#疑似逆累積分布関数の作成
est_values = np.dot(coeff, np.array(data_est).T)

for i in range(len(est_values.T)):
  delta = relativedelta(months=i)
  sns.kdeplot(est_values[:,i], kernel="epa", label=est_sdate+delta)
plt.show()
plt.close()
