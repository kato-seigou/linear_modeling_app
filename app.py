import streamlit as st

# フォントの設定
from mplfonts import use_font
use_font("Noto Sans CJK JP")
import matplotlib
matplotlib.rcParams["axes.unicode_minus"] = False

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="モデルの精度比較", layout="wide")

st.title("モデルの精度比較")

# -- セッションの初期化 --
# streamlitは「ユーザー操作がある度に、スクリプト全体が最初から最後まで再実行される」
for k, v in {
    "df_X": None, "df_y": None,
    "show_model_ui": None,
    "model_option": None,
    "mse": None, "alpha": None, "x_col": None
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# リセット関数
def reset_after_data():
    # データ生成をやり直した時にそれ以降の処理を初期化
    st.session_state.show_model_ui = False
    st.session_state.model_option = None
    st.session_state.mse = None
    st.session_state.alpha = None
    st.session_state.x_col = None

def reset_after_x():
    # グラフ作成でx列を変更したときにモデル以降の処理を初期化
    st.session_state.show_model_ui = False
    st.session_state.model_option = False
    st.session_state.mse = None
    st.session_state.alpha = None


### データの生成
st.header("サンプルデータの生成")
n_samples = st.slider("データのサイズ", 100, 1000, 100)
n_features = st.slider("データの種数", 5, 20, 5)
n_informative = st.slider("意味のあるデータの種数", 1, n_features, 1)
effective_rank = st.slider("共線性のあるデータの種数", 1, n_features, 1)


if st.button("データを生成する"): # ボタンを押したらデータが生成されるようにする
    X, y, true_coef = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=10.0,
        effective_rank=effective_rank,
        tail_strength=0.5,
        coef=True,
        random_state=42
    )
    
    st.session_state.df_X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    st.session_state.df_y = pd.DataFrame(y, columns=["y"])
    st.success("データが生成されました")

    df_X = st.session_state.df_X
    df_y = st.session_state.df_y

### グラフ
st.header(f"グラフの表示")
if st.session_state.df_X is not None:
    # 表示するデータの列を選ばせる
    columns = [c for c in st.session_state.df_X.columns]
    option = st.radio("どの列を選びますか", columns)

    # グラフの描写
    fig, ax = plt.subplots()
    sns.regplot(x=st.session_state.df_X[option], y=st.session_state.df_y["y"], ax=ax)
    ax.set_title(f"Relationship of {option} and Y")

    # グラフの表示
    if st.button("グラフを表示する"):
        st.pyplot(fig)
        # フラグを立てて、次の処理に行けるようにする
        st.session_state.show_model_ui = True

else:
    st.info(f"まず「データを生成する」ボタンを押してください")
    
### モデルを作成する
st.header(f"モデルの作成と精度評価")
if st.session_state.show_model_ui and st.session_state.df_X is not None:
    X= StandardScaler().fit_transform(st.session_state.df_X.values)
    y = st.session_state.df_y["y"].values

    # どのモデルを作るかを指定させる
    model_options = ["Normal", "Lasso", "Ridge"]
    st.session_state.model_option = st.selectbox("どのモデルを作りたいですか？", model_options)

    alphas = np.logspace(-3, 3, 20)

    if st.session_state.model_option == "Normal":
        reg = LinearRegression().fit(X, y)
    elif st.session_state.model_option == "Ridge":
        reg = RidgeCV(alphas=alphas, cv=5).fit(X, y)
    elif st.session_state.model_option == "Lasso":
        reg = LassoCV(alphas=alphas, cv=5).fit(X, y)

    y_pred = reg.predict(X)
    st.session_state.mse = mean_squared_error(y, y_pred)
    st.write(f"{st.session_state.model_option}のMSE: {st.session_state.mse:.3f}")
    
    if hasattr(reg, "alpha_"):
        st.session_state.alpha = reg.alpha_
        st.write(f"選択されたalpha: {st.session_state.alpha:.3f}")