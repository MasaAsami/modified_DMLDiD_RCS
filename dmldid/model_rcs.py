import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone
from sklearn.linear_model import LogisticRegressionCV, LassoCV
from IPython.display import clear_output

class DMLDiD_RCS:
    """
    modified DMLDID
    original: 
    Chang, Neng-Chieh. “Double/debiased machine learning for difference-in-differences models.” The Econometrics Journal 23.2 (2020): 177–191.
    """

    def __init__(
        self,
        d_model=LogisticRegressionCV(
            cv=5, random_state=333, penalty="l1", solver="saga"
        ),
        l2k_model=LassoCV(cv=5, random_state=333),
        **kwargs,
    ):
        # params
        self._att_list = []
        self._att = None
        # model
        self.d_model = d_model
        self.l2k_model = l2k_model

    def fit(
        self,
        df: pd.DataFrame,
        y_col: str,
        d_col: str,
        t_col: str,
        X_cols: list,
        dmldid=True,
        sim_cnt=1,
        eps=0.03,
        base_random_seed=0,
        progress_plot=False,
        d_model_t0_only=True,
        l2k_model_separate=True,
        l2k_model_alldata=False,
        l2k_ps_weight=False,
        **kwargs,
    ):
        K = 2  # ２分割
        self._att_list = []  # 初期化
        for l in range(sim_cnt):
            if progress_plot & (l > 0):
                clear_output(wait=True)
                print(f"{l}. att : ", np.mean(self._att_list))
            if dmldid:
                df_set = train_test_split(
                    df,
                    random_state=base_random_seed + l,
                    test_size=0.5,
                    stratify=df[[t_col, d_col]],
                )
                temp_att = []
                for i in range(K):
                    k = 0 if i == 0 else 1
                    c = 1 if i == 0 else 0

                    d_model = clone(self.d_model)

                    if d_model_t0_only:
                        d_model.fit(df_set[c].query(f"{t_col}<1")[X_cols], df_set[c].query(f"{t_col}<1")[d_col])
                    else:
                        d_model.fit(df_set[c][X_cols], df_set[c][d_col])

                    ghat = np.clip(
                        d_model.predict_proba(df_set[k][X_cols])[:, 1],
                        eps,
                        1 - eps,
                    )

                    lamda_hat = df_set[k][t_col].mean()
                    p_hat = df_set[k][d_col].mean()

                    if l2k_model_separate:
                        # l2kmodel for T=1 and T=0 for control
                        l2kmodel_t1_c = clone(self.l2k_model)
                        l2kmodel_t0_c = clone(self.l2k_model)
                        if l2k_model_alldata:
                            control_y = df_set[c][y_col]
                            control_x = df_set[c][X_cols]
                            t_index = df_set[c].query(f"{t_col} > 0").index
                            not_t_index = df_set[c].query(f"{t_col} < 1").index
                        else:
                            control_y = df_set[c].query(f"{d_col} < 1")[y_col]
                            control_x = df_set[c].query(f"{d_col} < 1")[X_cols]
                            t_index = df_set[c].query(f"{d_col} < 1 & {t_col} > 0").index
                            not_t_index = df_set[c].query(f"{d_col} < 1 & {t_col} < 1").index

                        if l2k_ps_weight:
                            ps = np.clip(
                                d_model.predict_proba(control_x.loc[t_index])[:, 1],
                                eps,
                                1 - eps,
                            )
                            l2kmodel_t1_c.fit(control_x.loc[t_index], control_y.loc[t_index], sample_weight = ps/(1-ps))
                            ps = np.clip(
                                d_model.predict_proba(control_x.loc[not_t_index])[:, 1],
                                eps,
                                1 - eps,
                            )
                            l2kmodel_t0_c.fit(
                                control_x.loc[not_t_index], control_y.loc[not_t_index], sample_weight= ps/(1-ps)
                            )
                        else:
                            l2kmodel_t1_c.fit(control_x.loc[t_index], control_y.loc[t_index])
                            l2kmodel_t0_c.fit(
                                control_x.loc[not_t_index], control_y.loc[not_t_index]
                            )

                        l2k_hat_control_pre = l2kmodel_t0_c.predict(df_set[k][X_cols])
                        l2k_hat_control_post = l2kmodel_t1_c.predict(df_set[k][X_cols])

                        l2k_hat_control = l2k_hat_control_post * df_set[k][t_col] + l2k_hat_control_pre * (1 - df_set[k][t_col])

                    else:
                        # l2kmodel = E[Y|T, X]
                        l2kmodel = clone(self.l2k_model)
                        if l2k_model_alldata:
                            control_y = df_set[c][y_col]
                            control_x = df_set[c][X_cols + [t_col]]
                        else:
                            control_y = df_set[c].query(f"{d_col} < 1")[y_col]
                            control_x = df_set[c].query(f"{d_col} < 1")[X_cols + [t_col]]

                        if l2k_ps_weight:
                            ps = np.clip(
                                d_model.predict_proba(control_x)[:, 1],
                                eps,
                                1 - eps,
                            )
                            l2kmodel.fit(control_x, control_y, sample_weight = ps/(1-ps))
                            
                        else:
                            l2kmodel.fit(control_x, control_y)

                        l2k_hat_control = l2kmodel.predict(df_set[k][X_cols + [t_col]])
                            
                    outcome_estimated_diff = df_set[k][y_col] - l2k_hat_control

                    _att = (
                        (df_set[k][t_col] - lamda_hat)
                        * outcome_estimated_diff
                        * (df_set[k][d_col] - ghat)
                        / ((1 - ghat) * lamda_hat * (1 - lamda_hat) * p_hat)
                    ).mean()
                    temp_att.append(_att)

                self._att_list.append(np.mean(temp_att))
            else:
                # Abadie (2005)
                d_model = clone(self.d_model)
                d_model.fit(df[X_cols], df[d_col])
                ghat = np.clip(
                    d_model.predict_proba(df[X_cols])[:, 1],
                    eps,
                    1 - eps,
                )
                p_hat = df[d_col].mean()
                lamda_hat = df[t_col].mean()

                self._att_list.append(
                    (
                        (df[t_col] - lamda_hat)
                        * df[y_col]
                        * (df[d_col] - ghat)
                        / ((1 - ghat) * lamda_hat * (1 - lamda_hat) * p_hat)
                    ).mean()
                )

    def att(self):
        return np.mean(self._att_list)
    
    def sim_att_result(self):
        return self._att_list
    
    # ===================================================================
    # ここからが「漸近分散」を計算するための関数例
    # ===================================================================

    def compute_rcs_variance(
        self,
        df: pd.DataFrame,
        y_col: str,
        d_col: str,
        t_col: str,
        X_cols: list,
        eps=0.03,
        base_random_seed=333,
        d_model_t0_only=True,
        l2k_model_separate=True,
        l2k_model_alldata=False,
        l2k_ps_weight=False,
    ):
        """
        Chang(2020) の Theorem 3.2 相当の式に基づき、
        Repeated Cross Sections 用の漸近分散推定量を計算するサンプル実装例。

        [手順]
          1) データを2分割（train/test）
          2) train で g(x), E[Y|X,T,D=0] を学習
          3) test に対する 影響関数 ψ_i と、補正項 (D-p) + (T-λ) の係数G を評価
          4) 2フォールド分をまとめて平均し、二乗平均で分散推定

        [注意]
          - G_{2p}, G_{2λ} の設定は論文付録を参照し、本当に正しい式か確認すること
          - ここでは簡単に G_{2p} = -θ / p, G_{2λ} = -θ / λ を使う
          - “att()” が既に学習済みの θ を返すものと仮定。
        """
        # すでに fit() を実行済みとして、推定された ATT を使う
        theta_hat = self.att()
        if np.isnan(theta_hat):
            raise ValueError("まず fit() を実行してから呼び出してください。")

        # 2-fold cross-fitting (同じrandom_seedでやると毎回同じ分割になる)
        df_set = train_test_split(
            df,
            random_state=base_random_seed,
            test_size=0.5,
            stratify=df[[t_col, d_col]],
        )

        # 影響関数 (IF) を溜めておくリスト
        psi_values = []
        
        # G_{2p}, G_{2λ} として簡易的に
        #   G_{2p} = - θ / p_hat
        #   G_{2λ} = - θ / λ_hat
        # を使う。2-foldなので foldごとに p_hat, λ_hat が異なる点に注意
        K = 2
        for i in range(K):
            k = 0 if i == 0 else 1
            c = 1 if i == 0 else 0

            d_model = clone(self.d_model)
            if d_model_t0_only:
                d_model.fit(df_set[c].query(f"{t_col}<1")[X_cols],
                            df_set[c].query(f"{t_col}<1")[d_col])
            else:
                d_model.fit(df_set[c][X_cols], df_set[c][d_col])

            # ps
            ghat = np.clip(
                d_model.predict_proba(df_set[k][X_cols])[:, 1],
                eps,
                1 - eps,
            )
            p_hat = df_set[k][d_col].mean()
            lamda_hat = df_set[k][t_col].mean()

            # アウトカムモデル
            if l2k_model_separate:
                l2kmodel_t1_c = clone(self.l2k_model)
                l2kmodel_t0_c = clone(self.l2k_model)
                if l2k_model_alldata:
                    control_y = df_set[c][y_col]
                    control_x = df_set[c][X_cols]
                    t_index = df_set[c].query(f"{t_col} > 0").index
                    not_t_index = df_set[c].query(f"{t_col} < 1").index
                else:
                    control_y = df_set[c].query(f"{d_col} < 1")[y_col]
                    control_x = df_set[c].query(f"{d_col} < 1")[X_cols]
                    t_index = df_set[c].query(f"{d_col} < 1 & {t_col} > 0").index
                    not_t_index = df_set[c].query(f"{d_col} < 1 & {t_col} < 1").index

                if l2k_ps_weight:
                    ps_t = np.clip(
                        d_model.predict_proba(control_x.loc[t_index])[:, 1],
                        eps,
                        1 - eps,
                    )
                    l2kmodel_t1_c.fit(control_x.loc[t_index], control_y.loc[t_index],
                                      sample_weight = ps_t/(1-ps_t))
                    ps_nt = np.clip(
                        d_model.predict_proba(control_x.loc[not_t_index])[:, 1],
                        eps,
                        1 - eps,
                    )
                    l2kmodel_t0_c.fit(control_x.loc[not_t_index], control_y.loc[not_t_index],
                                      sample_weight=ps_nt/(1-ps_nt))
                else:
                    l2kmodel_t1_c.fit(control_x.loc[t_index], control_y.loc[t_index])
                    l2kmodel_t0_c.fit(control_x.loc[not_t_index], control_y.loc[not_t_index])

                l2k_hat_control_pre = l2kmodel_t0_c.predict(df_set[k][X_cols])
                l2k_hat_control_post = l2kmodel_t1_c.predict(df_set[k][X_cols])
                l2k_hat_control = (
                    l2k_hat_control_post * df_set[k][t_col]
                    + l2k_hat_control_pre * (1 - df_set[k][t_col])
                )
            else:
                l2kmodel = clone(self.l2k_model)
                if l2k_model_alldata:
                    control_y = df_set[c][y_col]
                    control_x = df_set[c][X_cols + [t_col]]
                else:
                    control_y = df_set[c].query(f"{d_col} < 1")[y_col]
                    control_x = df_set[c].query(f"{d_col} < 1")[X_cols + [t_col]]
                if l2k_ps_weight:
                    ps_ctrl = np.clip(d_model.predict_proba(control_x)[:, 1],
                                      eps, 1 - eps)
                    l2kmodel.fit(control_x, control_y, sample_weight=ps_ctrl/(1-ps_ctrl))
                else:
                    l2kmodel.fit(control_x, control_y)

                l2k_hat_control = l2kmodel.predict(df_set[k][X_cols + [t_col]])

            # 影響関数の核: psi_2(W_i)
            Y_k = df_set[k][y_col].values
            T_k = df_set[k][t_col].values
            D_k = df_set[k][d_col].values
            res = Y_k - l2k_hat_control

            # ψ_2(W_i; θ, p_hat, λ_hat, η_2)
            psi_core = ((T_k - lamda_hat) * res * (D_k - ghat)
                        / ((1 - ghat) * lamda_hat * (1 - lamda_hat) * p_hat))

            # 補正項 G2p (D - p_hat) + G2λ (T - λ_hat)
            G2p = - theta_hat / p_hat
            G2lambda = - theta_hat / lamda_hat

            # 合算
            #  ψ_i + G2p * (D_i - p_hat) + G2lambda * (T_i - λ_hat)
            partial_if = psi_core \
                         + G2p * (D_k - p_hat) \
                         + G2lambda * (T_k - lamda_hat)

            psi_values.append(partial_if)

        # 2フォールドの partial IF をすべて concat
        psi_values_all = np.concatenate(psi_values)

        # Chang(2020) 式の分散推定は  n^{-1} * sum( (partial_if)^2 ) みたいな形
        # ここではサンプルサイズ = len(psi_values_all)
        n_all = len(psi_values_all)
        var_est = np.mean(psi_values_all**2)  # 不偏にするなら n/(n-1) を掛けるなど調整も可

        return var_est


class KmDMLDiD_RCS:
    """
    Modified DMLDID for Repeated Cross Sections
    (Chang, 2020) を参考に、K-fold cross-fitting 対応。
    
    K=2 とすれば旧実装と同じ二分割とみなせる。
    """

    def __init__(
        self,
        d_model=None,
        l2k_model=None,
    ):
        """
        d_model: Propensity score 用（LogisticRegressionCV など）
        l2k_model: Outcome regression 用（LassoCV など）
        """
        if d_model is None:
            d_model = LogisticRegressionCV(
                cv=5, random_state=333, penalty="l1", solver="saga"
            )
        if l2k_model is None:
            l2k_model = LassoCV(cv=5, random_state=333)

        self.d_model = d_model
        self.l2k_model = l2k_model
        
        # 推定結果を格納
        self._att_list = []
        self._att = None

    def fit(
        self,
        df: pd.DataFrame,
        y_col: str,
        d_col: str,
        t_col: str,
        X_cols: list,
        dmldid: bool = True,
        sim_cnt: int = 1,
        eps: float = 0.03,
        base_random_seed: int = 0,
        progress_plot: bool = False,
        d_model_t0_only: bool = True,
        l2k_model_separate: bool = True,
        l2k_model_alldata: bool = False,
        l2k_ps_weight: bool = False,
        K_folds: int = 2,
        **kwargs,
    ):
        """
        K_folds: クロスフィット分割数 (K=2 で旧実装に対応)
        """
        self._att_list = []  # 毎回 fit() するたびに初期化
        
        # データをコピーして、stratify用に group列を作る (T,D の組み合わせ)
        df_ = df.copy()
        df_["_group"] = df_[t_col]*2 + df_[d_col]  # 0..3 の4パターン想定

        # sim_cnt 回ループ (シミュレーション回数)
        for loop_i in range(sim_cnt):
            if progress_plot and loop_i > 0:
                clear_output(wait=True)
                print(f"{loop_i}/{sim_cnt} ... current ATT mean:", np.mean(self._att_list))

            if dmldid:
                # ===========================
                #  K-fold Cross Fitting
                # ===========================
                skf = StratifiedKFold(
                    n_splits=K_folds,
                    shuffle=True,
                    random_state=base_random_seed + loop_i
                )
                y_strat = df_["_group"].values  # stratify用

                fold_att = []
                for train_idx, test_idx in skf.split(df_, y_strat):
                    train_df = df_.iloc[train_idx]
                    test_df  = df_.iloc[test_idx]

                    # ---- 1) Propensity model ----
                    d_model_ = clone(self.d_model)
                    if d_model_t0_only:
                        # t=0 の部分だけで Dを学習
                        train_df_t0 = train_df.query(f"{t_col} < 1")
                        d_model_.fit(train_df_t0[X_cols], train_df_t0[d_col])
                    else:
                        d_model_.fit(train_df[X_cols], train_df[d_col])

                    # 予測
                    ghat = np.clip(
                        d_model_.predict_proba(test_df[X_cols])[:,1],
                        eps, 1 - eps
                    )
                    lamda_hat = test_df[t_col].mean()
                    p_hat = test_df[d_col].mean()

                    # ---- 2) Outcome model ----
                    if l2k_model_separate:
                        #  T=1 と T=0 の別モデル
                        # （以下は "control" = D=0 のデータだけ用いる例）
                        l2kmodel_t1_c = clone(self.l2k_model)
                        l2kmodel_t0_c = clone(self.l2k_model)

                        if l2k_model_alldata:
                            #  全部のdataで回帰 (制限しない)
                            df_train_ = train_df
                            control_y = df_train_[y_col]
                            control_x = df_train_[X_cols]
                            t_index = df_train_.query(f"{t_col} > 0").index
                            nt_index= df_train_.query(f"{t_col} < 1").index
                        else:
                            #  D=0 のデータに限定
                            df_ctrl_ = train_df.query(f"{d_col} < 1")
                            control_y = df_ctrl_[y_col]
                            control_x = df_ctrl_[X_cols]
                            t_index = df_ctrl_.query(f"{t_col} > 0").index
                            nt_index= df_ctrl_.query(f"{t_col} < 1").index

                        # PS重みなど
                        if l2k_ps_weight:
                            ps = np.clip(
                                d_model_.predict_proba(control_x.loc[t_index])[:,1],
                                eps, 1 - eps
                            )
                            l2kmodel_t1_c.fit(
                                control_x.loc[t_index],
                                control_y.loc[t_index],
                                sample_weight=ps/(1-ps)
                            )
                            ps = np.clip(
                                d_model_.predict_proba(control_x.loc[nt_index])[:,1],
                                eps, 1 - eps
                            )
                            l2kmodel_t0_c.fit(
                                control_x.loc[nt_index],
                                control_y.loc[nt_index],
                                sample_weight=ps/(1-ps)
                            )
                        else:
                            l2kmodel_t1_c.fit(control_x.loc[t_index], control_y.loc[t_index])
                            l2kmodel_t0_c.fit(control_x.loc[nt_index], control_y.loc[nt_index])

                        # テストデータに適用
                        l2k_hat_control_pre  = l2kmodel_t0_c.predict(test_df[X_cols])
                        l2k_hat_control_post = l2kmodel_t1_c.predict(test_df[X_cols])
                        l2k_hat_control = (
                            l2k_hat_control_post * test_df[t_col]
                            + l2k_hat_control_pre * (1 - test_df[t_col])
                        )

                    else:
                        #  単一モデル E[Y | X, T]
                        l2kmodel_ = clone(self.l2k_model)
                        if l2k_model_alldata:
                            df_train_ = train_df
                            control_y = df_train_[y_col]
                            control_x = df_train_[X_cols + [t_col]]
                        else:
                            df_ctrl_ = train_df.query(f"{d_col} < 1")
                            control_y = df_ctrl_[y_col]
                            control_x = df_ctrl_[X_cols + [t_col]]

                        if l2k_ps_weight:
                            ps_ctrl = np.clip(
                                d_model_.predict_proba(control_x)[:,1],
                                eps, 1 - eps
                            )
                            l2kmodel_.fit(control_x, control_y, sample_weight=ps_ctrl/(1-ps_ctrl))
                        else:
                            l2kmodel_.fit(control_x, control_y)

                        l2k_hat_control = l2kmodel_.predict(test_df[X_cols + [t_col]])

                    # ---- 3) ATT part (ψ_core) ----
                    Y_test = test_df[y_col].values
                    T_test = test_df[t_col].values
                    D_test = test_df[d_col].values
                    res_   = Y_test - l2k_hat_control

                    # difference-in-differences のコア
                    att_fold = (
                        (T_test - lamda_hat)*res_*(D_test - ghat)
                        / ((1 - ghat)*lamda_hat*(1 - lamda_hat)*p_hat)
                    ).mean()

                    fold_att.append(att_fold)

                #  fold平均が今回の試行のATT
                self._att_list.append(np.mean(fold_att))

            else:
                # ----------------
                # Abadie (2005) 方式
                # ----------------
                # (sim_cnt回 繰り返し評価するだけ)
                p_hat = df_[d_col].mean()
                lamda_hat = df_[t_col].mean()

                d_model_ = clone(self.d_model)
                d_model_.fit(df_[X_cols], df_[d_col])
                ghat = np.clip(
                    d_model_.predict_proba(df_[X_cols])[:,1],
                    eps,1-eps
                )

                att_abadie = (
                    (df_[t_col] - lamda_hat)
                    * df_[y_col]
                    * (df_[d_col] - ghat)
                    / ((1 - ghat)*lamda_hat*(1 - lamda_hat)*p_hat)
                ).mean()

                self._att_list.append(att_abadie)

        # fit 終了

    def att(self):
        """sim_cnt 回の平均推定値を返す"""
        return np.mean(self._att_list)

    def sim_att_result(self):
        """シミュレーション結果(各回の推定値)一覧"""
        return self._att_list

    # ====================================================
    #  漸近分散推定 (K-fold版)
    # ====================================================

    def compute_rcs_variance(
        self,
        df: pd.DataFrame,
        y_col: str,
        d_col: str,
        t_col: str,
        X_cols: list,
        eps=0.03,
        base_random_seed=999,
        d_model_t0_only=True,
        l2k_model_separate=True,
        l2k_model_alldata=False,
        l2k_ps_weight=False,
        K_folds=2,
    ):
        """
        Chang(2020) Theorem 3.2 風に、K-fold cross-fitting で
        Repeated Cross Sections の漸近分散を推定するサンプルコード。
        
        [手順]
          - fit() と同じく StratifiedKFold(K_folds) で分割し、
            foldごとに Propensity & Outcome を学習
          - 影響関数 (partial_if) を各サンプルに対して計算
          - それらを 全fold concat して 二乗平均 => 分散 とする。
          
        [注意]
          - G_{2p}, G_{2λ} = -θ / p, -θ / λ として簡易に書いています。
            Chang(2020) 付録と完全一致させるなら適宜修正してください。
          - 2ステージ推定部分(Outcomeが treated と control の両方ある場合)を
            より厳密に扱うなら若干スコア式が変わるかもしれません。
        """
        theta_hat = self.att()  # すでに fit() で求めた推定値
        if np.isnan(theta_hat):
            raise ValueError("先に fit() を呼び出してから var を計算してください。")

        df_ = df.copy()
        df_["_group"] = df_[t_col]*2 + df_[d_col]
        y_strat = df_["_group"].values

        # IFをためる
        partial_if_list = []

        skf = StratifiedKFold(
            n_splits=K_folds, shuffle=True,
            random_state=base_random_seed
        )

        for train_idx, test_idx in skf.split(df_, y_strat):
            train_df = df_.iloc[train_idx]
            test_df  = df_.iloc[test_idx]

            # ---- 1) Propensity ----
            d_model_ = clone(self.d_model)
            if d_model_t0_only:
                train_df_t0 = train_df.query(f"{t_col} < 1")
                d_model_.fit(
                    train_df_t0[X_cols],
                    train_df_t0[d_col]
                )
            else:
                d_model_.fit(
                    train_df[X_cols],
                    train_df[d_col]
                )

            ghat = np.clip(
                d_model_.predict_proba(test_df[X_cols])[:,1],
                eps,1-eps
            )
            p_hat = test_df[d_col].mean()
            lamda_hat = test_df[t_col].mean()

            # ---- 2) outcome (control) ----
            if l2k_model_separate:
                l2k_t1_c = clone(self.l2k_model)
                l2k_t0_c = clone(self.l2k_model)

                if l2k_model_alldata:
                    cdf_ = train_df
                    ctrl_y = cdf_[y_col]
                    ctrl_x = cdf_[X_cols]
                    idx_t1 = cdf_.query(f"{t_col} > 0").index
                    idx_t0 = cdf_.query(f"{t_col} < 1").index
                else:
                    cdf_ = train_df.query(f"{d_col} < 1")
                    ctrl_y = cdf_[y_col]
                    ctrl_x = cdf_[X_cols]
                    idx_t1 = cdf_.query(f"{t_col} > 0").index
                    idx_t0 = cdf_.query(f"{t_col} < 1").index

                if l2k_ps_weight:
                    ps_t1 = np.clip(
                        d_model_.predict_proba(ctrl_x.loc[idx_t1])[:,1],
                        eps, 1-eps
                    )
                    l2k_t1_c.fit(
                        ctrl_x.loc[idx_t1], ctrl_y.loc[idx_t1],
                        sample_weight=ps_t1/(1-ps_t1)
                    )
                    ps_t0 = np.clip(
                        d_model_.predict_proba(ctrl_x.loc[idx_t0])[:,1],
                        eps, 1-eps
                    )
                    l2k_t0_c.fit(
                        ctrl_x.loc[idx_t0], ctrl_y.loc[idx_t0],
                        sample_weight=ps_t0/(1-ps_t0)
                    )
                else:
                    l2k_t1_c.fit(ctrl_x.loc[idx_t1], ctrl_y.loc[idx_t1])
                    l2k_t0_c.fit(ctrl_x.loc[idx_t0], ctrl_y.loc[idx_t0])

                c_pre  = l2k_t0_c.predict(test_df[X_cols])
                c_post = l2k_t1_c.predict(test_df[X_cols])
                hat_c  = c_post * test_df[t_col] + c_pre*(1 - test_df[t_col])

            else:
                l2k_ = clone(self.l2k_model)
                if l2k_model_alldata:
                    ctrl_y = train_df[y_col]
                    ctrl_x = train_df[X_cols + [t_col]]
                else:
                    df_ctrl_ = train_df.query(f"{d_col} < 1")
                    ctrl_y = df_ctrl_[y_col]
                    ctrl_x = df_ctrl_[X_cols + [t_col]]

                if l2k_ps_weight:
                    ps_ = np.clip(
                        d_model_.predict_proba(ctrl_x)[:,1],
                        eps,1-eps
                    )
                    l2k_.fit(ctrl_x, ctrl_y, sample_weight=ps_/(1-ps_))
                else:
                    l2k_.fit(ctrl_x, ctrl_y)

                hat_c = l2k_.predict(test_df[X_cols + [t_col]])

            # ---- 3) 影響関数 ψ_core ----
            Y_te = test_df[y_col].values
            T_te = test_df[t_col].values
            D_te = test_df[d_col].values
            res_ = Y_te - hat_c

            psi_core = (
                (T_te - lamda_hat)*res_*(D_te - ghat)
                / ((1 - ghat)*lamda_hat*(1 - lamda_hat)*p_hat)
            )

            # ---- G_{2p}, G_{2λ} = -θ / p, -θ / λ (簡易形) ----
            G2p = - theta_hat / p_hat
            G2lambda = - theta_hat / lamda_hat

            partial_if = psi_core \
                       + G2p*(D_te - p_hat) \
                       + G2lambda*(T_te - lamda_hat)

            partial_if_list.append(partial_if)

        # 全foldの IF を concat
        partial_if_arr = np.concatenate(partial_if_list)
        # 分散推定
        var_est = np.mean(partial_if_arr**2)

        return var_est