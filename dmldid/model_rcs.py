import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone
from sklearn.linear_model import LogisticRegressionCV, LassoCV
from IPython.display import clear_output

class DMLDiD_RCS:
    """
    Modified DMLDiD for RCS with 2-fold cross-fitting.
    Adds per-observation influence scores and a multiplier bootstrap for SE/CI.
    """

    def __init__(
        self,
        d_model=LogisticRegressionCV(cv=5, random_state=333, penalty="l1", solver="saga"),
        l2k_model=LassoCV(cv=5, random_state=333),
        **kwargs,
    ):
        # 点推定とブート用に保存するもの
        self._att_list = []
        self.theta_hat = None      # 点推定
        self.psi_ = None           # 各観測 i の影響関数 ψ_i
        self.n_ = None             # サンプルサイズ

        # 学習器
        self.d_model = d_model
        self.l2k_model = l2k_model

    @staticmethod
    def _dlambda(T, lam):
        """d/dλ {(T-λ)/[λ(1-λ)]}  (T∈{0,1}) を要素ごとに計算"""
        return (-(1.0/(lam*(1.0-lam)))
                - ((T - lam) * (1.0 - 2.0*lam)) / (lam**2 * (1.0 - lam)**2))

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
        l2k_ps_weight=False,
        **kwargs,
    ):
        """
        dmldid=True  : 残差型・直交スコア（Modified DMLDiD）
        dmldid=False : Abadie-type（PSのみ）
        推定後に各観測の影響関数 self.psi_ を保存（→ multiplier_bootstrap に使用）
        """
        self._att_list = []
        self.theta_hat = None
        self.psi_ = None
        self.n_ = len(df)
        df = df.reset_index(drop=True).copy()

        K = 2  # 2-fold
        rng_base = base_random_seed

        for l in range(sim_cnt):
            if progress_plot and (l > 0):
                clear_output(wait=True)
                print(f"{l}. current att mean: ", np.mean(self._att_list))

            if dmldid:
                # ---------------- Modified DMLDiD（クロスフィット） ----------------
                df_set = train_test_split(
                    df, random_state=rng_base + l, test_size=0.5, stratify=df[[t_col, d_col]]
                )
                temp_att = []
                # ベーススコア項 s_i（後で ψ_i を作る素材）
                s_vec = np.full(len(df), np.nan)

                fold_info = []
                for i in range(K):
                    k = 0 if i == 0 else 1  # 予測に使う fold
                    c = 1 if i == 0 else 0  # 学習に使う fold

                    # --- g(X) ---
                    d_model = clone(self.d_model)
                    if d_model_t0_only:
                        d_model.fit(
                            df_set[c].query(f"{t_col}<1")[X_cols],
                            df_set[c].query(f"{t_col}<1")[d_col],
                        )
                    else:
                        d_model.fit(df_set[c][X_cols], df_set[c][d_col])

                    ghat_k = np.clip(
                        d_model.predict_proba(df_set[k][X_cols])[:, 1],
                        eps, 1 - eps,
                    )

                    lam_k = df_set[k][t_col].mean()
                    p_k = df_set[k][d_col].mean()

                    # --- ℓ2 (対照群アウトカム) ---
                    if l2k_model_separate:
                        l2_t1 = clone(self.l2k_model)
                        l2_t0 = clone(self.l2k_model)

                        ctrl = (df_set[c][d_col] < 1)
                        y_ctrl = df_set[c].loc[ctrl, y_col]
                        X_ctrl = df_set[c].loc[ctrl, X_cols]
                        idx_t1 = df_set[c].loc[ctrl & (df_set[c][t_col] > 0)].index
                        idx_t0 = df_set[c].loc[ctrl & (df_set[c][t_col] < 1)].index

                        if l2k_ps_weight:
                            ps1 = np.clip(d_model.predict_proba(X_ctrl.loc[idx_t1])[:, 1], eps, 1-eps)
                            l2_t1.fit(X_ctrl.loc[idx_t1], y_ctrl.loc[idx_t1], sample_weight=ps1/(1-ps1))
                            ps0 = np.clip(d_model.predict_proba(X_ctrl.loc[idx_t0])[:, 1], eps, 1-eps)
                            l2_t0.fit(X_ctrl.loc[idx_t0], y_ctrl.loc[idx_t0], sample_weight=ps0/(1-ps0))
                        else:
                            l2_t1.fit(X_ctrl.loc[idx_t1], y_ctrl.loc[idx_t1])
                            l2_t0.fit(X_ctrl.loc[idx_t0], y_ctrl.loc[idx_t0])

                        X_pred = df_set[k][X_cols]
                        t_pred = df_set[k][t_col].to_numpy()
                        l2_hat_post = l2_t1.predict(X_pred)
                        l2_hat_pre  = l2_t0.predict(X_pred)
                        l2_hat_k = l2_hat_post * t_pred + l2_hat_pre * (1 - t_pred)
                    else:
                        l2 = clone(self.l2k_model)
                        ctrl = (df_set[c][d_col] < 1)
                        y_ctrl = df_set[c].loc[ctrl, y_col]
                        XT_ctrl = df_set[c].loc[ctrl, X_cols + [t_col]]
                        if l2k_ps_weight:
                            ps = np.clip(d_model.predict_proba(XT_ctrl[X_cols])[:, 1], eps, 1-eps)
                            l2.fit(XT_ctrl, y_ctrl, sample_weight=ps/(1-ps))
                        else:
                            l2.fit(XT_ctrl, y_ctrl)
                        l2_hat_k = l2.predict(df_set[k][X_cols + [t_col]])

                    # --- スコア s_i と θ_k ---
                    yk = df_set[k][y_col].to_numpy()
                    tk = df_set[k][t_col].to_numpy()
                    dk = df_set[k][d_col].to_numpy()
                    idxk = df_set[k].index.to_numpy()

                    numer_t = (tk - lam_k) / (lam_k * (1.0 - lam_k))
                    numer_d = (dk - ghat_k) / (p_k * (1.0 - ghat_k))
                    s_k = numer_t * numer_d * (yk - l2_hat_k)

                    theta_k = float(np.mean(s_k))
                    temp_att.append(theta_k)
                    s_vec[idxk] = s_k

                    # IF のための導関数（foldごと）
                    G2p_k = - theta_k / p_k
                    dlam = self._dlambda(tk, lam_k)
                    G2lam_k = float(np.mean(numer_d * dlam * (yk - l2_hat_k)))

                    fold_info.append(dict(k=k, p_k=p_k, lam_k=lam_k,
                                          G2p_k=G2p_k, G2lam_k=G2lam_k))

                theta_hat = float(np.mean(temp_att))
                self.theta_hat = theta_hat
                self._att_list.append(theta_hat)

                # 観測 i ごとの ψ_i を組み立て（所属 fold の係数を使う）
                psi = np.full(len(df), np.nan)
                for fi in fold_info:
                    k = fi["k"]
                    idxk = df_set[k].index.to_numpy()
                    p_k  = fi["p_k"];  lam_k = fi["lam_k"]
                    G2p_k = fi["G2p_k"];  G2lam_k = fi["G2lam_k"]
                    psi[idxk] = (s_vec[idxk] - theta_hat
                                 + G2p_k * (df_set[k][d_col].to_numpy() - p_k)
                                 + G2lam_k * (df_set[k][t_col].to_numpy() - lam_k))
                # 数値安定のため中心化
                psi = psi - np.nanmean(psi)
                self.psi_ = psi

            else:
                # ---------------- Abadie-type（PSのみ） ----------------
                d_model = clone(self.d_model)
                d_model.fit(df[X_cols], df[d_col])
                ghat = np.clip(d_model.predict_proba(df[X_cols])[:, 1], eps, 1 - eps)

                p_hat = df[d_col].mean()
                lam_hat = df[t_col].mean()

                y = df[y_col].to_numpy()
                t = df[t_col].to_numpy()
                d = df[d_col].to_numpy()

                numer_t = (t - lam_hat) / (lam_hat * (1.0 - lam_hat))
                numer_d = (d - ghat) / (p_hat * (1.0 - ghat))
                s_i = numer_t * numer_d * y

                theta_hat = float(np.mean(s_i))
                self.theta_hat = theta_hat
                self._att_list.append(theta_hat)

                G2p = - theta_hat / p_hat
                dlam = self._dlambda(t, lam_hat)
                G2lam = float(np.mean(numer_d * dlam * y))

                psi = (s_i - theta_hat
                       + G2p * (d - p_hat)
                       + G2lam * (t - lam_hat))
                psi = psi - np.mean(psi)
                self.psi_ = psi

        return self

    def att(self):
        """点推定（_att_list の平均）"""
        if not self._att_list:
            return None
        return float(np.mean(self._att_list))

    def sim_att_result(self):
        return self._att_list

    def multiplier_bootstrap(self, B=999, kind="rademacher", random_state=1):
        """
        乗数ブートストラップ：ψ_i に乱数重み ξ_i を掛けて
        θ* = θ̂ + n^{-1/2} (1/n) Σ ξ_i ψ_i を B 回生成。
        戻り値: (se, (ci_low, ci_high), theta_star_array)
        """
        if self.psi_ is None or self.theta_hat is None:
            raise RuntimeError("Call fit() first to compute psi_ and theta_hat.")
        n = self.n_
        rng = np.random.default_rng(random_state)

        if kind.lower() in ("rademacher", "rad"):
            xi = rng.choice([-1.0, 1.0], size=(B, n))
        elif kind.lower() in ("normal", "gaussian"):
            xi = rng.standard_normal(size=(B, n))
        else:
            raise ValueError("kind must be 'rademacher' or 'normal'")

        shifts = (xi @ self.psi_) / (n ** 1.5)  # = n^{-1/2} * (1/n) Σ ξ_i ψ_i
        theta_star = self.theta_hat + shifts
        se = float(np.std(theta_star - self.theta_hat, ddof=1))
        ci = (float(np.quantile(theta_star, 0.025)), float(np.quantile(theta_star, 0.975)))
        return se, ci, theta_star
