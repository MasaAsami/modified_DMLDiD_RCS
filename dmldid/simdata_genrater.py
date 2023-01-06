import pandas as pd
import numpy as np


def generate_simdata_ro(
    true_att=3, _n=100, _dim=200, base_seed=1, xd_unobserved=False
) -> pd.DataFrame:
    np.random.seed(seed=base_seed)
    _gamma = [1 / (_i + 1) if _i < 5 else 0 for _i in range(_dim)]
    _beta = [-1 / (_i + 1) + 5 if _i < 5 else 0 for _i in range(_dim)]
    pre_beta = [2 / (_i + 1)  if _i < 5 else 0 for _i in range(_dim)]

    X = np.random.normal(0, 1, size=(_n, _dim))
    _df = pd.DataFrame(X)
    _df.columns = [f"x{i}" for i in range(_dim)]

    if xd_unobserved:
        _df["latent_xd"] = np.random.uniform(-1, 1, size=_n)
        for _i in range(5):
            _df[f"x{_i}"] += _df["latent_xd"] * (1 / (_i + 2))
        _gamma.append(-1.0)
        _beta.append(0)
        pre_beta.append(0)
        _X = _df.values
        log_odds = np.dot(_X, _gamma) + np.random.normal(0, 1, size=_n)
        ps = 1 / (1 + np.exp(-log_odds))
        del _df["latent_xd"]

        _df = _df.assign(
            D=np.random.binomial(1, ps),
            preY=np.dot(_X, pre_beta) + np.random.normal(0, 1, size=_n) - 5,
            pre_post_diff=np.dot(_X, _beta) + np.random.normal(0, 1, size=_n),
        )
    else:
        log_odds = np.dot(X, _gamma) + np.random.normal(0, 1, size=_n)
        ps = 1 / (1 + np.exp(-log_odds))
        _df = _df.assign(
            D=np.random.binomial(1, ps),
            preY=np.dot(X, pre_beta) + np.random.normal(0, 1, size=_n) - 5,
            pre_post_diff=np.dot(X, _beta) + np.random.normal(0, 1, size=_n),
        )
    _df["postY"] = _df.eval("preY + pre_post_diff")
    _df["postY"] = np.where(_df["D"] > 0, _df["postY"] + true_att, _df["postY"])

    del _df["pre_post_diff"]

    return _df


def convert_rcs(_df) -> pd.DataFrame:
    np.random.seed(seed=1)
    _df["y_flag"] = np.random.binomial(1, 0.5, size=len(_df))
    _df["Y"] = _df.apply(lambda x: x["postY"] if x["y_flag"] > 0 else x["preY"], axis=1)
    _df["T"] = _df.apply(lambda x: 1 if x["y_flag"] > 0 else 0, axis=1)
    return _df.drop(["preY", "postY", "y_flag"], axis=1)