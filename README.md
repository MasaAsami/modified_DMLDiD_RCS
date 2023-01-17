# modifiedDMLDiD
Modifying DMLDiD :Double/debiased machine learning for difference-in-differences models.

# previous work
## main paper
[1] Chang, Neng-Chieh. (2020). [Double/debiased machine learning for difference-in-differences models](https://academic.oup.com/ectj/article/23/2/177/5722119#247745047). The Econometrics Journal 23.2 : 177–191
- This paper proposes the following two models:
  - repeated outcome (dmldid_ro)：
    - I believe this to be a theoretically valid and excellent achievement.
    - However, there are several deficiencies in the implementation of Chang (2020). In addition, the simulation data in the paper were not suitable to demonstrate this theory. These points have been corrected and presented in [My blog](https://medium.com/@masa_asami/double-debiased-ml-for-did-1-fd08bebcf033).
  - repeated cross-section  (dmldid_rcs):
    - I do not believe this model is adequate. This repository will examine and attempt to modify this.

## sub papers
[2] Abadie A. (2005). Semiparametric difference-in-differences estimators, Review of Economic Studies, 72, 1–19.
- References in Chang (2020)
- Proposes an IPW estimator for DiD.

[3] Chernozhukov V., D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W. Newey, J. Robins (2018). Double/debiased machine learning for treatment and structural parameters, Econometrics Journal, 21, C1–C68.
- original DML

# What is wrong with dmldid_rcs ?
## dmldid_rcs's ATT in Chang (2020)

- Chang (2020) defines the ATT estimator for repeated cross-section data as follows:
- Y : outcome
- T : binary value (post =1 / pre = 0)
- D : binary value (treated group =１/ control group = 0)
- X : Covariates
- p̂k : D's average（cross-fitting）
- λ̂k : T's average（cross-fitting）
- ĝk(X) : propensity socre（cross-fitting）

<img width="611" alt="image" src="https://user-images.githubusercontent.com/16971400/199974792-be66c6e3-19b5-4982-aff5-4763674134ce.png">

- l2k is the following ML model (Chang(2020) uses Lasso, but essentially any ML is OK)
  - Supervised label：(Ti−λ̂k)Yi
  - features：covariates X（X with arbitrary transformations as q）
  - training data：untreated data only 
  - with cross-fitting
  
![image](https://user-images.githubusercontent.com/16971400/198495736-bc34e71b-a1bd-443f-ab83-2347c9ab0f28.png)


## l2k is an unrealistic prediction task
- The label is positive if T=1 and negative if T=0. The sign is easily reversed by a change in T alone.
- On the other hand, X(q) does **not** necessarily contain time-dependent variables.　And even if it does contain such variables, it is difficult to properly predict the above label with a linear model (Lasso) such as Chang (2020)
  - For example, if the covariates are not time-dependent (e.g., demographic information such as male/female, race, etc.), prediction of this label is not possible (although it can be learned = no error occurs, but the prediction is meaningless)
  - Even if the covariates include time-dependent variables, a large number of time-dependent *time-independent interaction terms must be thrown in. Still, it is almost impossible to predict the label such that it flips 180 degrees positive or negative due to unobserved variables (for l2k, T is unobserved)
- The (Ti-λ̂k) part need not be included in the prediction task. It should only be designed to estimate the latent outcome of Y

<img src="https://github.com/MasaAsami/modified_DMLDiD_RCS/blob/main/fig/covariate_to_label.png" alt="l2k label plot" title="l2k label plot">

- the notebook on this issue is [here](https://github.com/MasaAsami/modified_DMLDiD_RCS/blob/main/notebooks/Why_original_l2kmodel_is_wrong.ipynb)

# My solution

- First, divide l2k into two parts l2k_t1, l2k_t0 are arbitrary machine learning models
![\hat{\ell }_{2k}^{t } = E[Y_i   |  T=t,D=0, X_i]](https://render.githubusercontent.com/render/math?math=%5Ccolor%7Bwhite%7D%5Clarge+%5Cdisplaystyle+%5Chat%7B%5Cell+%7D_%7B2k%7D%5E%7Bt+%7D+%3D+E%5BY_i+++%7C++T%3Dt%2CD%3D0%2C+X_i%5D)
  - l2k_t1 = E[Y | T=1, D=0, X] 
  - l2k_t0 = E[Y | T=0, D=0, X]
- With this l2k_t1, l2k_t0, the ATT can be modified as follows.
  - Doubly robust in propensity score and l2k (outcome model) is achieved.
  - cross-fitting

<img width="546" alt="image" src="https://user-images.githubusercontent.com/16971400/199974910-cdb70ede-9f2f-4613-b4c2-230d1426967c.png">

# Simulation result
## simulation data
The following simulation data were created:
- repeated cross-section data
- true ATT := 3
- dim(X) := 10
- N = 500
## result: modified-DMLDiD is better than Chang(2020)
- The notebook is [here](https://github.com/MasaAsami/modified_DMLDiD_RCS/blob/main/notebooks/DMLDiD_RCS_with_SIMDATA.ipynb).
- ture ATT = 3


<img src="https://github.com/MasaAsami/modified_DMLDiD_RCS/blob/main/fig/simdata_result_unobserved_ps.png" alt="lasso dmldid" title="lasso dmldid">

