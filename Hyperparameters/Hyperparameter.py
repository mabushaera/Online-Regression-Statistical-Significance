# Gradient Descent
gd_learning_rate = 0.01
gd_learning_rate2 = 0.001
gd_learning_rate3 = 0.0001

# Stochastic Gradient Descent
gd_stochastic_epochs = lambda n_samples, factor: int(n_samples * factor)

# Mini-batch Gradient Descent
gd_mini_batch_epochs = lambda n_samples, n_features, factor: int(n_samples / (n_features * 5) * factor)
gd_mini_batch_size = lambda n_features, user_defined_val: int(max(user_defined_val, (n_features + 1) * 5))
gd_mini_batch_size2 = lambda n_features, user_defined_val: int(max(user_defined_val, (n_features) * 5))

# Widrow-Hoff (LMS)
wf_learning_rate = 0.01
wf_learning_rate2 = 0.001
wf_learning_rate3 = 0.0001

# Online Ridge Regression and Online Lasso Regression
ridge_lasso_learning_rate = 0.01
ridge_lasso_learning_rate2 = 0.001
ridge_lasso_epochs = lambda n_samples, factor: int(n_samples * factor)
ridge_lasso_regularization_param = 0.1
ridge_lasso_regularization_param2 = 0.01
ridge_lasso_regularization_param4 = 0.001
ridge_lasso_regularization_param5 = 0.0001
ridge_lasso_regularization_param3 = 1e-10

# Recursive Least Squares (RLS)
rls_lambda_ = .99
rls_lambda_2 = .18
rls_delta = .01

# Accurate Online Support Vector regression (AOSVR)
osvr_C = 10
osvr_eps = 0.1
osvr_kernelParam = 30
osvr_bias = 0

# Online Passive-Aggressive (PA)
pa_C = .1  # reg param
pa_epsilon = .1  # epsilon is a positive parameter which controls the sensitivity
# to prediction mistakes. used to compute the loss.
pa_C2 = 0.01
pa_epsilon2 = 0.01

pa_C3 = 0.0001



# OLR_WA
olr_wa_batch_size = 10
olr_wa_w_base = .5  # default value
olr_wa_w_inc = .5  # default_value

olr_wa_w_base1 = .9
olr_wa_w_inc1 = .1

olr_wa_w_base2 = .1
olr_wa_w_inc2 = .9

olr_wa_w_base_adv1 = .1
olr_wa_w_inc_adv1 = 2

olr_wa_w_base_adv2 = 4
olr_wa_w_inc_adv2 = 0.01

olr_wa_base_model_size0 = 1
olr_wa_base_model_size1 = 10
olr_wa_base_model_size2 = 2
olr_wa_increment_size = lambda n_features, user_defined_val: int(max(user_defined_val, (n_features + 1) * 5))
olr_wa_increment_size2 = lambda n_features, user_defined_val: int(max(user_defined_val, (n_features ) * 5))


HYPERPARAMETERS_ALTERNATIVES = {
        "Case1: Equal Weights: w_base = 0.5  w_inc=0.5": {
            "w_base": .5,
            "w_inc": .5
        },
        "Case2: Favor Inc: w_base = 0.1  w_inc=0.9": {
            "w_base": .1,
            "w_inc": .9
        },
        "Case3: Favor Inc Radically: w_base = 0.01  w_inc=4": {
                    "w_base": .01,
                    "w_inc": 4
        },
        "Case4: Favor Base: w_base = 0.9  w_inc=0.1": {
            "w_base": .9,
            "w_inc": .1
        },
        "Case5: Favor Base Radically: w_base = 4  w_inc=0.01": {
            "w_base": 4,
            "w_inc": .01
        }
    }

