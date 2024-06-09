plotting_path = 'C:/data/plots/'

SEEDS = [0, 42, 1337, 24601, 8675309]



MODEL_NAME_OLW_WA = 'OLR_WA'
MODEL_NAME_SGD = 'SGD'
MODEL_NAME_MBGD = 'MBGD'
MODEL_NAME_LMS = 'LMS'
MODEL_NAME_ORR = 'ORR'
MODEL_NAME_OLR = 'OLR'
MODEL_NAME_RLS = 'RLS'
MODEL_NAME_PA = 'PA'

OLR_WA_EXP14_FORMATTED_PARAMETERS = r" [$W_{base}=0.5$, $W_{inc}=0.5$, $BK=N \times 0.01$, $K=M \times " \
                                    r"(\mathbb{Z}^{+} = 5)$]"
SGD_EXP14_FORMATTED_PARAMETERS = r" [$\eta=0.01$, E=N $\times (\mathbb{Z}^{+} = 2)$]"
MBGD_EXP14_FORMATTED_PARAMETERS = r" [$\eta=0.01$, K=M $\times (\mathbb{Z}^{+} = 5)$, E=$\frac{N}{K} \times " \
                                  r"(\mathbb{Z}^{+} = 5)$]"
LMS_EXP14_FORMATTED_PARAMETERS = r" [$\eta=0.01$]"
ORR_EXP14_FORMATTED_PARAMETERS = r" [$\eta=0.01$, E = N $\times (\mathbb{Z}^{+} = 2)$,  $\lambda = 0.1$]"
OLR_EXP14_FORMATTED_PARAMETERS = r" [$\eta=0.01$, E = N $\times (\mathbb{Z}^{+} = 2)$,  $\lambda = 0.1$]"
RLS_EXP14_FORMATTED_PARAMETERS = r" [$\lambda = 0.99$, $\delta=0.01$]"
PA_EXP14_FORMATTED_PARAMETERS = r" [$C=0.1$, $\epsilon=0.1$]"


