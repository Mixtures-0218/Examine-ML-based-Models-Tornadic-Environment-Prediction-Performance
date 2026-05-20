# Examine-ML-based-Models-Tornadic-Environment-Prediction-Performance
*Code Contributers: Yiding Luo, Shetian Xia, and Royce Lee*

All the necessary code for Evaluation of Machine-Learning Weather Models in Reproducing Mesoscale Tornado-Favorable Convective Environments 

## Compute Tornado-Releavent Parameteres
Run `tornado_parameters.py` to get the parameters, like CAPE, K Index, and SRH, etc., once all the necssary output data was obtained
Please change the varaible `models` and `times` to control which model and which event time you want to compute
The result folder should named by 'YYYY-MM-DD-HHUTC', and the structure is listed below
```plain
├── root
│   ├── YYYY-MM-DD-HHUTC
        │   ├── real.nc
        │   ├── real_surface.nc
        │   ├── fengwu_v1
            │   ├── fengwu_v1_24.nc
            │   ├── fengwu_v1_48.nc
            │   ├── fengwu_v1_72.nc
            │   ├── fengwu_v1_120.nc
            │   ├── fengwu_v1_168.nc
        │   ├── fengwu_v2
        │   ├── pangu
        │   ├── fuxi
```
*The other models' results should have the similar structure as fengwu_v1 shown above
