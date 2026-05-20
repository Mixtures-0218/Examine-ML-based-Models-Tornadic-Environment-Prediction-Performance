# Examine-ML-based-Models-Tornadic-Environment-Prediction-Performance
*Code Contributers: Yiding Luo, Shetian Xia, and Royce Lee*

All the necessary code for Evaluation of Machine-Learning Weather Models in Reproducing Mesoscale Tornado-Favorable Convective Environments 

## Compute Tornado-Relevant Parameteres
Run `tornado_parameters.py` to get the parameters, like CAPE, K Index, SRH, etc., once all the necssary output data was obtained
Please change the varaible `models` and `times` to control which model and which event time you want to compute
The result folder should named by `YYYY-MM-DD-HHUTC` under the same path of `tornado_parameters.py`, and the structure is listed below
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
*The other models' results should have the similar structure as fengwu_v1 shown above*
The parameter output files will be stored in the correspondent `YYYY-MM-DD-HHUTC` folder

## Get Metrics such as POD, SR, FSS, etc.
Run script `Scores.py` with intended varaibles `models`, `times`, and `parameters`.
The resulted files will be stored in the correspondent `YYYY-MM-DD-HHUTC` folder with a sub folder named `scores`
The script `Score_avg.py` will average out the scores for the selected times and compute the spread, and the result will be several `.csv` files stored in the `avg_scores` folder

## Performance Diagram
Run script `performance_diagram.py` to get the diagram for a single target time, single model, and single tornado-Relevant paramter by change the variables `time`, `model`, and `parameter`
Run script `performance_diagram_avg.py` to get the diagram for a single parameter, a single model, but with averaged target times by change the variables `model`, and `parameter`. This script only works if the `avg_scores` folder exist.
Run script `performance_diagram_cross_avg.py` to get the diagram for multiple models, parameter with averaged target times. The function paramter `draw_ellipse` controls whether to draw the 1 $\sigma$ spread on the graph or not

## Fractional Skill Score
Run script `FSS_avg_plot.py` to plot graphs for multiple paramters, models with averaged target times. The result graph will be stored in the `avg_fss` folder

## Method of Object-based Diagnosis Evaluation
Run script `MODE.py` with intended varaibles `models`, `times`, and `parameters`.
The resulted files will be stored in the correspondent `YYYY-MM-DD-HHUTC` folder with a sub folder named `MODE`
The script `MODE_avg.py` will average out the scores for the selected times and compute the spread, and the result will be several `.csv` files stored in the `avg_scores` folder
