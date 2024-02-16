

###_gt.p files structure:
[{'Camera': T_wc, 'Object_1': [T_co, T_co, T_co...], 'Object_2': [T_co, T_co, T_co...], ...}  
.  
.  
.]  
T_xx = np.array(), shape = (4, 4)

estimate_progress files structure:
[{'Object_1': [{'T': T_wo, Q: Q_oo}, {'T': T_wo, Q: Q_oo}...],
 'Object_2': [{...}, {...} ...]
...
}
.
.
.
]
T_xx = np.array(), shape = (4, 4)
Q_oo = np.array(), shape = (6, 6)


```
frames_prediction.p structure:

[{Object_1': [T_co, T_co, T_co...], 'Object_2': [T_co, T_co, T_co...], ...}  
.  
.  
.]  
T_xx = np.array(), shape = (4, 4)
```