# bytedance-icme2019

## Info
https://www.biendata.com/competition/icmechallenge2019/


## Config
```python 3.6.6```

```tensorflow 1.11.0```

```Keras 2.2.4```

```deepctr 0.3.1```


## Log


2019-2-28

- Based on [deepctr](https://deepctr-doc.readthedocs.io/en/latest/)
- Try AutoInt Model
- f, l = [0.708724421609, 0.911346540388]

2019-3-10

- Based on [deepctr](https://deepctr-doc.readthedocs.io/en/latest/)
- Try xDeepFM Model
- Use ```reduce_mem_usage``` to reduce memory that pandas use (reduce by more than 50%)
- Add audio features
- f, l = [0.7054214385161377, 0.91909530488006386]

2019-3-11

- Based on [deepctr](https://deepctr-doc.readthedocs.io/en/latest/)
- Try xDeepFM Model
- Train and test chunk by chunk(use ```DataFrame.get_chunk()```).In the process of training the model, the loss curve is jagged(Loss has been slowly decreasing, but when changing chunks, it suddenly increases), perhaps the model is overfitting.
- f, l = [0.70538118358974211, 0.91660164186399473]