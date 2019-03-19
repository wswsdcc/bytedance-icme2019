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
- Add audio features as ```sequence feature``` of the model input.
- f, l = [0.7054214385161377, 0.91909530488006386]

2019-3-11

- Based on [deepctr](https://deepctr-doc.readthedocs.io/en/latest/)
- Try xDeepFM Model
- Train and test chunk by chunk(use ```DataFrame.get_chunk()```).In the process of training the model, the loss curve is jagged(Loss has been slowly decreasing, but when changing chunks, it suddenly increases), perhaps the model is overfitting.
- f, l = [0.70538118358974211, 0.91660164186399473]
  
2019-3-15

- Add title features: 
   - Train xDeepFM on user features and NLP submodel(embedding + LSTM) on title features. 
   - The last layer of NLP submodel is a dense layer, it output a number for every title. 
   - Then the number is added up with b as the input of the predict layer.
- The result is no better than 3-10's

2019-3-19

- Add face features and title features: 
   - Abstract the number of men's faces(named in dataframe as ```'man'```), the number of women's faces(named as ```'woman'```), the average beauty value(named as ```'avg_beauty'```), and the average face position(a four-dimensional vector, named as ```'position_x'```)
   - The face features is add in ```dense_features``` list
   - The title features are treated the same as the model 3-15
- When average face position is feed into the model, it's easy to overfit, so I remove it from the input.
- f, l = [0.70730000991394681, 0.92167217152790604]