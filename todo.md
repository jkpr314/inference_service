


1. review train/predict process data parts - find a way to unify. 
https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features
2. move train configuration outside of script
3. move predict configuration outside of script

4. Create model package class that encapsulates
   1. data transformation
   2. getting data from model registry
   3. loading model 

5. add model registry
   1. product version - how to track product version? 
   2. data version - how to track data version?
   3. how to track model version? 
   
6. add sqlite as db