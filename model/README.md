## Abstract
`DOTA` `RoI-Transformer`
<br/><br/><br/>

## Result
|Backbone|Multi-Scale + Random Rotation|Augmentation|Loss|F1-Score|mAP|
|---|---|---|---|---|---|
|ResNet50|<div align="center">✓</div>||<div align="center">0.5555</div>|<div align="center">0.7320</div>|<div align="center">0.7618</div>|
|ResNet101|<div align="center">✓</div>||<div align="center">0.3342</div>|<div align="center">0.7432</div>|<div align="center">0.7624</div>|
|ResNet101|<div align="center">✓</div>|<div align="center">✓</div>|<div align="center">0.2915</div>|<div align="center">0.7400|<div align="center">0.7660</div>|

<br/><br/><br/>
## Final Model Performance
|||Before Improvement|After Improvement|
|---|---|---|---|
|Vehicle|Recall|0.868|0.869|
||Precision|0.624|0.623|
||F1-Score|0.726|0.726|
||AP|0.799|0.799|
|Ship|Recall|0.761|0.777|
||Precision|0.539|0.570|
||F1-Score|0.631|0.657|
||AP|0.670|0.684|
|Airplane|Recall|0.876|0.881|
||Precision|0.805|0.798|
||F1-Score|0.839|0.838|
||AP|0.816|0.817|
  
<br/><br/>
||Before Improvement|After Improvement|
|---|---|---|
|Mean Recall|0.835|0.843|
|Mean Precision|0.656|0.664|
|Mean F1 Score|0.732|0.740|
|Mean AP|0.762|0.766|
