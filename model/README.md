## Abstract
`DOTA` `RoI-Transformer`
<br/><br/>

## Result
|Backbone|Multi-Scale + Random Rotation|Augmentation|Loss|F1-Score|mAP|
|---|---|---|---|---|---|
|ResNet50|<div align="center">✓</div>||<div align="center">0.5555</div>|<div align="center">0.7320</div>|<div align="center">0.7618</div>|
|ResNet101|<div align="center">✓</div>||<div align="center">0.3342</div>|<div align="center">0.7432</div>|<div align="center">0.7624</div>|
|ResNet101|<div align="center">✓</div>|<div align="center">✓</div>|<div align="center">0.2915</div>|<div align="center">0.7450|<div align="center">0.8110</div>|

<br/><br/><br/>
## Final Model Performance
||Before Improvement|After Improvement|
|---|---|---|
|Vehicle|Recall|0.868|0.840|
||Precision|0.624|0.560|
||F1-Score|0.726|0.672|
|Ship|Recall|0.761|0.801|
||Precision|0.539|0.523|
||F1-Score|0.631|0.644|
|Airplane|Recall|0.876|0.969|
||Precision|0.805|0.892|
||F1-Score|0.839|0.929|
|mAP|0.762|0.811|
  
<br/><br/>
||Before Improvement|After Improvement|
|---|---|---|
|Mean Recall|0.835|0.870|
|Mean Precision|0.656|0.658|
|Mean F1 Score|0.732|0.745|
|Mean AP|0.762|0.811|
