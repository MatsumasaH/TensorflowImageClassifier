#  Tensorflow Image Classifier
## Overview

Sample Commands
```
python wide_deep.py --model_type=wide
python wide_deep.py --model_type=deep --export_dir=wide_deep_saved_model
tensorboard --logdir=csv_model
saved_model_cli show --dir=wide_deep_saved_model/1529182697/
saved_model_cli show --dir=wide_deep_saved_model/1529182697/ --tag_set serve --all
saved_model_cli run --dir=wide_deep_saved_model/1529182697/ --tag_set serve --signature_def="predict" --input_examples='examples=[{"width":[100.], "height":[100.]}]'
```
