# deep-learning-ml-engine-walk

```console
virtualenv cmle-env --python=python2.7

source cmle-env/bin/activate
```

```console
TRAIN_DATA=$(pwd)/data/iris_training.csv
```

```console
EVAL_DATA=$(pwd)/data/iris_test.csv
```

```console
MODEL_DIR=output
```

```console
gcloud ai-platform local train \
    --module-name trainer.task \
    --package-path trainer/ \
    --job-dir $MODEL_DIR \
    -- \
    --train-files $TRAIN_DATA \
    --eval-files $EVAL_DATA \
    --train-steps 1000 \
    --eval-steps 100
```

```console
tensorboard --logdir=$MODEL_DIR
```
