# Deep Learning Cloud ML Engine Walk Through

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
pip install -r ../requirements.txt
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

- Launch TensorBoard:

```console
tensorboard --logdir=$MODEL_DIR
```

- When you have started running TensorBoard, you can access it in your browser at http://localhost:6006

### Training in the cloud

#### Set up your Cloud Storage bucket

This section shows you how to create a new bucket. You can use an existing bucket, but if it is not part of the project you are using to run AI Platform, you must explicitly [grant access to the AI Platform service accounts](https://cloud.google.com/ml-engine/docs/tensorflow/working-with-cloud-storage#setup-different-project).

1. Specify a name for your new bucket. The name must be unique across all buckets in Cloud Storage.

   ```console
   BUCKET_NAME="your_bucket_name"
   ```

   For example, use your project name with `-mlengine` appended:

   ```console
   PROJECT_ID=$(gcloud config list project --format "value(core.project)")
   BUCKET_NAME=${PROJECT_ID}-mlengine
   ```

2. Check the bucket name that you created.

   ```console
    echo $BUCKET_NAME
   ```

3. Select a region for your bucket and set a REGION environment variable.

   For example, the following code creates REGION and sets it to `us-central1`:

   ```console
    REGION=us-central1
   ```

4. Create the new bucket:

   ```console
    gsutil mb -l $REGION gs://$BUCKET_NAME
   ```

   Note: Use the same region where you plan on running AI Platform jobs. The example uses us-central1 because that is the region used in the getting-started instructions.

#### Upload the data files to your Cloud Storage bucket.
