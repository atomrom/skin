#!/bin/sh

echo Cleaning previous export and datasets
cd /opt/dh/retrain/
rm -r dataset/train/* dataset/test/*

NOW=`date +"%d-%m-%Y_%T"`

# 4 weeks
TEST_PERIOD_START=`date --date="56 days ago" +"%d-%m-%Y_00:00:00"`
TEST_PERIOD_END=$NOW
# 4+4 weeks
TRAINING_PERIOD_START=`date --date="112 days ago" +"%d-%m-%Y_00:00:00"`
TRAINING_PERIOD_END=$TEST_PERIOD_START

export CLASSPATH=postgresql-42.2.5.jar:json-simple-1.1.1.jar:.
java CreateDatasets dataset $TRAINING_PERIOD_START $TRAINING_PERIOD_END $TEST_PERIOD_START $TEST_PERIOD_END

echo
echo Retrain
ID=$NOW
echo ID: $ID
mkdir models/$ID
python3 retrain.py --dataset_dir=dataset --baseline_model_path=models/baseline.model $ID
retrain_exit_code="$?"

if [ $retrain_exit_code -eq 0 ]
then
  echo Setting baseline model to the newly trained model: models/$ID/$ID.model
  rm models/baseline.model
  rm models/baseline_stats.json
  ln -s $ID/$ID.model models/baseline.model
  ln -s $ID/${ID}_stats.json models/baseline_stats.json

  echo Is the new model better than the current best one?
  python3 is_better.py --old_stats=models/best_stats.json --new_stats=models/baseline_stats.json
  is_better="$?"
  if [ $is_better -eq 1 ]
  then
    echo New model is better! Updating the links to the new model!
    rm models/best.model
    rm models/best_stats.json
    ln -s baseline.model models/best.model
    ln -s baseline_stats.json models/best_stats.json    
    echo Links updated.
  else
    echo New model is not better than the current best one! Not doing anything!
  fi
else
  echo Retrain failed
fi
