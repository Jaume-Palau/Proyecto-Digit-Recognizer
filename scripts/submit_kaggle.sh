#!/bin/bash

kaggle competitions submit \
  -c conquerx-b05-lec01-digit-recognizer \
  -f outputs/submissions/submission.csv \
  -m "submission from local project"