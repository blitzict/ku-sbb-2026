#!/bin/bash
#waits for the try3 runner to exit, then publishes the trained model if the run validated
cd /c/Users/Blitz/Downloads/Code/claude/ai
W=B:/AI_cpu_087910_20261108/work
while tasklist //FI "PID eq 78404" | grep -q python; do sleep 60; done
echo "runner exited $(date)"
if grep -q "^local AUC" cpu_088_rebuild/train_gpu.log && [ -f $W/cpu_validation.json ]; then
  .venv/Scripts/python.exe -B -m pipeline.publish_try3 --work $W
else
  echo "run did not finish cleanly; not publishing"; tail -5 cpu_088_rebuild/train_gpu.log
  exit 1
fi
