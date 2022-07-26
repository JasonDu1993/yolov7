#!/bin/bash
VAL=$1
#VAL="yolo_for_gesture"
#ps aux | grep ${VAL} | grep -v grep
kill $(ps aux | grep ${VAL} | grep -v grep | awk '{print $2}')
