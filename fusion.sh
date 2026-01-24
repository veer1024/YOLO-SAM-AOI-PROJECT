# show matching lines + line numbers
grep -n "fragment_fusion" yolo_detector.log

# show ~60 lines around the match (pick the line number you saw from grep)
# example: if grep says it's at line 5231:
nl -ba ml/yolo_detector.log | sed -n '5200,5260p'

# or: show 40 lines before/after the match (no line numbers)
grep -n "fragment_fusion" -n yolo_detector.log
grep -n "fragment_fusion" yolo_detector.log | head
