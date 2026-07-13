"""Session-level infestation decision logic, built on the classifier's raw per-window scores.

Pure stdlib + NumPy — no TFLite/librosa/tkinter imports, so this stays fast to test and
reusable from both the offline evaluation scripts and (eventually) main.py.
"""
