

import subprocess


model_1_process = subprocess.Popen(['python', 'Rnn/App2.py'])

model_2_process = subprocess.Popen(['python', 'xrayy.py'])

model_3_process = subprocess.Popen(['python', '_sep.py'])

model_1_process.wait()
model_2_process.wait()
model_3_process.wait()