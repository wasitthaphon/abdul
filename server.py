import sys

from flask import Flask, render_template, request, redirect, Response
import numpy as np 
import json
from resources import MLP_BP_PUB
from resources import SVM_PUB


app = Flask(__name__)

def compute(raw_data):

	# Return [STATUSE, BAD_WORD_PERCENT, GOOD_WORD_PERCENT, RESULT]
	data = MLP_BP_PUB.find_word(raw_data)
	data = data[0]
	new_data = data[:-1].copy()
	new_data = SVM_PUB.normalize_data(new_data)
	result = MLP_BP_PUB.predict(MLP_BP_PUB.load_weights(), new_data)
	svm_res = SVM_PUB.svm_test(new_data)
	print(new_data)
	text = ['เชิงลบ', 'เชิงบวก', 'ปกติ']
	return json.dumps({'status':'OK', 'data':data, 'result_mlp':text[result], 'result_svm':text[svm_res]})

@app.route("/")
def output():
	return render_template('index.html')

@app.route('/go_compute', methods=['POST'])
def go_compute():
	if request.method == 'POST':
		return compute(request.form['message'])
	else:
		return None

if __name__ == '__main__':
	# to run this python server.py <IP>
	app.run(sys.argv[1], '5010')