import sys

from flask import Flask, render_template, request, redirect, Response
import random, json

app = Flask(__name__)

@app.route("/")
def output():
	return render_template("index.html")


@app.route('/signUpUser', methods=['POST'])
def signUpUser():
	user = request.form['username']
	password = request.form['password']

	return json.dumps({'status':'OK', 'User':user, 'pass':password});


if __name__ == "__main__":
	app.run("192.168.1.102", "5010")
