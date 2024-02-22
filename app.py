from flask import Flask, render_template, request
import joblib
import os

# Configure Flask app to serve static files
app = Flask(__name__, static_url_path='/static')

# Load the trained machine learning model
model = joblib.load('student mark prediction.pkl')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input parameter (study hours) from the form
        student_hours = float(request.form['student_hours'])

        # Make prediction using the loaded model
        prediction = model.predict([[student_hours]])

        # Render template with prediction result
        return render_template('result.html', prediction=prediction[0])
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
