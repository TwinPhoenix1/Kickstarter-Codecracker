from functions import *
import pandas as pd
from xgboost import XGBClassifier
from flask import Flask, render_template, session, request
from flask_session import Session
import os

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

secret_key = os.urandom(12).hex()
app.secret_key = secret_key

df = pd.read_csv("Kickstarter.csv")
df, df_fit = preprocess_data(df)
X_train, y_train, X_test, y_test, X_val, y_val = split_data(df)
X_train, X_test, transformer = encode_data(X_train, X_test)
X_train, X_test, scaler = scale_data(X_train, X_test)
# model = train_model(X_train, y_train, X_test, y_test)
model = XGBClassifier(max_depth=5, learning_rate=0.15, n_estimators=150, gamma=0.4)
model.fit(X_train, y_train)
pipeline = create_prediction_pipeline(transformer, scaler, model, df_fit)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def generate_prediction():
    data = request.form
    new_instance = pd.DataFrame([{'category_name': data['category_name'], 
                                'currency': data['currency'], 
                                'created_at_weekday': data['created_at_weekday'], 
                                'launched_at_weekday': data['launched_at_weekday'], 
                                'deadline_weekday': data['deadline_weekday'], 
                                'created_at_two_hour_chunk': int(data['created_at_two_hour_chunk']), 
                                'launched_at_two_hour_chunk': int(data['launched_at_two_hour_chunk']), 
                                'deadline_two_hour_chunk': int(data['deadline_two_hour_chunk']), 
                                'campaign_length': int(data['campaign_length']), 
                                'buildup_length': int(data['buildup_length']), 
                                'goal_usd': int(data['goal_usd']),                                         
                                'prelaunch_activated': int(data['prelaunch_activated'])}])
    prediction = pipeline.predict(new_instance)
    return render_template('index.html', prediction=int(prediction[0]))    

if __name__ == '__main__':
    app.run(debug=True)

