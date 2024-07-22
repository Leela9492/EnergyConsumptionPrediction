from flask import Flask, render_template,request
import requests
app = Flask(__name__)
import pickle
with open('models/scaler.pkl', 'rb') as file:
    scaler_loaded = pickle.load(file)

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/', methods=['POST'])
def handle_form():
    ow_consum = float(request.form['low_consum'])
    high_consum = request.form['high_consum']
    hours = request.form['hours']
    t6 = request.form['t6']
    rh_6 = request.form['rh_6']
    lights = request.form['lights']
    hour_lights = request.form['hour_lights']
    tdewpoint = request.form['tdewpoint']
    visibility = request.form['visibility']
    press_mm_hg = request.form['press_mm_hg']
    windspeed = request.form['windspeed']
    keys = [ow_consum, high_consum, hours, t6, rh_6, lights, hour_lights, tdewpoint, visibility, press_mm_hg, windspeed]
    # Now you can do whatever you want with these values, like passing them to a model for prediction, etc.
    # For demonstration purpose, I'll just print them here
    print(f'ow_consum: {ow_consum}, high_consum: {high_consum}, hours: {hours}, t6: {t6}, rh_6: {rh_6}, lights: {lights}, hour_lights: {hour_lights}, tdewpoint: {tdewpoint}, visibility: {visibility}, press_mm_hg: {press_mm_hg}, windspeed: {windspeed}')
    data_array = [float(i) for i in keys]
    data_scaled = scaler_loaded.transform([data_array])
    import joblib

    # Load the model
    model = joblib.load('models/model_filename.pkl')

    # Assuming X_train[0] is the data point you want to make a prediction for
    data_point = data_scaled

    # Reshape the data point if necessary (depends on the shape expected by the model)
    data_point = data_point.reshape(1, -1)

    # Make a prediction
    prediction = model.predict(data_point)
    column_values = {
        'Low_consum': ow_consum,
        'high_consum': high_consum,
        'hours': hours,
        't6': t6,
        'rh_6': rh_6,
        'lights': lights,
        'hour_lights': hour_lights,
        'tdewpoint': tdewpoint,
        'visibility': visibility,
        'press_mm_hg': press_mm_hg,
        'windspeed': windspeed
    }


    return render_template('base.html',prediction=abs(prediction[0]),output=1,col=column_values)



if __name__ == '__main__':
    app.run(debug=True)
