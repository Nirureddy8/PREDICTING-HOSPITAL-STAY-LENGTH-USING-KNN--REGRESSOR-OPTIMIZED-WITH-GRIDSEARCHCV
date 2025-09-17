import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request, redirect, url_for, flash, session

# Initialize Flask App
app = Flask(__name__)
app.secret_key = '12345y78'

# Initialize users dictionary for demonstration (use database for production)
users = {}

# Load the trained model, scaler, feature order, and location encoding map
with open("Grid_Model1.pkl", "rb") as model_file:
    knn_model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("feature_order.pkl", "rb") as feature_file:
    feature_order = pickle.load(feature_file)

with open("location_map.pkl", "rb") as location_map_file:
    value_map = pickle.load(location_map_file)

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Home Page After Login
@app.route('/home')
def home():
    if 'user' in session:
        return render_template('home.html', user=session['user'])
    return redirect(url_for('login'))

# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        if email in users and users[email] == password:
            session['user'] = email
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password!', 'danger')
    
    return render_template('login.html')

# Register Page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
        elif email in users:
            flash('Email is already registered!', 'warning')
        else:
            users[email] = password
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    
    return render_template('register.html')

# Prediction route
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'user' in session:  # Ensure user is logged in
        if request.method == 'POST':
            try:
                # Get form data
                location_str = request.form['location']  # Dropdown or text input for location
                location = value_map.get(location_str.strip().upper(), None)  # Use value_map for encoding
                if location is None:
                    raise ValueError("Invalid location code")

                year = float(request.form['time'])  # Year input
                mri_units = float(request.form['mri_units'])  # MRI Units input
                ct_scanners = float(request.form['ct_scanners'])  # CT Scanners input
                hospital_beds = float(request.form['hospital_beds'])  # Hospital Beds input
                
                # Prepare the input as a DataFrame
                input_data = pd.DataFrame([[location, year, mri_units, ct_scanners, hospital_beds]], columns=feature_order)

                # Apply scaling (use the scaler loaded earlier)
                scaled_input = scaler.transform(input_data)
                print(f"Scaled input: {scaled_input}")

                # Make the prediction
                prediction_result = knn_model.predict(scaled_input)[0]
                print("Prediction:", prediction_result)

                # Return result to the template
                return render_template('result.html', 
                                       user=session['user'], 
                                       prediction=round(prediction_result, 2))
            
            except ValueError as e:
                # If there's an error (e.g., invalid input), show an error message
                flash(f"Error: {e}. Please enter valid values for all features.", 'danger')
            except Exception as e:
                # Handle any other errors
                flash(f"An error occurred: {e}", 'danger')

        # Render the prediction form if it's a GET request
        return render_template('prediction.html', user=session['user'])

    # Redirect to login page if user is not logged in
    return redirect(url_for('login'))

# Performance Page After Login
@app.route('/performance')
def performance():
    if 'user' in session:
        return render_template('performance.html', user=session['user'])
    return redirect(url_for('login'))

# Charts Page After Login
@app.route('/charts')
def charts():
    if 'user' in session:
        return render_template('charts.html', user=session['user'])
    return redirect(url_for('login'))

# Logout Route
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out!', 'info')
    return redirect(url_for('login'))

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
