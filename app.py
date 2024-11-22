import streamlit as st
import pandas as pd
import joblib

# Load the trained model and the encoder
lr_model = joblib.load('model.pkl')
encoder = joblib.load('encoder.pkl')

# Custom CSS for styling
st.markdown("""
    <style>
    /* Result Box Styling */
    .result-box {
        margin-top: 20px;
        padding: 10px;
        background-color: #1E1E1E;
        border-left: 5px solid #FF4C29;
        border-radius: 5px;
        font-size: 1.3em;
        color: #FFFFFF;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app title and subtitle
st.title("Formula 1 Race Performance: Weather-Based Predictions 🌧️🏎️")
st.markdown('<div class="subtitle">Predict finish position based on weather and other factors</div>', unsafe_allow_html=True)

# Add space
st.markdown('<br>', unsafe_allow_html=True)  # Adjust the number of <br> tags for more or less space

# Arrange input fields in two columns
col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)", value=25)
    qualifying_score = st.number_input("Qualifying Score", value=90)
    driver_avg_finish = st.number_input("Driver Average Finish Position", value=3)
    track_condition = st.selectbox("Track Condition", ['dry', 'wet'])

with col2:
    temperature_category = st.selectbox("Temperature Category", ['medium', 'very_high', 'low', 'high'])
    constructor_avg_points = st.number_input("Constructor Average Points", value=15)
    weather_type = st.selectbox("Weather Type", ['overcast', 'cloudy', 'sunny', 'rain'])

# Button to make a prediction
if st.button("Predict Finish Position"):
    # Prepare input data
    input_data = pd.DataFrame({
        'temperature': [temperature],
        'temperature_category': [temperature_category],
        'qualifying_score': [qualifying_score],
        'driver_avg_finish': [driver_avg_finish],
        'constructor_avg_points': [constructor_avg_points],
        'weather_type': [weather_type],
        'track_condition': [track_condition]
    })

    # Select categorical features for encoding
    categorical_features = input_data[['temperature_category', 'weather_type', 'track_condition']]
    numeric_features = input_data[['temperature', 'qualifying_score', 'driver_avg_finish', 'constructor_avg_points']]

    # Transform categorical features with the loaded encoder
    encoded_features = encoder.transform(categorical_features)
    encoded_features_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

    # Combine numeric and encoded features
    input_data_final = pd.concat([numeric_features.reset_index(drop=True), encoded_features_df.reset_index(drop=True)], axis=1)

    # Make prediction using the loaded model
    prediction = lr_model.predict(input_data_final)
    predicted_finish_position = max(int(prediction[0]), 1)  # Ensure minimum finish position is 1

    # Adjust predicted_finish_position based on additional conditions
    # Temperature adjustments
    if temperature_category in ['very_high', 'high']:
        predicted_finish_position = max(predicted_finish_position - 1, 1)  # Favorable, reduce position
    elif temperature_category == 'low':
        predicted_finish_position += 1  # Less favorable, increase position

    # Track condition adjustments
    if track_condition == 'wet':
        predicted_finish_position += 2  # Wet track is less favorable
    elif track_condition == 'dry':
        predicted_finish_position = max(predicted_finish_position - 1, 1)  # Favorable, reduce position

    # Weather type adjustments
    if weather_type == 'sunny':
        predicted_finish_position = max(predicted_finish_position - 1, 1)  # Sunny is favorable
    elif weather_type == 'rain':
        predicted_finish_position += 5  # Rainy significantly impacts performance
    elif weather_type == 'cloudy':
        predicted_finish_position += 2  # Cloudy has moderate impact

    # Display the result in a styled box
    st.markdown(f'<div class="result-box">🏁 Predicted Finish Position: {predicted_finish_position}</div>', unsafe_allow_html=True)

# Add space
st.markdown('<br>', unsafe_allow_html=True)  # Adjust the number of <br> tags for more or less space

# Credits text at the bottom
st.markdown('<div style="color:grey; text-align:center">Yeshaswi Prakash © 2024</div>', unsafe_allow_html=True)
