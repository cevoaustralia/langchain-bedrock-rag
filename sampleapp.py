# Importing necessary libraries
import streamlit as st

# Title of the application
st.title('Durga Sample Streamlit App')

# Header
st.header('Welcome to my Streamlit App')

# Subheader
st.subheader('Enter your details')

# Text input field for user's name
name = st.text_input('Enter your name', 'John Doe')

# Slider for user's age
age = st.slider('Select your age', 0, 100, 25)

# Checkbox for user's gender
gender = st.radio('Select your gender', ['Male', 'Female'])

# Button to submit the form
submit_button = st.button('Submit')

# Display the user's input upon clicking the submit button
if submit_button:
    st.write(f'Name: {name}')
    st.write(f'Age: {age}')
    st.write(f'Gender: {gender}')
