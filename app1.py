import streamlit as st
import pandas as pd
import joblib

# Load the preprocessed data
df = pd.read_csv('C:/Users/demonking/Downloads/demand forecasting dataset/SalesUpdated.csv')

# Load the pre-trained model
model = joblib.load('trained_model.joblib')

# Function to predict monthly quantity
def predict_monthly_quantity(store, brand, month, year):
    input_data = pd.DataFrame({
        'Store': [store],
        'Brand': [brand],
        'Month': [month],
        'Year': [year]
    })

    prediction = model.predict(input_data)
    return prediction[0]

# Static Dashboard
st.sidebar.title("Static Dashboard")

# Page navigation buttons
button_selection = st.sidebar.radio("Go to:", ("Home", "Inventory", "About"))

if button_selection == "Home":
    st.title('Monthly Quantity Prediction App')

    # User input
    store = st.selectbox('Select Store', df['Store'].unique())
    brand = st.selectbox('Select Brand', df['Brand'].unique())
    month = st.slider('Select Month', 1, 12, 1)
    selected_year = st.text_input('Enter Year', df['Year'].min())

    # Convert the entered year to an integer
    selected_year = int(selected_year)

    # Predict button
    if st.button('Predict Monthly Quantity'):
        # Make prediction
        prediction = predict_monthly_quantity(store, brand, month, selected_year)

        # Display the result
        st.success(f'Predicted Monthly Quantity: {prediction:.2f}')

elif button_selection == "Inventory":
    st.title('Inventory')
    # st.title('Analytics Page')

    # Perform calculations for SafetyStock, ReorderPoint, MaxStock, and ActualInventory
    df_beg_inventory = pd.read_csv('C:/Users/demonking/Downloads/demand forecasting dataset/BegInvFINAL12312016.csv')
    df_purchases = pd.read_csv('C:/Users/demonking/Downloads/demand forecasting dataset/PurchasesFINAL12312016.csv')

    # Calculate Safety Stock, Reorder Point, and Maximum Stock for df_beg_inventory
    df_beg_inventory['SafetyStock'] = df_beg_inventory['onHand'] * 0.2  # 20% of onHand as Safety Stock

    df_beg_inventory['ReorderPoint'] = df_beg_inventory['SafetyStock'] + df_purchases['Quantity'].mean()
    df_beg_inventory['MaxStock'] = df_beg_inventory['ReorderPoint'] * 2  # Setting Max Stock at twice the Reorder Point

    # Calculate the actual inventory at the begin of the period
    df_beg_inventory['ActualInventory'] = df_beg_inventory['onHand'] + df_purchases['Quantity'].sum()

    predictions_per_brand_store = df_beg_inventory.groupby(['Brand', 'Store']).agg({
            'SafetyStock': 'first',
            'ReorderPoint': 'first',
            'MaxStock': 'first',
            'ActualInventory': 'first'
        }).reset_index()

        # Display the results
    st.write("Predicted Safety Stock, Reorder Point, Max Stock, and Actual Inventory per Brand and Store:")
    st.write(predictions_per_brand_store)
    
    st.write("### Calendar Component")

    # Create a calendar component
    selected_date = stqdm.date_input("Select a Date", pd.to_datetime('today'))

    # Use the selected_date as needed in your app
    st.write("You selected:", selected_date)


elif button_selection == "About":
    st.title('About Page')
    # Add about content here