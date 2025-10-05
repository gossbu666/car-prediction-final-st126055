import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import os
import mlflow

# --- Step 1: Configure MLflow and Load the UNIFIED PIPELINE ---
os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password'
mlflow.set_tracking_uri("http://mlflow.ml.brain.cs.ait.ac.th/")

model = None
try:
    # --- Make sure to use the latest correct version number ---
    model_version = 7 # <-- Or your latest working pipeline version
    model_name = "st126055-a3-model" # <-- Use the final model name
    model_uri = f"models:/{model_name}/{model_version}"

    print(f"Attempting to load Unified Pipeline '{model_name}' (Version {model_version})...")
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"✅ Unified Pipeline loaded successfully!")

except Exception as e:
    print(f"❌ Error loading model from MLflow: {e}")

# --- Step 2: Create the Dash App Layout ---
price_classes = {0: 'Category 0 (Lowest Price)', 1: 'Category 1', 2: 'Category 2', 3: 'Category 3 (Highest Price)'}
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '800px', 'margin': 'auto', 'padding': '20px'}, children=[
    html.H1("🚗 Car Price Category Prediction", style={'textAlign': 'center', 'color': '#333'}),
    html.P("Enter the car's details below to predict its price category.", style={'textAlign': 'center', 'color': '#666'}),
    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px', 'marginTop': '30px'}, children=[
        html.Div([
            html.Label("Brand:", style={'fontWeight': 'bold'}), dcc.Input(id='brand', type='text', value='Maruti', style={'width': '100%', 'padding': '8px', 'marginTop': '5px'}),
            html.Label("Year:", style={'fontWeight': 'bold', 'marginTop': '15px'}), dcc.Input(id='year', type='number', value=2015, style={'width': '100%', 'padding': '8px', 'marginTop': '5px'}),
            html.Label("Kilometers Driven:", style={'fontWeight': 'bold', 'marginTop': '15px'}), dcc.Input(id='km_driven', type='number', value=50000, style={'width': '100%', 'padding': '8px', 'marginTop': '5px'}),
            html.Label("Fuel Type:", style={'fontWeight': 'bold', 'marginTop': '15px'}), dcc.Dropdown(id='fuel', options=['Petrol', 'Diesel'], value='Petrol', style={'width': '100%', 'marginTop': '5px'}),
            html.Label("Transmission:", style={'fontWeight': 'bold', 'marginTop': '15px'}), dcc.Dropdown(id='transmission', options=['Manual', 'Automatic'], value='Manual', style={'width': '100%', 'marginTop': '5px'}),
        ]),
        html.Div([
            html.Label("Max Power (bhp):", style={'fontWeight': 'bold'}), dcc.Input(id='max_power', type='number', value=80, style={'width': '100%', 'padding': '8px', 'marginTop': '5px'}),
            html.Label("Engine (CC):", style={'fontWeight': 'bold', 'marginTop': '15px'}), dcc.Input(id='engine', type='number', value=1200, style={'width': '100%', 'padding': '8px', 'marginTop': '5px'}),
            html.Label("Mileage (kmpl):", style={'fontWeight': 'bold', 'marginTop': '15px'}), dcc.Input(id='mileage', type='number', value=20.0, style={'width': '100%', 'padding': '8px', 'marginTop': '5px'}),
            html.Label("Seats:", style={'fontWeight': 'bold', 'marginTop': '15px'}), dcc.Input(id='seats', type='number', value=5, style={'width': '100%', 'padding': '8px', 'marginTop': '5px'}),
            html.Label("Owner:", style={'fontWeight': 'bold', 'marginTop': '15px'}), dcc.Dropdown(id='owner', options=[{'label': 'First Owner', 'value': 1}, {'label': 'Second Owner', 'value': 2}, {'label': 'Third Owner', 'value': 3}, {'label': 'Fourth & Above Owner', 'value': 4}], value=1, style={'width': '100%', 'marginTop': '5px'}),
        ]),
    ]),
    dcc.Store(id='seller_type', data='Individual'),
    html.Button('Predict Price Category', id='predict-button', n_clicks=0, style={'width': '100%', 'padding': '15px', 'fontSize': '18px', 'marginTop': '30px', 'backgroundColor': '#007BFF', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
    html.Div(id='prediction-output', style={'textAlign': 'center', 'marginTop': '30px', 'fontSize': '24px', 'fontWeight': 'bold', 'color': '#28a745'})
])

# --- Step 3: Callback for Prediction (With the final fix) ---
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [State('brand', 'value'), State('year', 'value'), State('km_driven', 'value'), State('fuel', 'value'), State('transmission', 'value'), State('max_power', 'value'), State('engine', 'value'), State('mileage', 'value'), State('seats', 'value'), State('owner', 'value'), State('seller_type', 'data')]
)
def update_output(n_clicks, brand, year, km_driven, fuel, transmission, max_power, engine, mileage, seats, owner, seller_type):
    if n_clicks == 0 or model is None:
        return ""

    input_df = pd.DataFrame([{
        'fuel': fuel, 'seller_type': seller_type, 'transmission': transmission, 'brand': brand,
        'engine': engine, 'max_power': max_power, 'mileage': mileage, 'seats': seats,
        'year': year, 'km_driven': km_driven, 'owner': owner
    }])

    # --- ✅ THE FINAL FIX IS HERE: Enforce the EXACT data types from the model's schema ---
    try:
        # Define the exact schema the model expects
        schema = {
            'engine': float,
            'max_power': float,
            'mileage': float,
            'seats': float,
            'year': int,
            'km_driven': int,
            'owner': int
        }
        
        # Apply the schema to the input DataFrame
        input_df = input_df.astype(schema)

        # The .predict() method now receives data in the perfect format.
        prediction_result = model.predict(input_df)
        predicted_class = prediction_result[0]
        return f"Predicted Category: {price_classes[predicted_class]}"
    except Exception as e:
        return f"Prediction Error: {e}"

# --- Step 4: Run the App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)

