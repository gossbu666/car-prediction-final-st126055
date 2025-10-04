import dash
from dash import html

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("🚗 A3 Car Price Prediction App"),
    html.P("This is a placeholder Dash app container, deployed via CI/CD.")
])

if __name__ == "__main__":
    # Dash >= 3 uses app.run instead of app.run_server
    app.run(host="0.0.0.0", port=8050, debug=False)