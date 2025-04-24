import shap
import plotly.express as px

def generate_shap(model, data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    
    fig = px.bar(
        pd.DataFrame({
            'features': data.columns,
            'importance': abs(shap_values).mean(0)
        }).sort_values('importance', ascending=False).head(10),
        x='importance',
        y='features',
        orientation='h'
    )
    
    fig.update_layout(
        title="Top 10 Risk Factors",
        height=500
    )
    return fig
