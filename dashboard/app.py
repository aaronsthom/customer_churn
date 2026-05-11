import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the Model and Data
@st.cache_data # Cache the data loading function to improve performance
def load_raw_data():
    df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    return df

df = load_raw_data()


# Sidebar
st.sidebar.title("Filters")

# Page Navigation 
page = st.sidebar.radio("Navigate to:", ["Overview", "Customer Segments",
                                         "Model Performance", "Predict Churn"])

if page == "Overview":
    st.title("Customer Churn Overview")
    st.markdown("A high-level overview of customer churn across the Telco dataset.")
    st.divider()

    # KPI Metrics Row
    total_customers = len(df)
    churned = df['Churn'].value_counts()['Yes']
    churn_rate = (churned / total_customers) * 100
    avg_tenure = df['tenure'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", total_customers)
    col2.metric("Churned Customers", churned)
    col3.metric("Churn Rate (%)", format(churn_rate, '.1f'))
    col4.metric("Average Tenure (months)", format(avg_tenure, '.0f'))
    st.divider()

    # Charts Row
    ## Pie Chart for Churn Distribution
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn Distribution")
        churn_counts = df['Churn'].value_counts().reset_index()
        churn_counts.columns = ['Churn', 'Count']
        fig_pie = px.pie(
            churn_counts,
            names='Churn',
            values='Count',
            color='Churn',
            hole=0.4, # For Pie chart effect
            color_discrete_map={'Yes': 'red', 'No': 'blue'}
        )
        fig_pie.update_traces(textposition='outside', textinfo='percent+label')
        fig_pie.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)
    
    ## Bar Chart for Churn by Contract Type
    with col2:
        st.subheader("Churn by Contract Type")
        contract_churn = df.groupby('Contract')['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100).reset_index()
        contract_churn.columns = ['Contract', 'Churn Rate (%)']
        contract_churn = contract_churn.sort_values(by='Churn Rate (%)', ascending=True)
        fig_bar = px.bar(
            contract_churn,
            x='Churn Rate (%)',
            y='Contract',
            orientation='h',
            color='Churn Rate (%)',
            color_continuous_scale='Reds',
            text='Churn Rate (%)'
        )
        fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_bar.update_layout(
            coloraxis_showscale=False,
            xaxis_range=[0,50],
            margin=dict(t=20, b=20, r=40),
            xaxis_title="Churn Rate (%)",
            yaxis_title=""
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    st.divider()

    # Box Plot Row
    st.subheader("Monthly Charges Distribution by Churn")
    fig_box = px.box(
        df,
        x='Churn',
        y='MonthlyCharges',
        color='Churn',
        color_discrete_map={'Yes': '#EF553B', 'No': '#636EFA'},
        labels={'Churn': 'Churn Status', 'MonthlyCharges': 'Monthly Charges ($)'}
    )
    fig_box.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_box, use_container_width=True)


elif page == "Customer Segments":
    st.title("Customer Segments")
    st.markdown("Explore how churn varies across different customer segments.")
    st.divider()

    # Sidebar filters for Customer Segments
    st.sidebar.subheader("Segment Filters")
    internet_filter = st.sidebar.multiselect(
        "Internet Service Type",
        options=df['InternetService'].unique().tolist(),
        default=df['InternetService'].unique().tolist()
    )
    payment_filter = st.sidebar.multiselect(
        "Payment Method",
        options=df['PaymentMethod'].unique().tolist(),
        default=df['PaymentMethod'].unique().tolist()
    )

    # Apply the filters to the dataframe
    filtered_df = df[
        (df['InternetService'].isin(internet_filter)) &
        (df['PaymentMethod'].isin(payment_filter))
    ]

    # Filter summary
    st.caption(f"Showing {len(filtered_df):,} of {len(df):,} customers after applying filters.")
    st.divider()

    ## Internet Service and Payment Method Rows
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn Rate by Internet Service")
        internet_churn = filtered_df.groupby('InternetService')['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100).reset_index()
        internet_churn.columns = ['InternetService', 'Churn Rate (%)']
        internet_churn = internet_churn.sort_values('Churn Rate (%)', ascending=True)
        fig_internet = px.bar(
            internet_churn,
            x='Churn Rate (%)',
            y='InternetService',
            orientation='h',
            color='Churn Rate (%)',
            color_continuous_scale='Reds',
            text='Churn Rate (%)'
        )
        fig_internet.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_internet.update_layout(
            coloraxis_showscale=False,
            xaxis_range=[0,80],
            margin=dict(t=20, b=20, r=40),
            xaxis_title="Churn Rate (%)",
            yaxis_title=""
        )
        st.plotly_chart(fig_internet, use_container_width=True)

    with col2:
        st.subheader("Churn Rate by Payment Method")
        payment_churn = filtered_df.groupby('PaymentMethod')['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100).reset_index()
        payment_churn.columns = ['PaymentMethod', 'Churn Rate (%)']
        payment_churn = payment_churn.sort_values('Churn Rate (%)', ascending=True)
        fig_payment = px.bar(
            payment_churn,
            x='Churn Rate (%)',
            y='PaymentMethod',
            orientation='h',
            color='Churn Rate (%)',
            color_continuous_scale='Reds',
            text='Churn Rate (%)'
        )
        fig_payment.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_payment.update_layout(
            coloraxis_showscale=False,
            xaxis_range=[0,60],
            margin=dict(t=20, b=20, r=40),
            xaxis_title="Churn Rate (%)",
            yaxis_title=""
        )
        st.plotly_chart(fig_payment, use_container_width=True)

    # Tenure Group Row
    st.subheader("Churn Rate by Tenure Group")
    filtered_df = filtered_df.copy() # Avoid SettingWithCopyWarning
    filtered_df['TenureGroup'] = pd.cut(
        filtered_df['tenure'],
        bins=[0, 12, 24, 36, 48, 60, 72],
        labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72']
    )
    tenure_churn = filtered_df.groupby('TenureGroup', observed=True)['Churn'].apply(
        lambda x: (x == 'Yes').sum() / len(x) * 100).reset_index()
    tenure_churn.columns = ['TenureGroup', 'Churn Rate (%)']
    fig_tenure = px.bar(
        tenure_churn,
        x='TenureGroup', 
        y='Churn Rate (%)',
        color='Churn Rate (%)',
        color_continuous_scale='Reds',
        text='Churn Rate (%)'
    )
    fig_tenure.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_tenure.update_layout(
        coloraxis_showscale=False,
        yaxis_range=[0,70],
        margin=dict(t=20, b=20),
        xaxis_title="Tenure Group (months)",
        yaxis_title="Churn Rate (%)"
    )
    st.plotly_chart(fig_tenure, use_container_width=True)
    st.divider()

    # Senior Citizens and Dependents Row
    col1_2, spacer, col2_2 = st.columns([1, 0.2, 1])

    with col1_2:
        st.subheader("Churn Rate by Senior Citizen Status")
        filtered_df['SeniorCitizen'] = filtered_df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        senior_churn = filtered_df.groupby('SeniorCitizen')['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100).reset_index()
        senior_churn.columns = ['SeniorCitizen', 'Churn Rate (%)']
        fig_senior = px.bar(
            senior_churn,
            x='SeniorCitizen',
            y='Churn Rate (%)',
            color='Churn Rate (%)',
            color_continuous_scale='Reds',
            text='Churn Rate (%)'
        )
        fig_senior.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_senior.update_layout(
            coloraxis_showscale=False,
            xaxis_title="Senior Citizen Status",
            margin=dict(t=20, b=20),
            yaxis_range=[0,50]
        )
        st.plotly_chart(fig_senior, use_container_width=True)

    with col2_2:
        st.subheader("Churn Rate by Dependents")
        dependents_churn = filtered_df.groupby('Dependents')['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100).reset_index()
        dependents_churn.columns = ['Dependents', 'Churn Rate (%)']
        fig_dependents = px.bar(
            dependents_churn,
            x='Dependents',
            y='Churn Rate (%)',
            color='Churn Rate (%)',
            color_continuous_scale='Reds',
            text='Churn Rate (%)'
        )
        fig_dependents.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_dependents.update_layout(
            coloraxis_showscale=False,
            xaxis_title="Has Dependents",
            margin=dict(t=20, b=20),
            yaxis_range=[0,40]
        )
        st.plotly_chart(fig_dependents, use_container_width=True)

elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown("Evaluation of all trained models with a focus on the final XGBoost model.")
    st.divider()

    # Load the processed data
    @st.cache_data
    def load_processed_data():
        X_test = pd.read_csv('../data/processed/X_test.csv')
        y_test = pd.read_csv('../data/processed/y_test.csv').iloc[:, 0] 
        X_train = pd.read_csv('../data/processed/X_train.csv')
        return X_test, y_test, X_train
    
    @st.cache_resource
    def load_models():
        return joblib.load('../models/xgb_model.pkl')
    
    X_test, y_test, X_train = load_processed_data()
    models = load_models()

    y_pred = models.predict(X_test)
    y_prob = models.predict_proba(X_test)[:, 1]

    # Model Comparison Table Row
    st.subheader("Model Comparison")
    results = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'SVM'],
        'Accuracy': [0.76, 0.76, 0.75, 0.76],
        'Precision': [0.54, 0.54, 0.52, 0.53],
        'Recall': [0.69, 0.63, 0.72, 0.63],
        'F1-Score': [0.60, 0.58, 0.61, 0.60],
        'ROC AUC': [0.83, 0.82, 0.83, 0.82]
    }
    results_df = pd.DataFrame(results).set_index('Model')

    ## Highlight the best performing model
    st.dataframe(
        results_df.style.highlight_max(
            axis=0, color='#1f4e2e').
            format("{:.2f}"),
        use_container_width=True)
    st.caption("XGBoost selected as final model - highest Recall (.72) and F1-Score (.61)")
    st.markdown("<br></br>", unsafe_allow_html=True)

    # ROC Curve Comparison Row
    st.subheader("ROC Curve Comparison")
    st.caption("ROC curves for all four models on the test set.")

    ## ROC AUC values from the notebook
    roc_data = {
        'Logistic Regression': 0.83,
        'Random Forest': 0.82,
        'XGBoost': 0.83,
        'SVM': 0.82
    }

    ## Recreate the ROC curve for XGBoost
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        name='XGBoost (AUC = {:.2f})'.format(roc_data['XGBoost']),
        line=dict(color='#EF553B', width=2)
    ))
    ## Add diagonal line for reference
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random Baseline',
        line=dict(color='gray', width=1, dash='dash')
    ))
    fig_roc.update_layout(
        height=400,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=0.6, y=0.1),
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    st.caption("Note: ROC curves for Logistic Regression, Random Forest, and SVM require their saved models. Only the final XGBoost is displayed here")

    st.markdown("<br></br>", unsafe_allow_html=True)

    # Confusion Matrix and Feature Importance Row
    col1_3, spacer, col2_3 = st.columns([1, 0.2, 1])

    with col1_3:
        st.subheader("Confusion Matrix - XGBoost")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale='Reds',
            x=['Predicted: No Churn', 'Predicted: Yes Churn'],
            y=['Actual: No Churn', 'Actual: Yes Churn'],
            aspect='auto'
        )
        fig_cm.update_layout(
            height=380,
            coloraxis_showscale=False,
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("False Negatives (bottom left) = missed churners - most costly error for business.")

    with col2_3:
        st.subheader("Top 10 Feature Importances - XGBoost")
        importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': models.feature_importances_
        }).sort_values(by='Importance', ascending=True).tail(10)

        fig_importance = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Reds',
            text='Importance'
        )
        fig_importance.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_importance.update_layout(
            height=380,
            coloraxis_showscale=False,
            margin=dict(t=20, b=20),
            yaxis_title=""
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # Final Model Performance Metrics Row
    st.subheader("Final Model Metrics - XGBoost")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", format(results_df.loc['XGBoost', 'Accuracy'], '.2f'))
    m2.metric("Precision", format(results_df.loc['XGBoost', 'Precision'], '.2f'))
    m3.metric("Recall", format(results_df.loc['XGBoost', 'Recall'], '.2f'))
    m4.metric("F1-Score", format(results_df.loc['XGBoost', 'F1-Score'], '.2f'))
    m5.metric("ROC AUC", format(results_df.loc['XGBoost', 'ROC AUC'], '.2f'))

elif page == "Predict Churn":
    st.title("Predict Customer Churn")
    st.markdown("Enter a customer's details to predict their likelihood of churning.")
    st.divider()

    # Load XGBoost and scaler models
    @st.cache_resource
    def load_xgb_scaler_models():
        model = joblib.load('../models/xgb_model.pkl')
        scaler = joblib.load('../models/standard_scaler.joblib')
        return model, scaler
    
    model, scaler = load_xgb_scaler_models()

    # Input form for customer details
    st.subheader("Customer Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Account Information**")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
        payment_method = st.selectbox("Payment Method", [
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])

    with col2:
        st.markdown("**Services**")
        internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
        phone_service = st.radio("Phone Service", ['Yes', 'No'])
        multiple_lines = st.radio("Multiple Lines", ['No phone service', 'No', 'Yes'])
        streaming_tv = st.radio("Streaming TV", ['No internet service', 'No', 'Yes'])
        streaming_movies = st.radio("Streaming Movies", ['No internet service', 'No', 'Yes'])

    with col3:
        st.markdown("**Add-on Services**")
        online_security = st.radio("Online Security", ['No internet service', 'No', 'Yes'])
        online_backup = st.radio("Online Backup", ['No internet service', 'No', 'Yes'])
        device_protection = st.radio("Device Protection", ['No internet service', 'No', 'Yes'])
        tech_support = st.radio("Tech Support", ['No internet service', 'No', 'Yes'])

    st.markdown("<br>", unsafe_allow_html=True)
    col_charge1, col_charge2, _ = st.columns([1, 1, 1])
    with col_charge1:
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0, step=0.01)
    with col_charge2:
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly_charges * tenure), step=0.01)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # Predict Button
    if st.button("Predict Churn", type='primary', use_container_width=True):
        # Feature Engineering
        ## Engineered features
        avg_monthly_charges = monthly_charges if tenure == 0 else total_charges / tenure
        has_streaming = 1 if streaming_tv == 'Yes' or streaming_movies == 'Yes' else 0
        num_add_ons = sum([
            online_security == "Yes",
            online_backup == "Yes",
            device_protection == "Yes",
            tech_support == "Yes",
            streaming_tv == "Yes",
            streaming_movies == "Yes"
        ])
        is_new_customer = 1 if tenure <= 12 else 0

        # Create raw input dictionary matching training columns
        input_dict = {
            'gender': 0,
            'SeniorCitizen': 0,
            'Partner': 0,
            'Dependents': 0,
            'tenure': tenure,
            'PhoneService': 1 if phone_service == 'Yes' else 0,
            'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'AvgMonthlyCharges': avg_monthly_charges,
            'HasStreamingServices': has_streaming,
            'NumAddOnServices': num_add_ons,
            'IsNewCustomer': is_new_customer,
            'MultipleLines_No phone service': 1 if multiple_lines == 'No phone service' else 0,
            'MultipleLines_Yes': 1 if multiple_lines == 'Yes' else 0,
            'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
            'InternetService_No': 1 if internet_service == 'No' else 0,
            'OnlineSecurity_No internet service': 1 if online_security == 'No internet service' else 0,
            'OnlineSecurity_Yes': 1 if online_security == 'Yes' else 0,
            'OnlineBackup_No internet service': 1 if online_backup == 'No internet service' else 0,
            'OnlineBackup_Yes': 1 if online_backup == 'Yes' else 0,
            'DeviceProtection_No internet service': 1 if device_protection == 'No internet service' else 0,
            'DeviceProtection_Yes': 1 if device_protection == 'Yes' else 0,
            'TechSupport_No internet service': 1 if tech_support == 'No internet service' else 0,
            'TechSupport_Yes': 1 if tech_support == 'Yes' else 0,
            'StreamingTV_No internet service': 1 if streaming_tv == 'No internet service' else 0,
            'StreamingTV_Yes': 1 if streaming_tv == 'Yes' else 0,
            'StreamingMovies_No internet service': 1 if streaming_movies == 'No internet service' else 0,
            'StreamingMovies_Yes': 1 if streaming_movies == 'Yes' else 0,
            'Contract_One year': 1 if contract == 'One year' else 0,
            'Contract_Two year': 1 if contract == 'Two year' else 0,
            'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
            'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic check' else 0,
            'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed check' else 0,
        }

        # Scale numerical features
        input_df = pd.DataFrame([input_dict])

        expected_cols = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
            'AvgMonthlyCharges', 'HasStreamingServices', 'NumAddOnServices', 'IsNewCustomer',
            'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No',
            'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
            'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes',
            'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
        ]
        input_df = input_df[expected_cols]

        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharges']
        input_df[num_cols] = scaler.transform(input_df[num_cols])


        # Prediction
        prob = model.predict_proba(input_df)[0][1]
        risk = "High Risk" if prob >= 0.7 else "Medium Risk" if prob >= 0.4 else "Low Risk"
        risk_color = "#EF553B" if risk == "High Risk" else "#FFA500" if risk == "Medium Risk" else "#00CC96"

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", # Show both the gauge and the number
            value=round(prob * 100, 1), # Convert to percentage
            number={'suffix': "%", 'font': {'size': 48}}, # Show probability as percentage with larger font
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 40], 'color': '#1a472a'}, # Low Risk
                    {'range': [40, 70], 'color': '#7d4e00'}, # Medium Risk
                    {'range': [70, 100], 'color': '#7d0000'} # High Risk
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': prob * 100
                }
            },
            title={'text': "Churn Risk Probability", 'font': {'size': 24}}
        ))
        fig_gauge.update_layout(height=400, margin=dict(t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Risk Level and Key Factors
        col_risk, spacer, col_factors = st.columns([1, 0.1, 1])
        with col_risk:
            st.markdown(f"""
            <div style='text-align: center; padding: 30px;
                         border-radius: 12px; 
                        background-color: {risk_color}22;
                        border: 2px solid {risk_color};'>
                <h2 style='color: {risk_color}; margin: 0;'>{risk}</h2>
                <p style='color:white; margin-top: 10px;'>
                    Churn Probability: <strong>{prob:.1%}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_factors:
            st.markdown("**Key Risk Factors Detected:**")
            factors = []
            if contract == 'Month-to-month':
                factors.append("⚠️ Month-to-month contract - highest churn risk")
            if payment_method == 'Electronic check':
                factors.append("⚠️ Electronic check payment - associated with higher churn")
            if internet_service == 'Fiber optic':
                factors.append("⚠️ Fiber optic internet - higher churn rates")
            if is_new_customer:
                factors.append("⚠️ New customer - higher risk in first year")
            if monthly_charges > 65:
                factors.append("⚠️ High monthly charges - above average spend associated with churn")
            if num_add_ons == 0:
                factors.append("⚠️ No add-on services - customers with fewer services more likely to churn")
            if not factors:
                factors.append("✅ No major risk factors detected based on input features.")
            for f in factors:
                st.markdown(f)

