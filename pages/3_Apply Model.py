import streamlit as st
import pandas as pd
import altair as alt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import warnings
import os
from lime import lime_tabular
import shap
import numpy as np
warnings.simplefilter(action='ignore', category=UserWarning)
# st.set_page_config(layout="wide")
# step 0 : read in supportive data and cleaning
GDP_df = pd.read_csv('./supportive data/GDP.csv')
GDP_df = GDP_df[['Country or Area', 'Year', "GDP per Capita"]]

# select average working hours for full-time employment across all ages. 
working_hr_df = pd.read_csv('./supportive data/average working hours.csv')
working_hr_df = working_hr_df[(working_hr_df['SEX']=='MEN') | (working_hr_df['SEX']=='WOMEN')]
working_hr_df = working_hr_df[(working_hr_df['JOBTYPE']=='FT') & (working_hr_df['EMPSTAT']=='TE')]
working_hr_df = working_hr_df[(working_hr_df['Age']=='Total')]
working_hr_df = working_hr_df[['COUNTRY', 'Sex', 'Time', 'Value']]
working_hr_df.columns = ['country', 'sex', 'year', 'working_hr']
working_hr_df['sex'] = working_hr_df['sex'].replace({'Men': 'Male', 'Women': 'Female'})


disease_name_list = [each.split('.csv')[0] for each in os.listdir('./data') if each[-4:]=='.csv']
disease = st.sidebar.selectbox("Disease you want to explore: ", disease_name_list)

disease_df = pd.read_csv(f'./data/{disease}.csv', index_col=False, skiprows=6)
disease_df = disease_df[(disease_df['Age group code']!='Age_unknown') & (disease_df['Sex']!='Unknown')]
disease_df = disease_df[(disease_df['Age group code']!='Age_all') & (disease_df['Sex']!='All')]
disease_df = disease_df[['Country Code', 'Country Name','Year','Sex', 'Age Group', "Death rate per 100 000 population"]]


df_whole = disease_df.dropna()
df_whole = df_whole.merge(GDP_df, how='left', left_on=['Country Name', 'Year'], right_on=['Country or Area', 'Year'])
df_whole = df_whole.drop(columns=['Country or Area'])

# df_whole = df_whole.fillna(df_whole['GDP per Capita'].mean())
df_whole = df_whole.merge(working_hr_df, how='left', left_on=['Country Code', 'Sex', 'Year'], right_on=['country', 'sex', 'year'])
df_whole = df_whole.drop(columns=['country', 'sex', 'year'])
# df_whole = df_whole.fillna(df_whole['working_hr'].mean())
df_whole.dropna(inplace=True)

st.subheader('Train the model!')
model_name = st.sidebar.selectbox('Choose the model you want to use', ['Linear Regression', 'Decision Tree'])



# select features
whole_for_train = df_whole.drop(columns=['Country Code', 'Country Name', 'Year'])
# train_test_split
train_set, test_set = train_test_split(whole_for_train, test_size=0.2, random_state=42)
health_train = train_set.drop('Death rate per 100 000 population', axis=1)
health_train_labels = train_set['Death rate per 100 000 population'].copy()

health_test = test_set.drop('Death rate per 100 000 population', axis=1)
health_test_labels = test_set['Death rate per 100 000 population'].copy()

# one hot encoding categorical features
health_num = health_train.drop(['Sex'], axis=1)
health_num = health_num.drop(['Age Group'], axis=1)


num_features = list(health_num.keys())
cat_features = ['Sex', 'Age Group']

full_pipeline = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(), cat_features),
])

health_train_prepared = full_pipeline.fit_transform(health_train)
health_test_prepared = full_pipeline.transform(health_test)
column_names = list(health_num.columns) + list(full_pipeline.transformers_[1][1].get_feature_names_out())

if model_name == 'Linear Regression':
    # fit linear regression model
    model = LinearRegression()
    model.fit(health_train_prepared, health_train_labels)

    health_predictions = model.predict(health_test_prepared)

    # post-process: negative value->0
    health_predictions[health_predictions < 0] = 0
    mse = mean_squared_error(health_test_labels, health_predictions)
    pred_df = pd.DataFrame({'prediction': health_predictions, 'ground truth': health_test_labels})
    pred_fig = alt.Chart(pred_df).mark_point().encode(
        alt.X('ground truth'),
        alt.Y('prediction'),
    ).properties(
        height=600
    )
    weight_df = pd.DataFrame({'features': column_names, 'weight': model.coef_})
    weight_fig = alt.Chart(weight_df).mark_bar().encode(
        alt.X('weight'),
        alt.Y('features', sort='-x'),
    ).properties(
        height=600
    )
else:
    model = DecisionTreeRegressor()
    model.fit(health_train_prepared, health_train_labels)

    health_predictions = model.predict(health_test_prepared)

    # post-process: negative value->0
    health_predictions[health_predictions < 0] = 0
    mse = mean_squared_error(health_test_labels, health_predictions)
    pred_df = pd.DataFrame({'prediction': health_predictions, 'ground truth': health_test_labels})
    pred_fig = alt.Chart(pred_df).mark_point().encode(
        alt.X('ground truth'),
        alt.Y('prediction'),
    ).properties(
        height=600
    )
    tree_feature_df = pd.DataFrame({"importance Score": model.feature_importances_, 'Features': list(column_names)})

    weight_fig = alt.Chart(tree_feature_df).mark_bar().encode(
        alt.X('importance Score'),
        alt.Y("Features", sort='-x')
    ).properties(
        height=600
    )

ref = alt.Chart(pd.DataFrame({'x':[0, max(health_test_labels)], 'y':[0, max(health_test_labels)]})).mark_line(color='red').encode(
    alt.X('x'),
    alt.Y('y'),
)
col_model1, col_model2 = st.columns(2)
with col_model1: 
    st.altair_chart(pred_fig+ref, True)
with col_model2:
    st.altair_chart(weight_fig, True)

health_train_prepared_named = pd.DataFrame(health_train_prepared.toarray(), columns=column_names)
display = PartialDependenceDisplay.from_estimator(model, health_train_prepared_named, ['GDP per Capita', 'working_hr'])
display.plot()
st.pyplot(plt.gcf())
st.subheader('try the model on your own data')
GDP = st.number_input('GDP per Capita', 0.0, value=10000.0)
working_hr = st.number_input('Average Working hours per week', 0.0, value=40.0)
age_group = st.selectbox('age_group', [each.split('_')[-1] for each in column_names[4:]])
sex = st.selectbox('sex', [each.split('_')[-1] for each in column_names[2:4]])
if st.button('Predict!'):
    custom_data = pd.DataFrame({'Sex': [sex], 'Age Group': [age_group], 'GDP per Capita': [GDP], 'working_hr': [working_hr]})
    feature_vector = full_pipeline.transform(custom_data)
    pred = model.predict(feature_vector)[0]
    if pred < 0:
        pred = 0.0
    st.write(f"The predicted number of deaths per 100,000 people is {pred:.4}")
    
    #################### lime ####################
    explainer = lime_tabular.LimeTabularExplainer(training_data=health_train_prepared.toarray(), feature_names=list(column_names), mode='regression')
    exp = explainer.explain_instance(data_row=feature_vector.toarray().reshape(-1), predict_fn=model.predict)
    exp.save_to_file('exp.html')
    with open('exp.html', 'r', encoding='utf-8') as f:
        html_str = f.read()
    st.components.v1.html(html_str, scrolling=True, height=400, width=925)
    # st.write(fig)
    # st.pyplot()
    # plt.clf()
    # st.markdown(exp.as_html(), unsafe_allow_html=True)
    ######################################################################

    if model_name == 'Decision Tree':
        explainer = shap.TreeExplainer(model)
    else: 
        explainer = shap.LinearExplainer(model, health_train_prepared)
    shap_values = explainer.shap_values(feature_vector.toarray().reshape(1, -1))

    plt.figure()
    shap.force_plot(explainer.expected_value, shap_values[0], feature_vector.toarray(), matplotlib=True, feature_names=column_names)
    f = plt.gcf()
    st.pyplot(f)