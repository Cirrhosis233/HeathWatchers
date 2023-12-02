import streamlit as st
import pandas as pd
import altair as alt
import os
import plotly.express as px
import matplotlib.pyplot as plt
import warnings
import pycountry

warnings.simplefilter(action='ignore', category=UserWarning)
# st.set_page_config(layout="centered")

# Load dataset
disease_name_list = [each.split('.csv')[0] for each in os.listdir('./data') if each[-4:]=='.csv']

# On page: 1. choose disease to explore
# disease = st.sidebar.selectbox("Disease you want to explore: ", disease_name_list)
disease = st.selectbox('Disease you want to explore: ', disease_name_list)
disease_df = pd.read_csv(f'./data/{disease}.csv', index_col=False, skiprows=6)

# Fix disease, data of all sex and age group over the years
disease_all_df = disease_df[(disease_df['Age group code']=='Age_all') & (disease_df['Sex']=='All')]
disease_all_df.drop(disease_df.columns[[-4, -3, -2]], axis = 1, inplace = True)
# Fix disease, data of separate sex and age groups over the years
disease_df = disease_df[(disease_df['Age group code']!='Age_unknown') & (disease_df['Sex']!='Unknown')]
disease_df = disease_df[(disease_df['Age group code']!='Age_all') & (disease_df['Sex']!='All')]

# change Age Group code to numeric
mapping = {'Age00': 0, 'Age01_04': 1, 'Age05_09': 2, 'Age10_14': 3, 'Age15_19': 4, 'Age20_24': 5, 'Age25_29': 6, 'Age30_34': 7, 'Age35_39': 8, 'Age40_44': 9, 'Age45_49': 10, 'Age50_54': 11, 'Age55_59': 12, 'Age60_64': 13, 'Age65_69': 14, 'Age70_74': 15, 'Age75_79': 16, 'Age80_84': 17, 'Age85_over': 18}
disease_df['Age group code'] = disease_df['Age group code'].apply(lambda x:mapping[x])

# On sidebar: 1. select country to explore
default_ix = list(disease_df['Country Name'].unique()).index('United States of America')
country = st.sidebar.selectbox('Country: ', disease_df['Country Name'].unique(), index=default_ix)


disease_df = disease_df[['Region Name','Country Code', 'Country Name','Year','Sex', 'Age Group', 'Age group code', "Death rate per 100 000 population"]]
df_select = disease_df[disease_df['Country Name'] == country]
st.subheader('From the perspective of the country')

fig = px.sunburst(df_select, path=['Sex', 'Age Group', 'Year'], values='Death rate per 100 000 population', color='Age Group')
st.plotly_chart(fig)

# On page: 2. select age group. [year vs. death rate]
age_group = st.selectbox('Select an age group: ', df_select['Age Group'].unique(), index=7)

death_year = alt.Chart(df_select[df_select['Age Group'] == age_group]).mark_line().encode(
    alt.X('Year'),
    alt.Y('Death rate per 100 000 population'),
    alt.Color('Sex')
)
st.altair_chart(death_year, use_container_width=True)

# On page: 3. select year. [age group vs. death rate]
year = st.selectbox('Select a year: ', list(range(disease_df['Year'].min(), disease_df['Year'].max()+1)), index=len(list(range(disease_df['Year'].min(), disease_df['Year'].max()+1)))-2)
if len(df_select[df_select['Year'] == year]) == 0:
    st.markdown(f"No death rate data available for :red[{country}] in year :blue[{year}].")
else:
    sort_result = st.checkbox('Sort on Death Rate')
    if sort_result:
        death_age = alt.Chart(df_select[df_select['Year'] == year]).mark_bar().encode(
            alt.X('Age Group', sort='-y'),
            alt.Y('Death rate per 100 000 population'),
            alt.Color('Sex')
        )
    else:
        death_age = alt.Chart(df_select[df_select['Year'] == year]).mark_bar().encode(
            alt.X('Age Group', sort=alt.EncodingSortField(field="Age group code", order='ascending')),
            alt.Y('Death rate per 100 000 population'),
            alt.Color('Sex')
        )
    st.altair_chart(death_age, use_container_width=True)

# On page: 4. supportive dataset - doc and nurse per capita
# Read and combine doctor & nurse per 1000 population dataset
tmp_doc_df = pd.read_csv('supportive data/Doctors_Per_Capital_By_Country.csv', index_col=False)[["LOCATION", "TIME", "Value"]]
tmp_nurse_df = pd.read_csv('supportive data/Nurses_Per_Capital_By_Country.csv', index_col=False)[["LOCATION", "TIME", "Value"]]
doc_nurse_df = tmp_doc_df.merge(tmp_nurse_df, left_on=['LOCATION', 'TIME'], right_on=['LOCATION', 'TIME'], how='outer').dropna()
doc_nurse_df["Value"] = (doc_nurse_df["Value_x"] + doc_nurse_df["Value_y"]) * 100
# join with death rate for all sex and age groups
death_med_df = doc_nurse_df.merge(disease_all_df[['Country Code', 'Year', 'Death rate per 100 000 population']], 
                   left_on=['LOCATION', 'TIME'], right_on=['Country Code', 'Year'], how='inner')

tmp1, tmp2 = pycountry.countries.get(official_name=country), pycountry.countries.get(name=country)
country_code = tmp1.alpha_3 if tmp1 else tmp2.alpha_3

cur_df = death_med_df[death_med_df["LOCATION"] == country_code]
if len(cur_df) == 0:
    st.markdown(f"No doctor and nurse data available for :red[{country}].")
else:
    # dual y-axis chart
    st.markdown("Death rate and physicians availability over the years")
    base = alt.Chart(cur_df, title="Number Per 100 000 Population").encode(
        alt.X('Year:N').axis(title='Year', grid=True)
    )
    line1 = base.mark_line(stroke='#b01e1e').encode(
        alt.Y('Death rate per 100 000 population', axis=alt.Axis(title='Death Rate', titleColor='#b01e1e')),
    )
    line2 = base.mark_line(stroke='#57A44C').encode(
        alt.Y('Value', axis=alt.Axis(title='Number of Doctor & Nurse', titleColor='#57A44C'))
    )
    death_med = alt.layer(line1, line2).resolve_scale(
        y='independent'
    )
    # correlation chart
    corr = alt.Chart(cur_df, title="Number Per 100 000 Population").mark_point(size=40).encode(
        alt.X('Death rate per 100 000 population', scale=alt.Scale(domain=[cur_df['Death rate per 100 000 population'].min(), cur_df['Death rate per 100 000 population'].max()])).axis(title='Death Rate'),
        alt.Y('Value', scale=alt.Scale(domain=[cur_df['Value'].min(), cur_df['Value'].max()])).axis(title='Number of Doctor & Nurse')
    )
    st.altair_chart(death_med, use_container_width=True)
    st.altair_chart(corr, use_container_width=True)