import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from vega_datasets import data
import os
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
# st.set_page_config(layout="centered")
# step 0 : read in supportive data and cleaning
GDP_df = pd.read_csv('./supportive data/GDP.csv')
GDP_df = GDP_df[['Country or Area', 'Year', "GDP per Capita"]]

# select average working hours for full-time employment across all ages. 
working_hr_df = pd.read_csv('./supportive data/average working hours.csv')
# working_hr_df = working_hr_df[(working_hr_df['SEX']=='MEN') | (working_hr_df['SEX']=='WOMEN')]
working_hr_df = working_hr_df[(working_hr_df['JOBTYPE']=='FT') & (working_hr_df['EMPSTAT']=='TE')]
working_hr_df = working_hr_df[(working_hr_df['Age']=='Total')]
working_hr_df = working_hr_df[['COUNTRY', 'Sex', 'Time', 'Value']]
working_hr_df.columns = ['country', 'sex', 'year', 'working_hr']
working_hr_df['sex'] = working_hr_df['sex'].replace({'Men': 'Male', 'Women': 'Female', 'All persons': 'All'})

# st.write(working_hr_df)

# step 1: choose disease to explore
st.subheader('From the perspective of the disease')


disease_name_list = [each.split('.csv')[0] for each in os.listdir('./data') if each[-4:]=='.csv']
disease = st.sidebar.selectbox("Disease you want to explore: ", disease_name_list)

disease_df = pd.read_csv(f'./data/{disease}.csv', index_col=False, skiprows=6)
disease_df = disease_df[(disease_df['Age group code']!='Age_unknown') & (disease_df['Sex']!='Unknown')]
# disease_df = disease_df[(disease_df['Age group code']!='Age_all') & (disease_df['Sex']!='All')]
disease_df = disease_df[['Region Name','Country Code', 'Country Name','Year','Sex', 'Age Group', "Death rate per 100 000 population"]]

# year = st.sidebar.selectbox('Year: ', list(range(disease_df['Year'].min(), disease_df['Year'].max()+1)), index=len(list(range(disease_df['Year'].min(), disease_df['Year'].max()+1)))-2)
year = st.sidebar.slider('Year', disease_df["Year"].min(), disease_df["Year"].max(), value=2020)

age_group = st.sidebar.selectbox('Age Group: ', disease_df['Age Group'].unique())
sex = st.sidebar.selectbox('Sex: ', disease_df['Sex'].unique())

default_ix = list(disease_df['Country Name'].unique()).index('United States of America')


df_select = disease_df[(disease_df['Year'] == int(year)) & (disease_df['Age Group'] == age_group) & (disease_df['Sex'] == sex)]

# Country Plot reference to: https://nbviewer.org/github/bast/altair-geographic-plots/blob/fc9c036/choropleth.ipynb
# country_codes = pd.read_csv("./supportive data/ISO-3166-Countries-with-Regional-Codes.csv")
countries = alt.topo_feature(data.world_110m.url, 'countries')
# background = alt.Chart(countries).mark_geoshape(fill="lightgray")

# # st.write(df_select)
# # st.write(country_codes)
# # st.write(countries)

# foreground = (
#     alt.Chart(countries)
#     .mark_geoshape()
#     .transform_lookup(
#         lookup="id",
#         from_=alt.LookupData(data=country_codes, key="country-code", fields=["name"]),
#     )
#     .transform_lookup(
#         lookup="name",
#         from_=alt.LookupData(data=df_select, key="Country Name", fields=["Death rate per 100 000 population"]),
#     )
#     .encode(
#         fill=alt.Color(
#             "Death rate per 100 000 population:Q",
#             scale=alt.Scale(scheme="reds"),
#         )
#     )
# )
# world_map = (
#     (background + foreground)
#     .project(
#         type="equalEarth",
#     )
# )

# st.altair_chart(world_map, use_container_width=True)


fig = px.choropleth(df_select, locations="Country Code",
                    color="Death rate per 100 000 population", # lifeExp is a column of gapminder
                    hover_name="Country Name", # column to add to hover information
                    color_continuous_scale=px.colors.sequential.OrRd,
                    title=f'Total number of reported deaths in {sex} at age {age_group} caused by {disease} in {year}',
                    width=900,
                    )
fig.update_geos(
    showcountries=True, countrycolor="Black",
)
# fig.update_traces(marker_line_width = 4)
fig.update_layout(
        margin=dict(l=0, r=0, b=0, pad=0),
        coloraxis_colorbar_lenmode = "fraction",
        coloraxis_colorbar_len = 0.9,
        coloraxis_colorbar_y = 0.5
    )

st.plotly_chart(fig)

country_rank = alt.Chart(df_select.sort_values('Death rate per 100 000 population', ascending=False).iloc[:10, :]).mark_bar().encode(
    alt.X('Death rate per 100 000 population'),
    alt.Y('Country Name', sort='-x'),
    color='Region Name'
).properties(
    title=f'Top 10 countries with the highest death rate in {year}, {age_group} year-old {sex} '
)

st.altair_chart(country_rank, use_container_width=True)

df_whole = disease_df.merge(GDP_df, how='left', left_on=['Country Name', 'Year'], right_on=['Country or Area', 'Year'])
df_whole = df_whole.drop(columns=['Country or Area'])

# df_whole = df_whole.fillna(df_whole['GDP per Capita'].mean())
df_whole = df_whole.merge(working_hr_df, how='left', left_on=['Country Code', 'Sex', 'Year'], right_on=['country', 'sex', 'year'])
df_whole = df_whole.drop(columns=['country', 'sex', 'year'])
# df_whole = df_whole.fillna(df_whole['working_hr'].mean())

whole_select = df_whole[(df_whole['Year'] == int(year)) & (df_whole['Age Group'] == age_group)  & (df_whole['Sex'] == sex)]
col1, col2 = st.columns(2, gap='large')

# st.write(whole_select)
if whole_select['GDP per Capita'].isna().sum() == len(whole_select['GDP per Capita']):
    st.write('GDP data not available for these criterons.')
else:
    rate_GDP = alt.Chart(whole_select).mark_point().encode(
        alt.X('GDP per Capita'),
        alt.Y('Death rate per 100 000 population'),
        alt.Color('Region Name'),
        alt.Tooltip(['Country Name', 'GDP per Capita', 'working_hr', 'Death rate per 100 000 population'])
    ).properties(
        title='Death rate vs GDP per Capita'
    )

    st.altair_chart(rate_GDP, use_container_width=True)
if whole_select['working_hr'].isna().sum() == len(whole_select['working_hr']):
    st.write('Working Hours data not available for these criterons.')
else:
    rate_hr = alt.Chart(whole_select).mark_point().encode(
        alt.X('working_hr', scale=alt.Scale(domain=[35, 55])),
        alt.Y('Death rate per 100 000 population'),
        alt.Color('Region Name'),
        alt.Tooltip(['Country Name', 'GDP per Capita', 'working_hr', 'Death rate per 100 000 population'])
    ).properties(
        title='Death rate vs average weekly working hours'
    )
    st.altair_chart(rate_hr, use_container_width=True)
