import streamlit as st
from PIL import Image
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from vega_datasets import data
import os
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

st.title('Team Name: HealthWatchers')
st.write("In this project, we collected death statistics for 26 different diseases.\
        Combining with some supportive data, including GDP per Capita and average working hour, \
        we first draw plots to find interesting correlations between the death rate and other factors.\
        And then, we fit different machine learning models to predict the death rate for the future \
        to alert people and authorites about the potential risk of diseases. ")
st.write("In this dashboard, we should some interesting trend found throught the process.\
        You can do your own experiments by exploring the other pages.")

st.divider()

st.header("Interactive Worldwide Trending")

st.markdown("The following charts are interactive charts that showcase the worldwide trending of death rate by Sex and Country of certain disease in certain year. By two paralleling bar charts, you can easily tell the difference between males' and females' death rates for certain diseases, and whether these trends apply to all countries and all years.")

# step 0 : read in supportive data and cleaning
population_df = pd.read_csv('./supportive data/Population.csv')

# step 1: choose disease to explore
disease_name_list = [each.split('.csv')[0] for each in os.listdir('./data') if each[-4:]=='.csv']
c1, emp, c2 = st.columns(3)
with c1:
        disease = st.selectbox("Disease you want to explore: ", disease_name_list)

disease_df = pd.read_csv(f'./data/{disease}.csv', index_col=False, skiprows=6)
disease_df = disease_df[(disease_df['Age group code']!='Age_unknown') & (disease_df['Sex']!='Unknown')]
disease_df = disease_df[(disease_df['Age group code']!='Age_all') & (disease_df['Sex']!='All')]
disease_df = disease_df[['Region Name','Country Code', 'Country Name','Year','Sex', 'Age Group', "Death rate per 100 000 population"]]

# 处理disease_df
# 把disease_df中不在population_df的country的row全部drop掉
country_codes_disease = set(disease_df['Country Code'].unique())
country_codes_population = set(population_df['LOCATION'].unique())
common_country_codes = country_codes_disease.intersection(country_codes_population)
# 只保留 disease_df 中有对应国家代码的行
disease_df = disease_df[disease_df['Country Code'].isin(common_country_codes)]

# 处理population_df
# 只保留 population_df 中有对应国家代码的行
population_df = population_df[population_df['LOCATION'].isin(common_country_codes)]
# 只保留 AGE 列值为 TOTAL 的行, 已经M和F的行
population_df = population_df[population_df['AGE'] == 'TOTAL']
population_df = population_df[population_df['SEX'] != 'T']
# 替换 'SEX' 列中的值
population_df['SEX'] = population_df['SEX'].replace({'W': 'Female', 'M': 'Male'})


# step 2: choose year to explore
# year = st.sidebar.selectbox('Year: ', list(range(2010, 2022)), index=len(list(range(2010, 2022)))-2)
with c2:
        year = st.slider('Year', 2010, 2021, 2020)


# age_group = st.sidebar.selectbox('Age Group: ', disease_df['Age Group'].unique(), index=7)
# sex = st.sidebar.selectbox('Sex: ', disease_df['Sex'].unique())

# step 3: combine the datasets
# 根据选定的年份过滤数据
disease_df_year = disease_df[disease_df['Year'] == year]
population_df_year = population_df[population_df['TIME'] == year]
# 汇总每个国家所有年龄组的死亡率 per 100,000 人
country_deaths_rate = disease_df_year.groupby(['Country Code', 'Sex'])['Death rate per 100 000 population'].sum()
# 重置索引，以便于后续的合并操作
country_deaths_rate = country_deaths_rate.reset_index()


# 合并死亡率数据和人口数据
merged_df = pd.merge(country_deaths_rate, population_df, left_on=['Country Code', 'Sex'], right_on=['LOCATION', 'SEX'])
# 计算总死亡人数
merged_df['Estimated Deaths'] = (merged_df['Death rate per 100 000 population'] * merged_df['Value']) / 100000
# 查看 merged_df 中不同国家的数量
# num_countries = merged_df['Country Code'].nunique()
# print("Number of unique countries in merged_df:", num_countries)

# 汇总每个性别的全球死亡人数
global_deaths_by_sex = merged_df.groupby('Sex_x')['Estimated Deaths'].sum().reset_index()
# 获取每个性别的全球总人口数
global_population_by_sex = merged_df.groupby('Sex_x')['Value'].sum().reset_index()

# 计算死亡率
# 确保 'Estimated Deaths' 和 'Value' 列是数值类型
global_deaths_by_sex['Estimated Deaths'] = pd.to_numeric(global_deaths_by_sex['Estimated Deaths'], errors='coerce')
global_population_by_sex['Value'] = pd.to_numeric(global_population_by_sex['Value'], errors='coerce')

# 计算死亡率
death_rate = (global_deaths_by_sex['Estimated Deaths'] / global_population_by_sex['Value']) * 100000
death_rate = death_rate.reset_index().rename(columns={0: 'Death Rate per 100,000'})
# 添加性别标签
sex_labels = ['Female', 'Male']
death_rate['Sex'] = [sex_labels[i] for i in death_rate['index']]

#print('----\n','global_deaths_by_sex:', global_deaths_by_sex)
#print('----\n','global_population_by_sex:', global_population_by_sex)
# 创建柱状图
column1, column2 = st.columns(2)
bar_chart = alt.Chart(death_rate).mark_bar(size=60).encode(
    y='Sex:N',         # 性别作为 X 轴
    x='Death Rate per 100,000:Q',  # 死亡率作为 Y 轴
    color='Sex:N'      # 根据性别着色
).properties(
    title=f'Death Rate by Sex',
    height=600
)
with column1:
        st.altair_chart(bar_chart, use_container_width=True)

# st.write(merged_df)
# 对每个国家和性别的死亡人数进行汇总
country_sex_deaths = merged_df.groupby(['Country Code', 'Sex_x',]).agg(
    Total_Deaths=('Estimated Deaths', 'sum'),
    Total_Population=('Value', 'sum'),
    country=('Country', 'first'),
).reset_index()

# 计算死亡率
country_sex_deaths['Death Rate per 100,000'] = (country_sex_deaths['Total_Deaths'] / country_sex_deaths['Total_Population']) * 100000
# 重命名 'Sex_x' 列为 'Sex'
country_sex_deaths = country_sex_deaths.rename(columns={'Sex_x': 'Sex'})

# Streamlit 控件
# sort_order = st.checkbox('Sort by Death Rate')

# 基于用户选择排序
# if sort_order:
#     country_sex_deaths = country_sex_deaths.sort_values(by='Death Rate per 100,000', ascending=False)
country_sex_deaths = country_sex_deaths.sort_values(by='Death Rate per 100,000', ascending=False)
# st.write(country_sex_deaths)
# 创建柱状图
chart = alt.Chart(country_sex_deaths).mark_bar().encode(
    y=alt.X('Country Code:N', sort=None),
    x='Death Rate per 100,000:Q',
    color='Sex:N',
    tooltip=['country', 'Sex', 'Death Rate per 100,000']
).properties(
    title=f'Death Rate by Country and Sex',
    height=600
)

# 显示图表
with column2:
        st.altair_chart(chart, use_container_width=True)

st.markdown("Had fun with our charts? Sometimes you'll need to combine other interactive charts with more details to find out the potential reason behind the pattern.")

st.markdown("**Alzheimer Example**")

st.markdown("If you choose **Alzheimer** from the above list of diseases, the first thing you will notice is that the death rate is higher for females, and you are wondering why.")

st.markdown("Now go to the `Country Focus` page through the side bar, and choose **Alzheimer** on the filter, notice the trend in the first sunburst graph.")

st.markdown("Then do you have any ideas of why Females have higher death rate than Males for Alzheimer?")

with st.expander("See explanation"):
        image = Image.open('./img/front page/alzheimer_1.png')
        st.image(image, use_column_width=True)
        st.markdown("The main reason for this greater risk is because women live longer than men and old age is the biggest risk factor for this disease.")

st.divider()

st.header('EDA')
# st.write("TODO")

# EDA on Country page
st.subheader("Explore Death Rate Based on Country Focus")
st.markdown("We first explored the relationship between death rate(per 100,000 population) and age factor, then observe the change of death rate for \
            a specific age group over the years. After that, we use the supportive dataset of number of doctors and nurses per capita to expolore \
            potential insights.")
st.markdown("The left figure shows the death rate in US for people of age 60-64 dying of athsma, and we can see a general decreasing trend \
            after 2000. The right figure still chooses to explore mortality causes in the US, and this time it compares the death rate in year 2020 \
            for HIV between different age groups. The statistics shows a semi-normal distribution, and people of age 55-59 are the most prominent \
            age population that deceases due to athsma.")
col1, col2 = st.columns(2)
with col1:
       image = Image.open('./img/front page/us_year_athsma.png')
       st.image(image, use_column_width=True)
with col2:
      image = Image.open('./img/front page/us_age_hiv.png')
      st.image(image, use_column_width=True)

st.markdown("Using supportive dataset, we first map a dual-y-axis line plot showing change of death rate for all sex and all age groups over \
            the years and change of the doctor & nurse number over the years. Y axis denotes the number per 100,000 population. We're still \
            investigating the US for now. The statistics shows a special case where we found that death rate decreases over time while \
            number of doctors & nurses increases, and a linear correlation seem to exists between the two in the scatter plot. However, \
            that's not a common case as we explore other countries, the relationship between death rate and doctor & nurse number is random \
            and possesses no special correlation.")
col1, col2 = st.columns(2)
with col1:
       image = Image.open('./img/front page/us_country1.png')
       st.image(image, use_column_width=True)
with col2:
      image = Image.open('./img/front page/us_country2.png')
      st.image(image, use_column_width=True)


st.subheader("Explore Death Rate by Disease Focus")

st.markdown("In the Disease Focus Page, there are 4 types of graphs, we will use the `Disease`, `year` , `age group`, `sex` as our example case and introduce you to all the four charts.")

st.markdown("We first illustrate the trend for disease, alcohol overuse, in the year 2020, for all ages and all genders.")

col1, col2 = st.columns([0.2, 0.8])
with col1:
       image = Image.open('./img/front page/alcohol_2020_all_0.png')
       st.image(image, use_column_width=True)
with col2:
      image = Image.open('./img/front page/alcohol_2020_all_1.png')
      st.image(image, use_column_width=True)

st.markdown("As you can see, the first chart is a world map with a color bar representing the range of death rates, with darker colors representing higher death rates. The second chart is a horizontal bar chart, which is another representation of the death rate comparison among countries, sorted by death rate and only shows the top 10 countries. The color legend on the right shows the Region of the countries.")

st.markdown("#### What do we know till now?")
st.markdown("Before introducing the rest two graphs. Let's see what kind of knowledge we can learn from the first two charts about alcohol overuse in 2020. Well, obviously our datasets do not have data for all countries in the world. At least we can tell that around China and Africa, the data is clearly missing, but we can still dig out some interesting facts and hypotheses. First of all, from the world map, and the bar chart, it is clear that the death caused by alcohol overuse happened mostly in Europe.")


image = Image.open('./img/front page/alcohol_2020_all_2.png')
st.image(image, use_column_width=True)

image = Image.open('./img/front page/alcohol_2020_all_3.png')
st.image(image, use_column_width=True)

st.markdown("Now we have two more charts, and they bring us even more interesting visualizations. The first chart is the Death rate vs GDP, colored by Regions. Here we can compare the death rate among different countries by their GDP values. The second chart shows the death rate vs weekly working hours, also colored by Regions. How do workloads influence the death rate of certain diseases?")

st.markdown("#### What else do we find now?")
st.markdown("At first glance, you may think the dots are confusing. To our intuition, a higher GDP usually indicates better life quality and medical resources. And fewer working hours may related to a healthier life. But none of them applies here. The dots of death rate vs GDP is a mess, there is no clear patterns, and for working hours, we even see a decreasing trend: when working hour increases, the death rate decreases. However, considering our special death cause here: Alcohol Overuse. All the mysteries might make sense. As long as there is a steady supply of drinking alcohol, the overuse of alcohol may not related to the life quality of life or medical resources, but more about cultural and government control. Europe has the most developed countries (and many small countries, which in turn creates more dots and takes the majority of trends on graphs) in the world, and perhaps most of the countries like drinking alcohol, and people can afford a large amount of alcohol consumption. We know that many diseases (or accidents) related to alcohol cannot be cured, and all of the reasons combined together may give us the reason behind the strange trends.")

st.markdown("#### Can we find a similar pattern?")
st.markdown("To alcohol overuse? YES! Actually, there is a disease closely related to long-term alcohol drinking, and it showed very similar patterns: **Cirrhosis of the liver**. Let's apply the same filter and see how does the graphs look like.")

image = Image.open('./img/front page/alcohol_2020_all_4.png')
st.image(image, use_column_width=True)

image = Image.open('./img/front page/alcohol_2020_all_5.png')
st.image(image, use_column_width=True)

st.markdown("Very similar patterns compared to alcohol overuse! Almont has the same majority of groups of countries in Europe and the same patterns of working hours. The only major difference might be the trend in GDP, we can see that compared to alcohol overuse, cirrhosis of the liver shows a clear decreasing trend when the GDP value grows. This makes sense because cirrhosis of the liver is an actual disease compared to alcohol overuse, which is more like a combined death cause. Better medical resources indicated by higher GDP may help to reduce the death rate here.")

st.markdown("#### Does the pattern apply to all other diseases?")

st.markdown("No, the patterns observed in the case of alcohol overuse do not necessarily apply to all other diseases. Different diseases have unique characteristics, risk factors, and dynamics that influence their trends and patterns in populations. To illustrate this point, let's explore the trend for HIV (Human Immunodeficiency Virus), a significantly different disease in terms of its causes, transmission, and global impact.")

# HIV
image = Image.open('./img/front page/hiv_world_map.png')
st.image(image, use_column_width=True)

image = Image.open('./img/front page/hiv_country.png')
st.image(image, use_column_width=True)

st.markdown("Using HIV data from 2020 for all ages and sexes, we first see two charts: a world map color-coded to show death rates (darker colors indicate higher rates) and a horizontal bar chart displaying the top 10 countries by death rate, with a color legend indicating their regions.")

st.markdown("#### What do we know till now?")
st.markdown("Before delving into additional visualizations, we can extract some insights about HIV-related death rates in 2020 from the provided bar chart. The chart suggests a regional pattern; for instance, the majority of the top 10 countries with the highest death rates are from Central and South America, as indicated by the red bars. Mauritius stands out as the country with the highest death rate in this dataset, followed by Saint Vincent and the Grenadines. The presence of countries from Africa and North America and the Caribbean regions also indicates that the issue is not confined to a single area, although the data might not encompass all countries globally. ")

st.markdown("#### What else do we find now?")
image = Image.open('./img/front page/hiv_gdp.png')
st.image(image, use_column_width=True)

image = Image.open('./img/front page/hiv_working_hour.png')
st.image(image, use_column_width=True)

st.markdown("Analyzing the two additional charts for HIV, we observe intriguing insights regarding the relationship between HIV death rates and economic factors. The first scatter plot portrays the death rate versus GDP per capita, with distinct colors representing different regions. Notably, there is a trend where countries with lower GDP per capita have higher HIV death rates, particularly visible among African nations, indicating a potential link between economic status and HIV mortality. On the other hand, there is no clear pattern that emerges between HIV death rates and weekly working hours, suggesting that workload may not have a significant direct correlation with HIV mortality. These visuals align with WHO official statistics and disease pathology, showing that lower economic status often correlates with higher HIV mortality rates, while the lack of correlation between workload and HIV death rates fits with current understandings of the disease's dynamics.")

st.markdown("#### Can we find a similar pattern?")
st.markdown("Next, we turn our attention to Hepatitis B, analyzing data from the year 2020 that encompasses all age groups and genders. Hepatitis B, a disease often exacerbated by healthcare access and preventive measures, mirrors the pattern seen with economic factors similar to HIV.  Applying the same analysis criteria—considering all ages and genders in 2020—reveals that lower GDP correlates with higher death rates.  Let’s visualize this data to confirm the trend.")

image = Image.open('./img/front page/hp_world_map.png')
st.image(image, use_column_width=True)

image = Image.open('./img/front page/hp_country.png')
st.image(image, use_column_width=True)

image = Image.open('./img/front page/hp_gdp.png')
st.image(image, use_column_width=True)

image = Image.open('./img/front page/hp_working_hour.png')
st.image(image, use_column_width=True)

st.markdown("Similarly to the trends observed for HIV, the data for Hepatitis B suggests that countries with lower GDP per capita experience higher death rates. This trend aligns with the notion that economic constraints may limit access to healthcare and effective vaccination programs, which are crucial for controlling Hepatitis B. Moreover, the data indicates no significant correlation between working hours and Hepatitis B mortality rates, suggesting that, much like HIV, the transmission and progression of Hepatitis B are influenced by factors other than labor patterns.")

st.markdown("In conclusion, visualization plays a crucial role in providing critical insights into trends and correlations within complex datasets, as demonstrated by the patterns observed in the data for various diseases.")

st.divider()

st.header("Modeling")
st.markdown("For each disease, we used **Sex**, **Age**, **GDP**, and **Average Working Hours** as features\
            to train linear regression and regression tree models and predict the death per 100,000 population. Then, we plot the scatter plot between ground truth and \
            predicted value to evaluate the performance. Besides, we plot the weights/importance score and use Partial Dependecy to interpret model.\
            Also, we allow you to try our model by manually input features. Besides getting the prediction, we demonstrate the decision-making process of our model by using **lime** and **shap**.")

st.write("Take Endocrine Disorders as an example. We fit a decision tree model to the dataset.")
col1, col2 = st.columns(2)
with col1:
    image = Image.open('./img/front page/gt_pred.png')
    st.image(image, use_column_width=True)
with col2:
    image = Image.open('./img/front page/importance.png')
    st.image(image, use_column_width=True)
st.markdown("In the left figure, we can see the point lies around the line y=x, which means the prediction is close to the ground truth label. \
        The decision tree model makes reasonable predictions about the death rate. In the right figure, we can also see that **Age > 85 or not**, **GDP per Captia**, and **average working hour** are the most important factors.\
        To further explore their effects, we plot the partial dependency figure.")

image = Image.open('./img/front page/partial dependency.png')
st.image(image, use_column_width=True,caption='Partial Dependency of Endocrine Disorders--Decision Tree.')
st.markdown('From the partial dependency figure, we can see that both **GDP per Capita** and **Average Working Hours** have a positive impact to the death rate.\
            Given **Sex** and **Age**, the higher GPD/average working hour it is, the higher death rate we may observe. This observation makes sense, \
            as Endocrine Disorders might be caused by big working pressure and unhealthy dietary habits, which are positively correlated with GDP and average working hours.')
st.divider()
st.write("As you can image, these features may have different impact on different diseases.\
        For example, the higher the average working hour is, the lower the death rate of **alcohol overuse** is.\
        An interesting guess of the reason is that the busier people are, the less time they have to consume alcohol.")
image = Image.open('./img/front page/alcohol.png')
st.image(image, use_column_width=True, caption='Partial Dependency of Alcohol Overuse--Decision Tree.')
st.write("You can try the models on different disease on your own at the **Apply Model** page.")