from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import lxml.html
import re
from sklearn.linear_model import LinearRegression

pd.options.display.max_columns = None

CO2_df = pd.read_csv('API_19_DS2_en_csv_v2_10515758.csv')
CO2_df.rename(columns={CO2_df.columns[0]: "Country_Name", CO2_df.columns[2]: "Indicator_Name", CO2_df.columns[3]:
    "Indicator_Code"}, inplace=True)

df_CO2 = CO2_df.loc[CO2_df['Indicator_Code'] == "EN.ATM.CO2E.KT"].reset_index(drop=True)
df_CO2 = df_CO2.loc[:, ['Country_Name', '2014']]
df_CO2.columns = ['Country', '2014_CO2_emission']
country_og = pd.read_csv('Metadata_Country_API_NY.GDP.MKTP.CD_DS2_en_csv_v2_10515210.csv')
country = country_og.loc[:, ['TableName', 'IncomeGroup']]
country.columns = ['Country', 'IncomeGroup']
Income_CO2 = pd.merge(country, df_CO2, on='Country')
Group_Income = Income_CO2['2014_CO2_emission'].groupby(Income_CO2['IncomeGroup'])
print(Group_Income.sum())
print(Group_Income.min())
print(Group_Income.max())
# These figures demonstrate that the more economically advanced the overall amount of carbon dioxide emitted,
# the greater the emissions

df_Arg_CO2 = CO2_df.loc[(CO2_df['Country_Name'] == "Argentina") & (CO2_df['Indicator_Code'] == "EN.ATM.CO2E.KT")].\
    reset_index(drop=True)
Arg_CO2 = df_Arg_CO2.loc[0, '1962':'2014']

df_Arg_Pop = CO2_df.loc[(CO2_df['Country_Name'] == "Argentina") & (CO2_df['Indicator_Code'] == "SP.POP.TOTL")].\
    reset_index(drop=True)
Arg_Pop = df_Arg_Pop.loc[0, '1962':'2014']

GDP_df = pd.read_csv('API_NY.GDP.MKTP.CD_DS2_en_csv_v2_10515210.csv')
GDP_df.rename(columns={GDP_df.columns[0]: "Country_Name", GDP_df.columns[2]: "Indicator_Name", GDP_df.columns[3]:
    "Indicator_Code"}, inplace=True)
df_Arg_GDP = GDP_df.loc[(GDP_df['Country_Name'] == "Argentina")].reset_index(drop=True)
Arg_GDP = df_Arg_GDP.loc[0, '1962':'2014']
Arg_summary = pd.DataFrame({'A': Arg_GDP/Arg_Pop, 'B': Arg_CO2*1000/Arg_Pop})
fig, ax = plt.subplots()
ax2 = ax.twinx()
Arg_summary.A.plot(ax=ax, style='b-')
Arg_summary.B.plot(ax=ax2, style='r-', secondary_y=True)
print(Arg_summary)
plt.show()

Arg_summary['A'] = Arg_summary['A'].astype('int')
Arg_summary['B'] = Arg_summary['B'].astype('int')
print(Arg_summary.corr())
# Argentina's example demonstrates that CO2 does indeed correlate with GDP on a time dimension



def check_link(url):
    """
    Check the validity of the url and get its content.
    :param url: Target url address
    :return: The target website XML tree or error reminder.
    """
    try:

        r = requests.get(url)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        print('Connection Failed！！！')


def get_contents(ulist, rurl):
    """
    Find all the data in the form of the web (tr, td)
    :param ulist: A list storing contents of the form.
    :param rurl: The target url XML tree.
    :return: Edited ulist.
    """
    soup = BeautifulSoup(rurl, 'lxml')
    trs = soup.find_all('tr')
    for tr in trs:
        ui = []
        for td in tr:
            ui.append(td.string)
        ulist.append(ui)
    del ulist[0:2]
    del ulist[-1]


def creat_df(urlist):
    """
    Export the list to a DataFrame which is convenient for next step.
    :param urlist: The list which is needed to change to DataFrame.
    :return: A new DataFrame from the list.
    """
    country = []
    head = []
    for i in range(len(urlist)):
        country.append(urlist[i][2])
        head.append(urlist[i][4])
    df = pd.DataFrame.from_dict({"Country": country, "Head": head})
    return df

urli = []
url = "https://www.drovers.com/article/world-cattle-inventory-ranking-countries-fao"
rs = check_link(url)
get_contents(urli, rs)
Cattle = creat_df(urli)

vehicles_df_og = pd.read_csv('Motor vehicles per 1000 people.csv')
vehicles_df = vehicles_df_og.drop(columns='Date', axis=1)

df_Pop = CO2_df.loc[CO2_df['Indicator_Code'] == "SP.POP.TOTL"].reset_index(drop=True)
df_Pop = df_Pop.loc[:, ['Country_Name', '2014']]
df_Pop.columns = ['Country', 'Population']

df_Forest = CO2_df.loc[CO2_df['Indicator_Code'] == "AG.LND.FRST.ZS"].reset_index(drop=True)
df_Forest = df_Forest.loc[:, ['Country_Name', '2014']]
df_Forest.columns = ['Country', 'Forest_Land_Percentage_of_all_land']

result1 = pd.merge(df_Pop, df_Forest, on='Country')
result = pd.merge(pd.merge(vehicles_df, Cattle, on='Country'), result1, on='Country')
result['Head'] = result['Head'].str.replace(',', '')
result['Head'] = result['Head'].astype('int')
result['Amount'] = result['Amount'].astype('int')
result.eval('Heads_per_thousand_people = Head*1000/Population', inplace=True)
result = result.drop(columns=['Head', 'Population'], axis=1)
result.rename(columns={'Amount': 'Number_of_cars_per_thousand_people'})
# print(result)

co2_emission = CO2_df[CO2_df['Indicator_Code'] == 'EN.ATM.CO2E.KT']
co2_emission_2014 = co2_emission[['Country_Name','2014']]

population = CO2_df[CO2_df['Indicator_Code'] == 'SP.POP.TOTL']
population_2014 = population[['Country_Name','2014']]

co2_emission_2014 = co2_emission_2014.rename(columns={'2014':'2014_co2'})
population_2014 = population_2014.rename(columns= {'2014': '2014_pop'})
co2_per_thousand_df = pd.merge(co2_emission_2014, population_2014, on = 'Country_Name')


per_thousand_co2_em = (co2_per_thousand_df['2014_co2']/co2_per_thousand_df['2014_pop'])*1000000
co2_per_thousand_df['co2_per_thousand'] = per_thousand_co2_em

co2_per_thousand_df.dropna()


#developed countries:
developed_url = 'http://worldpopulationreview.com/countries/developed-countries/'
r = requests.get(developed_url)
tree = lxml.html.fromstring(r.content)
developed_country = tree.xpath('//li/a/text()')
developed_country = developed_country[8:]

#developing countries:
developing_url = 'https://isge2018.isgesociety.com/registration/list-of-developing-countries/'
r2 = requests.get(developing_url)
tree2 = lxml.html.fromstring(r2.content)
developing_country = tree2.xpath('//div/ul/li/text()')
developing_country = developing_country[4:]
real_name = []
for country_name in developing_country:
    real_name.append(country_name[:-1])
developing_country = real_name

developed_country_co2 = co2_per_thousand_df[co2_per_thousand_df['Country_Name'].isin(developed_country)]

developing_country_co2 = co2_per_thousand_df[co2_per_thousand_df['Country_Name'].isin(developing_country)]
result.rename(columns={"Country": "Country_Name", "Amount": "Number_of_cars_per_thousand_people"}, inplace=True)

developed_co2_cattle_vehicle_forest = pd.merge(result, developed_country_co2, on='Country_Name')


model = LinearRegression()
cor_matrix = developed_co2_cattle_vehicle_forest.corr()

model.fit(developed_co2_cattle_vehicle_forest[['Number_of_cars_per_thousand_people',
                                               'Forest_Land_Percentage_of_all_land',
                                               'Heads_per_thousand_people']],
          developed_co2_cattle_vehicle_forest['co2_per_thousand'])

#dataframe for calculating correlation:
cor_df = developed_co2_cattle_vehicle_forest[['co2_per_thousand',
                                              'Number_of_cars_per_thousand_people',
                                              'Forest_Land_Percentage_of_all_land',
                                              'Heads_per_thousand_people']]
cor_matrix1 = cor_df.corr()
print(cor_matrix1)
# Correlation coefficient matrix1

developing_co2_cattle_vehicle_forest = pd.merge(result,developing_country_co2, on='Country_Name')
cor_df2 = developing_co2_cattle_vehicle_forest[['co2_per_thousand',
                                              'Number_of_cars_per_thousand_people',
                                              'Forest_Land_Percentage_of_all_land',
                                              'Heads_per_thousand_people']]

cor_matrix2 = cor_df2.corr()
print(cor_matrix2)
# Correlation coefficient matrix. The difference between matrix1 and matrix2 imply that same factors will have
# different correlation with CO2 emission in countries with different economical levels.

developing_co2_cattle_vehicle_forest = developing_co2_cattle_vehicle_forest.dropna()
model2 = LinearRegression()
model2.fit(developing_co2_cattle_vehicle_forest[['Number_of_cars_per_thousand_people',
                                               'Forest_Land_Percentage_of_all_land',
                                               'Heads_per_thousand_people']],
           developing_co2_cattle_vehicle_forest['co2_per_thousand'])

print("developed:", model.coef_, model.intercept_)
print("developing:", model2.coef_, model2.intercept_)
# Coefficient and intercept of multivariable linear regression equation are obtained





