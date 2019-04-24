from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.max_columns = None

CO2_df = pd.read_csv('API_19_DS2_en_csv_v2_10515758.csv')
CO2_df.rename(columns={CO2_df.columns[0]: "Country_Name", CO2_df.columns[2]: "Indicator_Name", CO2_df.columns[3]:
    "Indicator_Code"}, inplace=True)

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
# print(Arg_summary)

fig, ax = plt.subplots()
ax2 = ax.twinx()
Arg_summary.A.plot(ax=ax, style='b-')
Arg_summary.B.plot(ax=ax2, style='r-', secondary_y=True)

# plt.show()


def check_link(url):
    try:

        r = requests.get(url)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        print('Connection Failed！！！')


def get_contents(ulist, rurl):
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
result.eval('Heads_per_thousand_people = Head*1000/Population', inplace=True)
result = result.drop(columns=['Head', 'Population'], axis=1)
result.rename(columns={'Amount': 'Number_of_cars_per_thousand_people'})
print(result)


