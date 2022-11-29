import pandas as pd
from scipy.stats import chi2_contingency,fisher_exact, shapiro
import pingouin as pg
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.weightstats import ttest_ind
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pandas_profiling import ProfileReport

alex_df=pd.read_csv("ASD_responses.csv")

TAS_list=['(ALX)_1','(ALX)_2','(ALX)_3','(ALX)_4','(ALX)_5','(ALX)_6','(ALX)_7','(ALX)_8',
'(ALX)_9','(ALX)_10','(ALX)_11','(ALX)_12','(ALX)_13','(ALX)_14','(ALX)_15','(ALX)_16','(ALX)_17',
'(ALX)_18','(ALX)_19','(ALX)_20']
SAS_list=[ '(SAS)_1', '(SAS)_2', '(SAS)_3', '(SAS)_4',
'(SAS)_5', '(SAS)_6', '(SAS)_7', '(SAS)_8', '(SAS)_9', '(SAS)_10']
DASS_list=[ '(DASS)_1', '(DASS)_2', '(DASS)_3', '(DASS)_4', '(DASS)_5', '(DASS)_6', '(DASS)_7', '(DASS)_8',
'(DASS)_9', '(DASS)_10', '(DASS)_11', '(DASS)_12', '(DASS)_13', '(DASS)_14', '(DASS)_15', '(DASS)_16',
 '(DASS)_17', '(DASS)_18', '(DASS)_19', '(DASS)_20', '(DASS)_21']


#Create TAS, SAS, & DASS total scores
alex_df['TAS_total'] = alex_df[TAS_list].sum(axis=1)
alex_df['SAS_total']=alex_df[SAS_list].sum(axis=1)
alex_df['DASS_score']=alex_df[DASS_list].sum(axis=1)
alex_df["DASS_score"]=2*alex_df["DASS_score"]

#Function to determine whether Smartphone addicted
def categorise(row):
    if row['Sex']==1 and row['SAS_total'] >=31:
        return 1
    elif row['Sex']==2 and row['SAS_total'] >=33:
        return 1
    else:
        return 0
#Function to determine alexythemia pos/neg
def categorise2(row):
    if  row['TAS_total'] >=60:
        return 1
    else:
        return 0

def categorise3(row):
    if  row['DASS_score'] >=60:
        return 1
    else:
        return 0

#apply previous functions
alex_df['smart_addicted'] = alex_df.apply(lambda row: categorise(row), axis=1)
alex_df['alex_pos']=alex_df.apply(lambda row: categorise2(row), axis=1)
alex_df['DASS_severe']=alex_df.apply(lambda row: categorise3(row), axis=1)

TAS_move = alex_df.pop('TAS_total')
alex_df.insert(37,'TAS_total', TAS_move)
SAS_move=alex_df.pop('SAS_total')
alex_df.insert(48,'SAS_total', SAS_move)
DASS_move=alex_df.pop("DASS_score")
alex_df.insert(70,'DASS_score', DASS_move)


alex_df.rename(columns={'Frequency of smartphone change each year':'Frequency'}, inplace=True)
alex_df["Frequency"]=alex_df["Frequency"].astype(str)
alex_df.rename(columns={'Monthly smartphone bill':'month_bill','Total grade':'grade'
,"Maternal status":'marital',"How many hours use mobile?  ":"hour_usage"
,'How often do you use social media sites':'socmed_freq','Which of these is most frequently used':'platforms',\
"Do you pay for attractions offered on social media ":"pay_attract"}
, inplace=True)
alex_df.replace(to_replace =["2-Jan"," 1-2"], value ="1-2", inplace=True)
alex_df.replace(to_replace =["3-Feb"," 2-3"], value ="2-3", inplace=True)
alex_df = alex_df[alex_df['platforms'].notnull()]
alex_df["WA"] = alex_df['platforms'].str.contains("Whatsapp")
alex_df["IG"] = alex_df['platforms'].str.contains("Instagram")
alex_df["FB"] = alex_df['platforms'].str.contains("Facebook")
alex_df["SC"] = alex_df['platforms'].str.contains("Snapchat")
alex_df["tw"] = alex_df['platforms'].str.contains("twitter")
alex_df["platforms"] = alex_df.platforms.str.split(',')
alex_df['platform_count'] = alex_df['platforms'].str.len()
print(alex_df[["WA","IG","FB","SC","tw","platforms","platform_count"]].head(15))

country = []
for index, row in alex_df.iterrows():
    if 'O' in row['Nationality']:
        country.append(1)
    elif 'E' in row['Nationality']:
        country.append(2)
    else:
        country.append(3)
alex_df['nation_code'] = country


#Survey validity tests
TAS_cronbach_df=alex_df[TAS_list]
SAS_cronbach_df=alex_df[SAS_list]
DASS_cronbach_df=alex_df[DASS_list]
print("TAS Cronbach:",pg.cronbach_alpha(data=TAS_cronbach_df),"\n")
print("SAS Cronbach:",pg.cronbach_alpha(data=SAS_cronbach_df),"\n")
print("DASS Cronbach:",pg.cronbach_alpha(data=DASS_cronbach_df),"\n")

#Whatsapp SAS chisq
WA_male_addicted= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] >=31) & (alex_df["WA"]==True)]
WA_male_not= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] <31) & (alex_df["WA"]==True)]
WA_female_addicted= alex_df[(alex_df["Sex"] ==2) & (alex_df["SAS_total"] >=33)& (alex_df["WA"]==True)]
WA_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] <33)& (alex_df["WA"]==True)]
NotWA_male_addicted= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] >=31) & (alex_df["WA"]==False)]
NotWA_male_not= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] <31) & (alex_df["WA"]==False)]
NotWA_female_addicted= alex_df[(alex_df["Sex"] ==2) & (alex_df["SAS_total"] >=33)& (alex_df["WA"]==False)]
NotWA_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] <33)& (alex_df["WA"]==False)]

wa_df=[[len(WA_male_addicted)+len(WA_female_addicted),len(WA_male_not)+len(WA_female_not)],
[len(NotWA_male_addicted)+len(NotWA_female_addicted),len(NotWA_male_not)+len(NotWA_female_not)]]
statw, pw, dofw, expectedw = chi2_contingency(wa_df)
print(wa_df)
print("Whatsapp(or not) SAS addiction ChiSq p value=",statw, pw,"\n")

#Whatsapp TAS chisq
wa_alex=alex_df[(alex_df["WA"]==True) & (alex_df["TAS_total"] >=60)]
wa_alex_neg= alex_df[(alex_df["WA"]==True) & (alex_df["TAS_total"] <60)]
nowa_alex= alex_df[(alex_df["WA"] ==False) & (alex_df["TAS_total"] >=60)]
nowa_alex_neg= alex_df[(alex_df["WA"]==False) & (alex_df["TAS_total"] <60)]

wa_tas_df=[[len(wa_alex),len(wa_alex_neg)],
[len(nowa_alex),len(nowa_alex_neg)]]
statwa_tas, pwa_tas, dofwa_tas, expectedwa_tas = chi2_contingency(wa_tas_df)
print(wa_tas_df)
print("Whatsapp(or not) TAS ChiSq p value=",statwa_tas, pwa_tas,"\n")

#Whatsapp DASS chisq
wa_dass=alex_df[(alex_df["WA"]==True) & (alex_df["DASS_score"] >=60)]
wa_dass_neg= alex_df[(alex_df["WA"]==True) & (alex_df["DASS_score"] <60)]
nowa_dass= alex_df[(alex_df["WA"] ==False) & (alex_df["DASS_score"] >=60)]
nowa_dass_neg= alex_df[(alex_df["WA"]==False) & (alex_df["DASS_score"] <60)]

wa_dass_df=[[len(wa_dass),len(wa_dass_neg)],
[len(nowa_dass),len(nowa_dass_neg)]]
statwa_dass, pwa_dass, dofwa_dass, expectedwa_dass = chi2_contingency(wa_dass_df)
print(wa_dass_df)
print("Whatsapp(or not) DASS severe ChiSq p value=",statwa_dass, pwa_dass,"\n")

#Whatsapp Oman SAS chisq
omanWA_male_addicted= alex_df[alex_df["Sex"]==1 & (alex_df["nation_code"]==1) & (alex_df["SAS_total"] >=31) & (alex_df["WA"]==True)]
omanWA_male_not= alex_df[alex_df["Sex"]==1 & (alex_df["nation_code"]==1) &(alex_df["SAS_total"] <31) & (alex_df["WA"]==True)]
omanWA_female_addicted= alex_df[(alex_df["Sex"] ==2) &(alex_df["nation_code"]==1) & (alex_df["SAS_total"] >=33)& (alex_df["WA"]==True)]
omanWA_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["nation_code"]==1) & (alex_df["SAS_total"] <33)& (alex_df["WA"]==True)]
omanNotWA_male_addicted= alex_df[alex_df["Sex"]==1 & (alex_df["nation_code"]==1) & (alex_df["SAS_total"] >=31) & (alex_df["WA"]==False)]
omanNotWA_male_not= alex_df[alex_df["Sex"]==1 & (alex_df["nation_code"]==1) & (alex_df["SAS_total"] <31) & (alex_df["WA"]==False)]
omanNotWA_female_addicted= alex_df[(alex_df["Sex"] ==2)& (alex_df["nation_code"]==1) & (alex_df["SAS_total"] >=33)& (alex_df["WA"]==False)]
omanNotWA_female_not= alex_df[(alex_df["Sex"]==2)& (alex_df["nation_code"]==1) & (alex_df["SAS_total"] <33)& (alex_df["WA"]==False)]

omanwa_sas_df=[[len(omanWA_male_addicted)+len(omanWA_female_addicted),len(omanWA_male_not)+len(omanWA_female_not)],
[len(omanNotWA_male_addicted)+len(omanNotWA_female_addicted),len(omanNotWA_male_not)+len(omanNotWA_female_not)]]
statw, pw, dofw, expectedw = chi2_contingency(omanwa_sas_df)
print(omanwa_sas_df)
print("oman Whatsapp(or not) SAS addiction ChiSq p value=",statw, pw,"\n")

#oman Whatsapp TAS chisq
omanwa_alex=alex_df[(alex_df["WA"]==True) &(alex_df["nation_code"]==1)& (alex_df["TAS_total"] >=60)]
omanwa_alex_neg= alex_df[(alex_df["WA"]==True) &(alex_df["nation_code"]==1)& (alex_df["TAS_total"] <60)]
omannowa_alex= alex_df[(alex_df["WA"] ==False) &(alex_df["nation_code"]==1)& (alex_df["TAS_total"] >=60)]
omannowa_alex_neg= alex_df[(alex_df["WA"]==False) &(alex_df["nation_code"]==1)& (alex_df["TAS_total"] <60)]

omanwa_tas_df=[[len(omanwa_alex),len(omanwa_alex_neg)],
[len(omannowa_alex),len(omannowa_alex_neg)]]
statwa_tas, pwa_tas, dofwa_tas, expectedwa_tas = chi2_contingency(omanwa_tas_df)
print(omanwa_tas_df)
print("oman Whatsapp(or not) TAS ChiSq p value=",statwa_tas, pwa_tas,"\n")

#oman Whatsapp DASS chisq
omanwa_dass=alex_df[(alex_df["WA"]==True) &(alex_df["nation_code"]==1)& (alex_df["DASS_score"] >=60)]
omanwa_dass_neg= alex_df[(alex_df["WA"]==True) &(alex_df["nation_code"]==1)& (alex_df["DASS_score"] <60)]
omannowa_dass= alex_df[(alex_df["WA"] ==False) &(alex_df["nation_code"]==1)& (alex_df["DASS_score"] >=60)]
omannowa_dass_neg= alex_df[(alex_df["WA"]==False)&(alex_df["nation_code"]==1) & (alex_df["DASS_score"] <60)]

omanwa_dass_df=[[len(omanwa_dass),len(omanwa_dass_neg)],
[len(omannowa_dass),len(omannowa_dass_neg)]]
statwa_dass, pwa_dass, dofwa_dass, expectedwa_dass = chi2_contingency(omanwa_dass_df)
print(omanwa_dass_df)
print("oman Whatsapp(or not) DASS severe ChiSq p value=",statwa_dass, pwa_dass,"\n")

#Whatsapp egypt SAS chisq
egyptWA_male_addicted= alex_df[alex_df["Sex"]==1 & (alex_df["nation_code"]==2) & (alex_df["SAS_total"] >=31) & (alex_df["WA"]==True)]
egyptWA_male_not= alex_df[alex_df["Sex"]==1 & (alex_df["nation_code"]==2) &(alex_df["SAS_total"] <31) & (alex_df["WA"]==True)]
egyptWA_female_addicted= alex_df[(alex_df["Sex"] ==2) &(alex_df["nation_code"]==2) & (alex_df["SAS_total"] >=33)& (alex_df["WA"]==True)]
egyptWA_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["nation_code"]==2) & (alex_df["SAS_total"] <33)& (alex_df["WA"]==True)]
egyptNotWA_male_addicted= alex_df[alex_df["Sex"]==1 & (alex_df["nation_code"]==2) & (alex_df["SAS_total"] >=31) & (alex_df["WA"]==False)]
egyptNotWA_male_not= alex_df[alex_df["Sex"]==1 & (alex_df["nation_code"]==2) & (alex_df["SAS_total"] <31) & (alex_df["WA"]==False)]
egyptNotWA_female_addicted= alex_df[(alex_df["Sex"] ==2)& (alex_df["nation_code"]==2) & (alex_df["SAS_total"] >=33)& (alex_df["WA"]==False)]
egyptNotWA_female_not= alex_df[(alex_df["Sex"]==2)& (alex_df["nation_code"]==2) & (alex_df["SAS_total"] <33)& (alex_df["WA"]==False)]

egyptwa_sas_df=[[len(egyptWA_male_addicted)+len(egyptWA_female_addicted),len(egyptWA_male_not)+len(egyptWA_female_not)],
[len(egyptNotWA_male_addicted)+len(egyptNotWA_female_addicted),len(egyptNotWA_male_not)+len(egyptNotWA_female_not)]]
statw, pw, dofw, expectedw = chi2_contingency(egyptwa_sas_df)
print(egyptwa_sas_df)
print("egypt Whatsapp(or not) SAS addiction ChiSq p value=",statw, pw,"\n")

#egypt Whatsapp TAS chisq
egyptwa_alex=alex_df[(alex_df["WA"]==True) &(alex_df["nation_code"]==2)& (alex_df["TAS_total"] >=60)]
egyptwa_alex_neg= alex_df[(alex_df["WA"]==True) &(alex_df["nation_code"]==2)& (alex_df["TAS_total"] <60)]
egyptnowa_alex= alex_df[(alex_df["WA"] ==False) &(alex_df["nation_code"]==2)& (alex_df["TAS_total"] >=60)]
egyptnowa_alex_neg= alex_df[(alex_df["WA"]==False) &(alex_df["nation_code"]==2)& (alex_df["TAS_total"] <60)]

egyptwa_tas_df=[[len(egyptwa_alex),len(egyptwa_alex_neg)],
[len(egyptnowa_alex),len(egyptnowa_alex_neg)]]
statwa_tas, pwa_tas, dofwa_tas, expectedwa_tas = chi2_contingency(egyptwa_tas_df)
print(egyptwa_tas_df)
print("egypt Whatsapp(or not) TAS ChiSq p value=",statwa_tas, pwa_tas,"\n")

#egypt Whatsapp DASS chisq
egyptwa_dass=alex_df[(alex_df["WA"]==True) &(alex_df["nation_code"]==2)& (alex_df["DASS_score"] >=60)]
egyptwa_dass_neg= alex_df[(alex_df["WA"]==True) &(alex_df["nation_code"]==2)& (alex_df["DASS_score"] <60)]
egyptnowa_dass= alex_df[(alex_df["WA"] ==False) &(alex_df["nation_code"]==2)& (alex_df["DASS_score"] >=60)]
egyptnowa_dass_neg= alex_df[(alex_df["WA"]==False)&(alex_df["nation_code"]==2) & (alex_df["DASS_score"] <60)]

egyptwa_dass_df=[[len(egyptwa_dass),len(egyptwa_dass_neg)],
[len(egyptnowa_dass),len(egyptnowa_dass_neg)]]
statwa_dass, pwa_dass, dofwa_dass, expectedwa_dass = chi2_contingency(egyptwa_dass_df)
print(egyptwa_dass_df)
print("egypt Whatsapp(or not) DASS severe ChiSq p value=",statwa_dass, pwa_dass,"\n")


#Instagram SAS chisq
IG_male_addicted= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] >=31) & (alex_df["IG"]==True)]
IG_male_not= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] <31) & (alex_df["IG"]==True)]
IG_female_addicted= alex_df[(alex_df["Sex"] ==2) & (alex_df["SAS_total"] >=33)& (alex_df["IG"]==True)]
IG_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] <33)& (alex_df["IG"]==True)]
noig_male_addicted= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] >=31) & (alex_df["IG"]==False)]
noig_male_not= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] <31) & (alex_df["IG"]==False)]
noig_female_addicted= alex_df[(alex_df["Sex"] ==2) & (alex_df["SAS_total"] >=33)& (alex_df["IG"]==False)]
noig_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] <33)& (alex_df["IG"]==False)]

ig_df=[[len(IG_male_addicted)+len(IG_female_addicted),len(IG_male_not)+len(IG_female_not)],
[len(noig_male_addicted)+len(noig_female_addicted),len(noig_male_not)+len(noig_female_not)]]
statig, pig, dofig, expectedig = chi2_contingency(ig_df)
print(ig_df)
print("IG(or not) SAS addiction ChiSq p value=",statig, pig,"\n")

#Instagram TAS chisq
ig_alex=alex_df[(alex_df["IG"]==True) & (alex_df["TAS_total"] >=60)]
ig_alex_neg= alex_df[(alex_df["IG"]==True) & (alex_df["TAS_total"] <60)]
noig_alex= alex_df[(alex_df["IG"] ==False) & (alex_df["TAS_total"] >=60)]
noig_alex_neg= alex_df[(alex_df["IG"]==False) & (alex_df["TAS_total"] <60)]

ig_tas_df=[[len(ig_alex),len(ig_alex_neg)],
[len(noig_alex),len(noig_alex_neg)]]
statig_tas, pig_tas, dofig_tas, expectedig_tas = chi2_contingency(ig_tas_df)
print(ig_tas_df)
print("IG(or not) TAS ChiSq p value=",statig_tas, pig_tas,"\n")

#Instagram DASS chisq
ig_dass= alex_df[(alex_df["IG"]==True) & (alex_df["DASS_score"] >=60)]
ig_dass_neg= alex_df[(alex_df["IG"]==True) & (alex_df["DASS_score"] <60)]
noig_dass= alex_df[(alex_df["IG"] ==False) & (alex_df["DASS_score"] >=60)]
noig_dass_neg= alex_df[(alex_df["IG"]==False) & (alex_df["DASS_score"] <60)]

ig_dass_df=[[len(ig_dass),len(ig_dass_neg)],
[len(noig_dass),len(noig_dass_neg)]]
statig_dass, pig_dass, dofig_dass, expectedig_dass = chi2_contingency(ig_dass_df)
print(ig_dass_df)
print("IG (or not) DASS severe ChiSq p value=",statig_dass, pig_dass,"\n")

#ig Oman SAS chisq
omanig_male_addicted= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==1) & (alex_df["SAS_total"] >=31) & (alex_df["IG"]==True)]
omanig_male_not= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==1) &(alex_df["SAS_total"] <31) & (alex_df["IG"]==True)]
omanig_female_addicted= alex_df[(alex_df["Sex"] ==2) &(alex_df["nation_code"]==1) & (alex_df["SAS_total"] >=33)& (alex_df["IG"]==True)]
omanig_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["nation_code"]==1) & (alex_df["SAS_total"] <33)& (alex_df["IG"]==True)]
omanNotig_male_addicted= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==1) & (alex_df["SAS_total"] >=31) & (alex_df["IG"]==False)]
omanNotig_male_not= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==1) & (alex_df["SAS_total"] <31) & (alex_df["IG"]==False)]
omanNotig_female_addicted= alex_df[(alex_df["Sex"] ==2)& (alex_df["nation_code"]==1) & (alex_df["SAS_total"] >=33)& (alex_df["IG"]==False)]
omanNotig_female_not= alex_df[(alex_df["Sex"]==2)& (alex_df["nation_code"]==1) & (alex_df["SAS_total"] <33)& (alex_df["IG"]==False)]

omanig_sas_df=[[len(omanig_male_addicted)+len(omanig_female_addicted),len(omanig_male_not)+len(omanig_female_not)],
[len(omanNotig_male_addicted)+len(omanNotig_female_addicted),len(omanNotig_male_not)+len(omanNotig_female_not)]]
statw, pw, dofw, expectedw = chi2_contingency(omanig_sas_df)
print(omanig_sas_df)
print("oman ig(or not) SAS addiction ChiSq p value=",statw, pw,"\n")

#oman ig TAS chisq
omanig_alex=alex_df[(alex_df["IG"]==True) &(alex_df["nation_code"]==1)& (alex_df["TAS_total"] >=60)]
omanig_alex_neg= alex_df[(alex_df["IG"]==True) &(alex_df["nation_code"]==1)& (alex_df["TAS_total"] <60)]
omannoig_alex= alex_df[(alex_df["IG"] ==False) &(alex_df["nation_code"]==1)& (alex_df["TAS_total"] >=60)]
omannoig_alex_neg= alex_df[(alex_df["IG"]==False) &(alex_df["nation_code"]==1)& (alex_df["TAS_total"] <60)]

omanig_tas_df=[[len(omanig_alex),len(omanig_alex_neg)],
[len(omannoig_alex),len(omannoig_alex_neg)]]
statwa_tas, pwa_tas, dofwa_tas, expectedwa_tas = chi2_contingency(omanig_tas_df)
print(omanig_tas_df)
print("oman ig TAS ChiSq p value=",statwa_tas, pwa_tas,"\n")

#oman ig DASS chisq
omanig_dass=alex_df[(alex_df["IG"]==True) &(alex_df["nation_code"]==1)& (alex_df["DASS_score"] >=60)]
omanig_dass_neg= alex_df[(alex_df["IG"]==True) &(alex_df["nation_code"]==1)& (alex_df["DASS_score"] <60)]
omannoig_dass= alex_df[(alex_df["IG"] ==False) &(alex_df["nation_code"]==1)& (alex_df["DASS_score"] >=60)]
omannoig_dass_neg= alex_df[(alex_df["IG"]==False)&(alex_df["nation_code"]==1) & (alex_df["DASS_score"] <60)]

omanig_dass_df=[[len(omanig_dass),len(omanig_dass_neg)],
[len(omannoig_dass),len(omannoig_dass_neg)]]
statwa_dass, pwa_dass, dofwa_dass, expectedwa_dass = chi2_contingency(omanig_dass_df)
print(omanig_dass_df)
print("oman ig DASS severe ChiSq p value=",statwa_dass, pwa_dass,"\n")

#ig egypt SAS chisq
egyptig_male_addicted= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==2) & (alex_df["SAS_total"] >=31) & (alex_df["IG"]==True)]
egyptig_male_not= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==2) &(alex_df["SAS_total"] <31) & (alex_df["IG"]==True)]
egyptig_female_addicted= alex_df[(alex_df["Sex"] ==2) &(alex_df["nation_code"]==2) & (alex_df["SAS_total"] >=33)& (alex_df["IG"]==True)]
egyptig_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["nation_code"]==2) & (alex_df["SAS_total"] <33)& (alex_df["IG"]==True)]
egyptNotig_male_addicted= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==2) & (alex_df["SAS_total"] >=31) & (alex_df["IG"]==False)]
egyptNotig_male_not= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==2) & (alex_df["SAS_total"] <31) & (alex_df["IG"]==False)]
egyptNotig_female_addicted= alex_df[(alex_df["Sex"] ==2)& (alex_df["nation_code"]==2) & (alex_df["SAS_total"] >=33)& (alex_df["IG"]==False)]
egyptNotig_female_not= alex_df[(alex_df["Sex"]==2)& (alex_df["nation_code"]==2) & (alex_df["SAS_total"] <33)& (alex_df["IG"]==False)]

egyptig_sas_df=[[len(egyptig_male_addicted)+len(egyptig_female_addicted),len(egyptig_male_not)+len(egyptig_female_not)],
[len(egyptNotig_male_addicted)+len(egyptNotig_female_addicted),len(egyptNotig_male_not)+len(egyptNotig_female_not)]]
statw, pw, dofw, expectedw = chi2_contingency(egyptig_sas_df)
print(egyptig_sas_df)
print("egypt ig SAS addiction ChiSq p value=",statw, pw,"\n")

#egypt Whatsapp TAS chisq
egyptig_alex=alex_df[(alex_df["IG"]==True) &(alex_df["nation_code"]==2)& (alex_df["TAS_total"] >=60)]
egyptig_alex_neg= alex_df[(alex_df["IG"]==True) &(alex_df["nation_code"]==2)& (alex_df["TAS_total"] <60)]
egyptnoig_alex= alex_df[(alex_df["IG"] ==False) &(alex_df["nation_code"]==2)& (alex_df["TAS_total"] >=60)]
egyptnoig_alex_neg= alex_df[(alex_df["IG"]==False) &(alex_df["nation_code"]==2)& (alex_df["TAS_total"] <60)]

egyptig_tas_df=[[len(egyptig_alex),len(egyptig_alex_neg)],
[len(egyptnoig_alex),len(egyptnoig_alex_neg)]]
statwa_tas, pwa_tas, dofwa_tas, expectedwa_tas = chi2_contingency(egyptig_tas_df)
print(egyptig_tas_df)
print("egypt ig TAS ChiSq p value=",statwa_tas, pwa_tas,"\n")

#egypt ig DASS chisq
egyptig_dass=alex_df[(alex_df["IG"]==True) &(alex_df["nation_code"]==2)& (alex_df["DASS_score"] >=60)]
egyptig_dass_neg= alex_df[(alex_df["IG"]==True) &(alex_df["nation_code"]==2)& (alex_df["DASS_score"] <60)]
egyptnoig_dass= alex_df[(alex_df["IG"] ==False) &(alex_df["nation_code"]==2)& (alex_df["DASS_score"] >=60)]
egyptnoig_dass_neg= alex_df[(alex_df["IG"]==False)&(alex_df["nation_code"]==2) & (alex_df["DASS_score"] <60)]

egyptig_dass_df=[[len(egyptig_dass),len(egyptig_dass_neg)],
[len(egyptnoig_dass),len(egyptnoig_dass_neg)]]
statwa_dass, pwa_dass, dofwa_dass, expectedwa_dass = chi2_contingency(egyptig_dass_df)
print(egyptig_dass_df)
print("egypt ig DASS severe ChiSq p value=",statwa_dass, pwa_dass,"\n")

#ig pakistan SAS chisq
pakistan_ig_male_addicted= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==3) & (alex_df["SAS_total"] >=31) & (alex_df["IG"]==True)]
pakistan_ig_male_not= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==3) &(alex_df["SAS_total"] <31) & (alex_df["IG"]==True)]
pakistan_ig_female_addicted= alex_df[(alex_df["Sex"] ==2) &(alex_df["nation_code"]==3) & (alex_df["SAS_total"] >=33)& (alex_df["IG"]==True)]
pakistan_ig_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["nation_code"]==3) & (alex_df["SAS_total"] <33)& (alex_df["IG"]==True)]
pakistan_Notig_male_addicted= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==3) & (alex_df["SAS_total"] >=31) & (alex_df["IG"]==False)]
pakistan_Notig_male_not= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==3) & (alex_df["SAS_total"] <31) & (alex_df["IG"]==False)]
pakistan_Notig_female_addicted= alex_df[(alex_df["Sex"] ==2)& (alex_df["nation_code"]==3) & (alex_df["SAS_total"] >=33)& (alex_df["IG"]==False)]
pakistan_Notig_female_not= alex_df[(alex_df["Sex"]==2)& (alex_df["nation_code"]==3) & (alex_df["SAS_total"] <33)& (alex_df["IG"]==False)]

pakistan_ig_sas_df=[[len(pakistan_ig_male_addicted)+len(pakistan_ig_female_addicted),len(pakistan_ig_male_not)+len(pakistan_ig_female_not)],
[len(pakistan_Notig_male_addicted)+len(pakistan_Notig_female_addicted),len(pakistan_Notig_male_not)+len(pakistan_Notig_female_not)]]
statw, pw, dofw, expectedw = chi2_contingency(pakistan_ig_sas_df)
print(pakistan_ig_sas_df)
print("pakistan_ ig SAS addiction ChiSq p value=",statw, pw,"\n")

#egypt Whatsapp TAS chisq
pakistan_ig_alex=alex_df[(alex_df["IG"]==True) &(alex_df["nation_code"]==3)& (alex_df["TAS_total"] >=60)]
pakistan_ig_alex_neg= alex_df[(alex_df["IG"]==True) &(alex_df["nation_code"]==3)& (alex_df["TAS_total"] <60)]
pakistan_noig_alex= alex_df[(alex_df["IG"] ==False) &(alex_df["nation_code"]==3)& (alex_df["TAS_total"] >=60)]
pakistan_noig_alex_neg= alex_df[(alex_df["IG"]==False) &(alex_df["nation_code"]==3)& (alex_df["TAS_total"] <60)]

pakistan_ig_tas_df=[[len(pakistan_ig_alex),len(pakistan_ig_alex_neg)],
[len(pakistan_noig_alex),len(pakistan_noig_alex_neg)]]
statwa_tas, pwa_tas, dofwa_tas, expectedwa_tas = chi2_contingency(pakistan_ig_tas_df)
print(pakistan_ig_tas_df)
print("pakistan_ ig TAS ChiSq p value=",statwa_tas, pwa_tas,"\n")

#egypt ig DASS chisq
pakistan_ig_dass=alex_df[(alex_df["IG"]==True) &(alex_df["nation_code"]==3)& (alex_df["DASS_score"] >=60)]
pakistan_ig_dass_neg= alex_df[(alex_df["IG"]==True) &(alex_df["nation_code"]==3)& (alex_df["DASS_score"] <60)]
pakistan_noig_dass= alex_df[(alex_df["IG"] ==False) &(alex_df["nation_code"]==3)& (alex_df["DASS_score"] >=60)]
pakistan_noig_dass_neg= alex_df[(alex_df["IG"]==False)&(alex_df["nation_code"]==3) & (alex_df["DASS_score"] <60)]

pakistan_ig_dass_df=[[len(pakistan_ig_dass),len(pakistan_ig_dass_neg)],
[len(pakistan_noig_dass),len(pakistan_noig_dass_neg)]]
statwa_dass, pwa_dass, dofwa_dass, expectedwa_dass = chi2_contingency(pakistan_ig_dass_df)
print(pakistan_ig_dass_df)
print("pakistan_ ig DASS severe ChiSq p value=",statwa_dass, pwa_dass,"\n")

#FacebookSAS chisq
fb_male_addicted= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] >=31) & (alex_df["FB"]==True)]
fb_male_not= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] <31) & (alex_df["FB"]==True)]
fb_female_addicted= alex_df[(alex_df["Sex"] ==2) & (alex_df["SAS_total"] >=33)& (alex_df["FB"]==True)]
fb_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] <33)& (alex_df["FB"]==True)]
nofb_male_addicted= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] >=31) & (alex_df["FB"]==False)]
nofb_male_not= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] <31) & (alex_df["FB"]==False)]
nofb_female_addicted= alex_df[(alex_df["Sex"] ==2) & (alex_df["SAS_total"] >=33)& (alex_df["FB"]==False)]
nofb_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] <33)& (alex_df["FB"]==False)]

fb_df=[[len(fb_male_addicted)+len(fb_female_addicted),len(fb_male_not)+len(fb_female_not)],
[len(nofb_male_addicted)+len(nofb_female_addicted),len(nofb_male_not)+len(nofb_female_not)]]
statfb, pfb, doffb, expectedfb = chi2_contingency(fb_df)
print(fb_df)
print("Facebook (or not) SAS addiction ChiSq p value=",statfb, pfb,"\n")

#FacebookTAS chisq
fb_alex=alex_df[(alex_df["FB"]==True) & (alex_df["TAS_total"] >=60)]
fb_alex_neg= alex_df[(alex_df["FB"]==True) & (alex_df["TAS_total"] <60)]
nofb_alex= alex_df[(alex_df["FB"] ==False) & (alex_df["TAS_total"] >=60)]
nofb_alex_neg= alex_df[(alex_df["FB"]==False) & (alex_df["TAS_total"] <60)]

fb_tas_df=[[len(fb_alex),len(fb_alex_neg)],
[len(nofb_alex),len(nofb_alex_neg)]]
statfb_tas, pfb_tas, doffb_tas, expectedfb_tas = chi2_contingency(fb_tas_df)
print(fb_tas_df)
print("Facebook(or not) TAS ChiSq p value=",statfb_tas, pfb_tas,"\n")

#Facebook DASS chisq
fb_dass= alex_df[(alex_df["FB"]==True) & (alex_df["DASS_score"] >=60)]
fb_dass_neg= alex_df[(alex_df["FB"]==True) & (alex_df["DASS_score"] <60)]
nofb_dass= alex_df[(alex_df["FB"] ==False) & (alex_df["DASS_score"] >=60)]
nofb_dass_neg= alex_df[(alex_df["FB"]==False) & (alex_df["DASS_score"] <60)]

fb_dass_df=[[len(fb_dass),len(fb_dass_neg)],
[len(nofb_dass),len(nofb_dass_neg)]]
statfb_dass, pfb_dass, doffb_dass, expectedfb_dass = chi2_contingency(fb_dass_df)
print(fb_dass_df)
print("Facebook (or not) DASS severe ChiSq p value=",statfb_dass, pfb_dass,"\n")

#fb Oman SAS chisq
omanfb_male_addicted= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==1) & (alex_df["SAS_total"] >=31) & (alex_df["FB"]==True)]
omanfb_male_not= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==1) &(alex_df["SAS_total"] <31) & (alex_df["FB"]==True)]
omanfb_female_addicted= alex_df[(alex_df["Sex"] ==2) &(alex_df["nation_code"]==1) & (alex_df["SAS_total"] >=33)& (alex_df["FB"]==True)]
omanfb_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["nation_code"]==1) & (alex_df["SAS_total"] <33)& (alex_df["FB"]==True)]
omanNotfb_male_addicted= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==1) & (alex_df["SAS_total"] >=31) & (alex_df["FB"]==False)]
omanNotfb_male_not= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==1) & (alex_df["SAS_total"] <31) & (alex_df["FB"]==False)]
omanNotfb_female_addicted= alex_df[(alex_df["Sex"] ==2)& (alex_df["nation_code"]==1) & (alex_df["SAS_total"] >=33)& (alex_df["FB"]==False)]
omanNotfb_female_not= alex_df[(alex_df["Sex"]==2)& (alex_df["nation_code"]==1) & (alex_df["SAS_total"] <33)& (alex_df["FB"]==False)]

omanfb_sas_df=[[len(omanfb_male_addicted)+len(omanfb_female_addicted),len(omanfb_male_not)+len(omanfb_female_not)],
[len(omanNotfb_male_addicted)+len(omanNotfb_female_addicted),len(omanNotfb_male_not)+len(omanNotfb_female_not)]]
statw, pw, dofw, expectedw = chi2_contingency(omanfb_sas_df)
print(omanfb_sas_df)
print("oman fb) SAS addiction ChiSq p value=",statw, pw,"\n")

#oman fb TAS chisq
omanfb_alex=alex_df[(alex_df["FB"]==True) &(alex_df["nation_code"]==1)& (alex_df["TAS_total"] >=60)]
omanfb_alex_neg= alex_df[(alex_df["FB"]==True) &(alex_df["nation_code"]==1)& (alex_df["TAS_total"] <60)]
omannofb_alex= alex_df[(alex_df["FB"] ==False) &(alex_df["nation_code"]==1)& (alex_df["TAS_total"] >=60)]
omannofb_alex_neg= alex_df[(alex_df["FB"]==False) &(alex_df["nation_code"]==1)& (alex_df["TAS_total"] <60)]

omanfb_tas_df=[[len(omanfb_alex),len(omanfb_alex_neg)],
[len(omannofb_alex),len(omannofb_alex_neg)]]
statwa_tas, pwa_tas, dofwa_tas, expectedwa_tas = chi2_contingency(omanfb_tas_df)
print(omanfb_tas_df)
print("oman fb TAS ChiSq p value=",statwa_tas, pwa_tas,"\n")

#oman fb DASS chisq
omanfb_dass=alex_df[(alex_df["FB"]==True) &(alex_df["nation_code"]==1)& (alex_df["DASS_score"] >=60)]
omanfb_dass_neg= alex_df[(alex_df["FB"]==True) &(alex_df["nation_code"]==1)& (alex_df["DASS_score"] <60)]
omannofb_dass= alex_df[(alex_df["FB"] ==False) &(alex_df["nation_code"]==1)& (alex_df["DASS_score"] >=60)]
omannofb_dass_neg= alex_df[(alex_df["FB"]==False)&(alex_df["nation_code"]==1) & (alex_df["DASS_score"] <60)]

omanfb_dass_df=[[len(omanfb_dass),len(omanfb_dass_neg)],
[len(omannofb_dass),len(omannofb_dass_neg)]]
statwa_dass, pwa_dass, dofwa_dass, expectedwa_dass = chi2_contingency(omanfb_dass_df)
print(omanfb_dass_df)
print("oman fb DASS severe ChiSq p value=",statwa_dass, pwa_dass,"\n")

#egypt fb SAS chisq
egyptfb_male_addicted= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==2) & (alex_df["SAS_total"] >=31) & (alex_df["FB"]==True)]
egyptfb_male_not= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==2) &(alex_df["SAS_total"] <31) & (alex_df["FB"]==True)]
egyptfb_female_addicted= alex_df[(alex_df["Sex"] ==2) &(alex_df["nation_code"]==2) & (alex_df["SAS_total"] >=33)& (alex_df["FB"]==True)]
egyptfb_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["nation_code"]==2) & (alex_df["SAS_total"] <33)& (alex_df["FB"]==True)]
egyptNotfb_male_addicted= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==2) & (alex_df["SAS_total"] >=31) & (alex_df["FB"]==False)]
egyptNotfb_male_not= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==2) & (alex_df["SAS_total"] <31) & (alex_df["FB"]==False)]
egyptNotfb_female_addicted= alex_df[(alex_df["Sex"] ==2)& (alex_df["nation_code"]==2) & (alex_df["SAS_total"] >=33)& (alex_df["FB"]==False)]
egyptNotfb_female_not= alex_df[(alex_df["Sex"]==2)& (alex_df["nation_code"]==2) & (alex_df["SAS_total"] <33)& (alex_df["FB"]==False)]

egyptfb_sas_df=[[len(egyptfb_male_addicted)+len(egyptfb_female_addicted),len(egyptfb_male_not)+len(egyptfb_female_not)],
[len(egyptNotfb_male_addicted)+len(egyptNotfb_female_addicted),len(egyptNotfb_male_not)+len(egyptNotfb_female_not)]]
statw, pw, dofw, expectedw = chi2_contingency(egyptfb_sas_df)
print(egyptfb_sas_df)
print("egypt fb) SAS addiction ChiSq p value=",statw, pw,"\n")

#egypt fb TAS chisq
egyptfb_alex=alex_df[(alex_df["FB"]==True) &(alex_df["nation_code"]==2)& (alex_df["TAS_total"] >=60)]
egyptfb_alex_neg= alex_df[(alex_df["FB"]==True) &(alex_df["nation_code"]==2)& (alex_df["TAS_total"] <60)]
egyptnofb_alex= alex_df[(alex_df["FB"] ==False) &(alex_df["nation_code"]==2)& (alex_df["TAS_total"] >=60)]
egyptnofb_alex_neg= alex_df[(alex_df["FB"]==False) &(alex_df["nation_code"]==2)& (alex_df["TAS_total"] <60)]

egyptfb_tas_df=[[len(egyptfb_alex),len(egyptfb_alex_neg)],
[len(egyptnofb_alex),len(egyptnofb_alex_neg)]]
statwa_tas, pwa_tas, dofwa_tas, expectedwa_tas = chi2_contingency(egyptfb_tas_df)
print(egyptfb_tas_df)
print("egyptfb TAS ChiSq p value=",statwa_tas, pwa_tas,"\n")

#egyptfb DASS chisq
egyptfb_dass=alex_df[(alex_df["FB"]==True) &(alex_df["nation_code"]==2)& (alex_df["DASS_score"] >=60)]
egyptfb_dass_neg= alex_df[(alex_df["FB"]==True) &(alex_df["nation_code"]==2)& (alex_df["DASS_score"] <60)]
egyptnofb_dass= alex_df[(alex_df["FB"] ==False) &(alex_df["nation_code"]==2)& (alex_df["DASS_score"] >=60)]
egyptnofb_dass_neg= alex_df[(alex_df["FB"]==False)&(alex_df["nation_code"]==2) & (alex_df["DASS_score"] <60)]

egyptfb_dass_df=[[len(egyptfb_dass),len(egyptfb_dass_neg)],
[len(egyptnofb_dass),len(egyptnofb_dass_neg)]]
statwa_dass, pwa_dass, dofwa_dass, expectedwa_dass = chi2_contingency(egyptfb_dass_df)
print(egyptfb_dass_df)
print("egypt fb DASS severe ChiSq p value=",statwa_dass, pwa_dass,"\n")


#twitter SAS chisq
tw_male_addicted= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] >=31) & (alex_df["tw"]==True)]
tw_male_not= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] <31) & (alex_df["tw"]==True)]
tw_female_addicted= alex_df[(alex_df["Sex"] ==2) & (alex_df["SAS_total"] >=33)& (alex_df["tw"]==True)]
tw_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] <33)& (alex_df["tw"]==True)]
notw_male_addicted= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] >=31) & (alex_df["tw"]==False)]
notw_male_not= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] <31) & (alex_df["tw"]==False)]
notw_female_addicted= alex_df[(alex_df["Sex"] ==2) & (alex_df["SAS_total"] >=33)& (alex_df["tw"]==False)]
notw_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] <33)& (alex_df["tw"]==False)]

tw_df=[[len(tw_male_addicted)+len(tw_female_addicted),len(tw_male_not)+len(tw_female_not)],
[len(notw_male_addicted)+len(notw_female_addicted),len(notw_male_not)+len(notw_female_not)]]
stattw, ptw, doftw, expectedtw = chi2_contingency(tw_df)
print(tw_df)
print("Twitter (or not) SAS addiction ChiSq p value=",stattw, ptw,"\n")

#twitterTAS chisq
tw_alex=alex_df[(alex_df["tw"]==True) & (alex_df["TAS_total"] >=60)]
tw_alex_neg= alex_df[(alex_df["tw"]==True) & (alex_df["TAS_total"] <60)]
notw_alex= alex_df[(alex_df["tw"] ==False) & (alex_df["TAS_total"] >=60)]
notw_alex_neg= alex_df[(alex_df["tw"]==False) & (alex_df["TAS_total"] <60)]

tw_tas_df=[[len(tw_alex),len(tw_alex_neg)],
[len(notw_alex),len(notw_alex_neg)]]
stattw_tas, ptw_tas, doftw_tas, expectedtw_tas = chi2_contingency(tw_tas_df)
print(tw_tas_df)
print("twitter(or not) TAS ChiSq p value=",stattw_tas, ptw_tas,"\n")

#twitter DASS chisq
tw_dass= alex_df[(alex_df["tw"]==True) & (alex_df["DASS_score"] >=60)]
tw_dass_neg= alex_df[(alex_df["tw"]==True) & (alex_df["DASS_score"] <60)]
notw_dass= alex_df[(alex_df["tw"] ==False) & (alex_df["DASS_score"] >=60)]
notw_dass_neg= alex_df[(alex_df["tw"]==False) & (alex_df["DASS_score"] <60)]

tw_dass_df=[[len(tw_dass),len(tw_dass_neg)],
[len(notw_dass),len(notw_dass_neg)]]
stattw_dass, ptw_dass, doftw_dass, expectedtw_dass = chi2_contingency(tw_dass_df)
print(tw_dass_df)
print("twitter (or not) DASS severe ChiSq p value=",stattw_dass, ptw_dass,"\n")

#oman twitter SAS chisq
omantw_male_addicted= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==1) & (alex_df["SAS_total"] >=31) & (alex_df["tw"]==True)]
omantw_male_not= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==1) &(alex_df["SAS_total"] <31) & (alex_df["tw"]==True)]
omantw_female_addicted= alex_df[(alex_df["Sex"] ==2) &(alex_df["nation_code"]==1) & (alex_df["SAS_total"] >=33)& (alex_df["tw"]==True)]
omantw_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["nation_code"]==1) & (alex_df["SAS_total"] <33)& (alex_df["tw"]==True)]
omanNottw_male_addicted= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==1) & (alex_df["SAS_total"] >=31) & (alex_df["tw"]==False)]
omanNottw_male_not= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==1) & (alex_df["SAS_total"] <31) & (alex_df["tw"]==False)]
omanNottw_female_addicted= alex_df[(alex_df["Sex"] ==2)& (alex_df["nation_code"]==1) & (alex_df["SAS_total"] >=33)& (alex_df["tw"]==False)]
omanNottw_female_not= alex_df[(alex_df["Sex"]==2)& (alex_df["nation_code"]==1) & (alex_df["SAS_total"] <33)& (alex_df["tw"]==False)]

omantw_sas_df=[[len(omantw_male_addicted)+len(omantw_female_addicted),len(omantw_male_not)+len(omantw_female_not)],
[len(omanNottw_male_addicted)+len(omanNottw_female_addicted),len(omanNottw_male_not)+len(omanNottw_female_not)]]
statw, pw, dofw, expectedw = chi2_contingency(omantw_sas_df)
print(omantw_sas_df)
print("oman twitter SAS addiction ChiSq p value=",statw, pw,"\n")

#oman twitter TAS chisq
omantw_alex=alex_df[(alex_df["tw"]==True) &(alex_df["nation_code"]==1)& (alex_df["TAS_total"] >=60)]
omantw_alex_neg= alex_df[(alex_df["tw"]==True) &(alex_df["nation_code"]==1)& (alex_df["TAS_total"] <60)]
omannotw_alex= alex_df[(alex_df["tw"] ==False) &(alex_df["nation_code"]==1)& (alex_df["TAS_total"] >=60)]
omannotw_alex_neg= alex_df[(alex_df["tw"]==False) &(alex_df["nation_code"]==1)& (alex_df["TAS_total"] <60)]

omantw_tas_df=[[len(omantw_alex),len(omantw_alex_neg)],
[len(omannotw_alex),len(omannotw_alex_neg)]]
statwa_tas, pwa_tas, dofwa_tas, expectedwa_tas = chi2_contingency(omantw_tas_df)
print(omantw_tas_df)
print("oman twitter TAS ChiSq p value=",statwa_tas, pwa_tas,"\n")

#oman twitter DASS chisq
omantw_dass=alex_df[(alex_df["tw"]==True) &(alex_df["nation_code"]==1)& (alex_df["DASS_score"] >=60)]
omantw_dass_neg= alex_df[(alex_df["tw"]==True) &(alex_df["nation_code"]==1)& (alex_df["DASS_score"] <60)]
omannotw_dass= alex_df[(alex_df["tw"] ==False) &(alex_df["nation_code"]==1)& (alex_df["DASS_score"] >=60)]
omannotw_dass_neg= alex_df[(alex_df["tw"]==False)&(alex_df["nation_code"]==1) & (alex_df["DASS_score"] <60)]

omantw_dass_df=[[len(omantw_dass),len(omantw_dass_neg)],
[len(omannotw_dass),len(omannotw_dass_neg)]]
statwa_dass, pwa_dass, dofwa_dass, expectedwa_dass = chi2_contingency(omantw_dass_df)
print(omantw_dass_df)
print("oman twitter DASS severe ChiSq p value=",statwa_dass, pwa_dass,"\n")

#egypt twitter SAS chisq
egypttw_male_addicted= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==2) & (alex_df["SAS_total"] >=31) & (alex_df["tw"]==True)]
egypttw_male_not= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==2) &(alex_df["SAS_total"] <31) & (alex_df["tw"]==True)]
egypttw_female_addicted= alex_df[(alex_df["Sex"] ==2) &(alex_df["nation_code"]==2) & (alex_df["SAS_total"] >=33)& (alex_df["tw"]==True)]
egypttw_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["nation_code"]==2) & (alex_df["SAS_total"] <33)& (alex_df["tw"]==True)]
egyptNottw_male_addicted= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==2) & (alex_df["SAS_total"] >=31) & (alex_df["tw"]==False)]
egyptNottw_male_not= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==2) & (alex_df["SAS_total"] <31) & (alex_df["tw"]==False)]
egyptNottw_female_addicted= alex_df[(alex_df["Sex"] ==2)& (alex_df["nation_code"]==2) & (alex_df["SAS_total"] >=33)& (alex_df["tw"]==False)]
egyptNottw_female_not= alex_df[(alex_df["Sex"]==2)& (alex_df["nation_code"]==2) & (alex_df["SAS_total"] <33)& (alex_df["tw"]==False)]

egypttw_sas_df=[[len(egypttw_male_addicted)+len(egypttw_female_addicted),len(egypttw_male_not)+len(egypttw_female_not)],
[len(egyptNottw_male_addicted)+len(egyptNottw_female_addicted),len(egyptNottw_male_not)+len(egyptNottw_female_not)]]
statw, pw, dofw, expectedw = chi2_contingency(egypttw_sas_df)
print(egypttw_sas_df)
print("egypt twitter SAS addiction ChiSq p value=",statw, pw,"\n")

#egypt twitter TAS chisq
egypttw_alex=alex_df[(alex_df["tw"]==True) &(alex_df["nation_code"]==2)& (alex_df["TAS_total"] >=60)]
egypttw_alex_neg= alex_df[(alex_df["tw"]==True) &(alex_df["nation_code"]==2)& (alex_df["TAS_total"] <60)]
egyptnotw_alex= alex_df[(alex_df["tw"] ==False) &(alex_df["nation_code"]==2)& (alex_df["TAS_total"] >=60)]
egyptnotw_alex_neg= alex_df[(alex_df["tw"]==False) &(alex_df["nation_code"]==2)& (alex_df["TAS_total"] <60)]

egypttw_tas_df=[[len(egypttw_alex),len(egypttw_alex_neg)],
[len(egyptnotw_alex),len(egyptnotw_alex_neg)]]
statwa_tas, pwa_tas, dofwa_tas, expectedwa_tas = chi2_contingency(egypttw_tas_df)
print(egypttw_tas_df)
print("egypt twitter  TAS ChiSq p value=",statwa_tas, pwa_tas,"\n")

#egypt twitter DASS chisq
egypttw_dass=alex_df[(alex_df["tw"]==True) &(alex_df["nation_code"]==2)& (alex_df["DASS_score"] >=60)]
egypttw_dass_neg= alex_df[(alex_df["tw"]==True) &(alex_df["nation_code"]==2)& (alex_df["DASS_score"] <60)]
egyptnotw_dass= alex_df[(alex_df["tw"] ==False) &(alex_df["nation_code"]==2)& (alex_df["DASS_score"] >=60)]
egyptnotw_dass_neg= alex_df[(alex_df["tw"]==False)&(alex_df["nation_code"]==2) & (alex_df["DASS_score"] <60)]

egypttw_dass_df=[[len(egypttw_dass),len(egypttw_dass_neg)],
[len(egyptnotw_dass),len(egyptnotw_dass_neg)]]
statwa_dass, pwa_dass, dofwa_dass, expectedwa_dass = chi2_contingency(egypttw_dass_df)
print(egypttw_dass_df)
print("egypt twitter DASS severe ChiSq p value=",statwa_dass, pwa_dass,"\n")

#pakistan twitter SAS chisq
pakistantw_male_addicted= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==3) & (alex_df["SAS_total"] >=31) & (alex_df["tw"]==True)]
pakistantw_male_not= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==3) &(alex_df["SAS_total"] <31) & (alex_df["tw"]==True)]
pakistantw_female_addicted= alex_df[(alex_df["Sex"] ==2) &(alex_df["nation_code"]==3) & (alex_df["SAS_total"] >=33)& (alex_df["tw"]==True)]
pakistantw_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["nation_code"]==3) & (alex_df["SAS_total"] <33)& (alex_df["tw"]==True)]
pakistanNottw_male_addicted= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==3) & (alex_df["SAS_total"] >=31) & (alex_df["tw"]==False)]
pakistanNottw_male_not= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==3) & (alex_df["SAS_total"] <31) & (alex_df["tw"]==False)]
pakistanNottw_female_addicted= alex_df[(alex_df["Sex"] ==2)& (alex_df["nation_code"]==3) & (alex_df["SAS_total"] >=33)& (alex_df["tw"]==False)]
pakistanNottw_female_not= alex_df[(alex_df["Sex"]==2)& (alex_df["nation_code"]==3) & (alex_df["SAS_total"] <33)& (alex_df["tw"]==False)]

pakistantw_sas_df=[[len(pakistantw_male_addicted)+len(pakistantw_female_addicted),len(pakistantw_male_not)+len(pakistantw_female_not)],
[len(pakistanNottw_male_addicted)+len(pakistanNottw_female_addicted),len(pakistanNottw_male_not)+len(pakistanNottw_female_not)]]
statw, pw, dofw, expectedw = chi2_contingency(pakistantw_sas_df)
print(pakistantw_sas_df)
print("pakistan twitter SAS addiction ChiSq p value=",statw, pw,"\n")

#pakistan twitter TAS chisq
pakistantw_alex=alex_df[(alex_df["tw"]==True) &(alex_df["nation_code"]==3)& (alex_df["TAS_total"] >=60)]
pakistantw_alex_neg= alex_df[(alex_df["tw"]==True) &(alex_df["nation_code"]==3)& (alex_df["TAS_total"] <60)]
pakistannotw_alex= alex_df[(alex_df["tw"] ==False) &(alex_df["nation_code"]==3)& (alex_df["TAS_total"] >=60)]
pakistannotw_alex_neg= alex_df[(alex_df["tw"]==False) &(alex_df["nation_code"]==3)& (alex_df["TAS_total"] <60)]

pakistantw_tas_df=[[len(pakistantw_alex),len(pakistantw_alex_neg)],
[len(pakistannotw_alex),len(pakistannotw_alex_neg)]]
statwa_tas, pwa_tas, dofwa_tas, expectedwa_tas = chi2_contingency(pakistantw_tas_df)
print(pakistantw_tas_df)
print("pakistan twitter  TAS ChiSq p value=",statwa_tas, pwa_tas,"\n")

#pakistan twitter DASS chisq
pakistantw_dass=alex_df[(alex_df["tw"]==True) &(alex_df["nation_code"]==3)& (alex_df["DASS_score"] >=60)]
pakistantw_dass_neg= alex_df[(alex_df["tw"]==True) &(alex_df["nation_code"]==3)& (alex_df["DASS_score"] <60)]
pakistannotw_dass= alex_df[(alex_df["tw"] ==False) &(alex_df["nation_code"]==3)& (alex_df["DASS_score"] >=60)]
pakistannotw_dass_neg= alex_df[(alex_df["tw"]==False)&(alex_df["nation_code"]==3) & (alex_df["DASS_score"] <60)]

pakistantw_dass_df=[[len(pakistantw_dass),len(pakistantw_dass_neg)],
[len(pakistannotw_dass),len(pakistannotw_dass_neg)]]
statwa_dass, pwa_dass, dofwa_dass, expectedwa_dass = chi2_contingency(pakistantw_dass_df)
print(pakistantw_dass_df)
print("pakistan twitter DASS severe ChiSq p value=",statwa_dass, pwa_dass,"\n")


#snapchat SAS chisq
sc_male_addicted= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] >=31) & (alex_df["SC"]==True)]
sc_male_not= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] <31) & (alex_df["SC"]==True)]
sc_female_addicted= alex_df[(alex_df["Sex"] ==2) & (alex_df["SAS_total"] >=33)& (alex_df["SC"]==True)]
sc_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] <33)& (alex_df["SC"]==True)]
nosc_male_addicted= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] >=31) & (alex_df["SC"]==False)]
nosc_male_not= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] <31) & (alex_df["SC"]==False)]
nosc_female_addicted= alex_df[(alex_df["Sex"] ==2) & (alex_df["SAS_total"] >=33)& (alex_df["SC"]==False)]
nosc_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] <33)& (alex_df["SC"]==False)]

sc_df=[[len(sc_male_addicted)+len(sc_female_addicted),len(sc_male_not)+len(sc_female_not)],
[len(nosc_male_addicted)+len(nosc_female_addicted),len(nosc_male_not)+len(nosc_female_not)]]
statsc, psc, dofsc, expectedsc = chi2_contingency(sc_df)
print(sc_df)
print("snapchat (or not) SAS addiction ChiSq p value=",statsc, psc,"\n")

#snapchat TAS chisq
sc_alex=alex_df[(alex_df["SC"]==True) & (alex_df["TAS_total"] >=60)]
sc_alex_neg= alex_df[(alex_df["SC"]==True) & (alex_df["TAS_total"] <60)]
nosc_alex= alex_df[(alex_df["SC"] ==False) & (alex_df["TAS_total"] >=60)]
nosc_alex_neg= alex_df[(alex_df["SC"]==False) & (alex_df["TAS_total"] <60)]

sc_tas_df=[[len(sc_alex),len(sc_alex_neg)],
[len(nosc_alex),len(nosc_alex_neg)]]
statsc_tas, psc_tas, dofsc_tas, expectedsc_tas = chi2_contingency(sc_tas_df)
print(sc_tas_df)
print("snapchat(or not) TAS ChiSq p value=",statsc_tas, psc_tas,"\n")

#snapchat DASS chisq
sc_dass= alex_df[(alex_df["SC"]==True) & (alex_df["DASS_score"] >=60)]
sc_dass_neg= alex_df[(alex_df["SC"]==True) & (alex_df["DASS_score"] <60)]
nosc_dass= alex_df[(alex_df["SC"] ==False) & (alex_df["DASS_score"] >=60)]
nosc_dass_neg= alex_df[(alex_df["SC"]==False) & (alex_df["DASS_score"] <60)]

sc_dass_df=[[len(sc_dass),len(sc_dass_neg)],
[len(nosc_dass),len(nosc_dass_neg)]]
statsc_dass, psc_dass, dofsc_dass, expectedsc_dass = chi2_contingency(sc_dass_df)
print(sc_dass_df)
print("snapchat (or not) DASS severe ChiSq p value=",statsc_dass, psc_dass,"\n")

#oman snapchat SAS chisq
omansc_male_addicted= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==1) & (alex_df["SAS_total"] >=31) & (alex_df["SC"]==True)]
omansc_male_not= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==1) &(alex_df["SAS_total"] <31) & (alex_df["SC"]==True)]
omansc_female_addicted= alex_df[(alex_df["Sex"] ==2) &(alex_df["nation_code"]==1) & (alex_df["SAS_total"] >=33)& (alex_df["SC"]==True)]
omansc_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["nation_code"]==1) & (alex_df["SAS_total"] <33)& (alex_df["SC"]==True)]
omanNotsc_male_addicted= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==1) & (alex_df["SAS_total"] >=31) & (alex_df["SC"]==False)]
omanNotsc_male_not= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==1) & (alex_df["SAS_total"] <31) & (alex_df["SC"]==False)]
omanNotsc_female_addicted= alex_df[(alex_df["Sex"] ==2)& (alex_df["nation_code"]==1) & (alex_df["SAS_total"] >=33)& (alex_df["SC"]==False)]
omanNotsc_female_not= alex_df[(alex_df["Sex"]==2)& (alex_df["nation_code"]==1) & (alex_df["SAS_total"] <33)& (alex_df["SC"]==False)]

omansc_sas_df=[[len(omansc_male_addicted)+len(omansc_female_addicted),len(omansc_male_not)+len(omansc_female_not)],
[len(omanNotsc_male_addicted)+len(omanNotsc_female_addicted),len(omanNotsc_male_not)+len(omanNotsc_female_not)]]
statw, pw, dofw, expectedw = chi2_contingency(omansc_sas_df)
print(omansc_sas_df)
print("oman snapchat SAS addiction ChiSq p value=",statw, pw,"\n")

#oman snapchatTAS chisq
omansc_alex=alex_df[(alex_df["SC"]==True) &(alex_df["nation_code"]==1)& (alex_df["TAS_total"] >=60)]
omansc_alex_neg= alex_df[(alex_df["SC"]==True) &(alex_df["nation_code"]==1)& (alex_df["TAS_total"] <60)]
omannosc_alex= alex_df[(alex_df["SC"] ==False) &(alex_df["nation_code"]==1)& (alex_df["TAS_total"] >=60)]
omannosc_alex_neg= alex_df[(alex_df["SC"]==False) &(alex_df["nation_code"]==1)& (alex_df["TAS_total"] <60)]

omansc_tas_df=[[len(omansc_alex),len(omansc_alex_neg)],
[len(omannosc_alex),len(omannosc_alex_neg)]]
statwa_tas, pwa_tas, dofwa_tas, expectedwa_tas = chi2_contingency(omansc_tas_df)
print(omansc_tas_df)
print("oman snapchat TAS ChiSq p value=",statwa_tas, pwa_tas,"\n")

#oman snapchat DASS chisq
omansc_dass=alex_df[(alex_df["SC"]==True) &(alex_df["nation_code"]==1)& (alex_df["DASS_score"] >=60)]
omansc_dass_neg= alex_df[(alex_df["SC"]==True) &(alex_df["nation_code"]==1)& (alex_df["DASS_score"] <60)]
omannosc_dass= alex_df[(alex_df["SC"] ==False) &(alex_df["nation_code"]==1)& (alex_df["DASS_score"] >=60)]
omannosc_dass_neg= alex_df[(alex_df["SC"]==False)&(alex_df["nation_code"]==1) & (alex_df["DASS_score"] <60)]

omansc_dass_df=[[len(omansc_dass),len(omansc_dass_neg)],
[len(omannosc_dass),len(omannosc_dass_neg)]]
statwa_dass, pwa_dass, dofwa_dass, expectedwa_dass = chi2_contingency(omansc_dass_df)
print(omansc_dass_df)
print("oman snapchat DASS severe ChiSq p value=",statwa_dass, pwa_dass,"\n")

#egypt snapchat SAS chisq
egyptsc_male_addicted= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==2) & (alex_df["SAS_total"] >=31) & (alex_df["SC"]==True)]
egyptsc_male_not= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==2) &(alex_df["SAS_total"] <31) & (alex_df["SC"]==True)]
egyptsc_female_addicted= alex_df[(alex_df["Sex"] ==2) &(alex_df["nation_code"]==2) & (alex_df["SAS_total"] >=33)& (alex_df["SC"]==True)]
egyptsc_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["nation_code"]==2) & (alex_df["SAS_total"] <33)& (alex_df["SC"]==True)]
egyptNotsc_male_addicted= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==2) & (alex_df["SAS_total"] >=31) & (alex_df["SC"]==False)]
egyptNotsc_male_not= alex_df[(alex_df["Sex"]==1) & (alex_df["nation_code"]==2) & (alex_df["SAS_total"] <31) & (alex_df["SC"]==False)]
egyptNotsc_female_addicted= alex_df[(alex_df["Sex"] ==2)& (alex_df["nation_code"]==2) & (alex_df["SAS_total"] >=33)& (alex_df["SC"]==False)]
egyptNotsc_female_not= alex_df[(alex_df["Sex"]==2)& (alex_df["nation_code"]==2) & (alex_df["SAS_total"] <33)& (alex_df["SC"]==False)]

egyptsc_sas_df=[[len(egyptsc_male_addicted)+len(egyptsc_female_addicted),len(egyptsc_male_not)+len(egyptsc_female_not)],
[len(egyptNotsc_male_addicted)+len(egyptNotsc_female_addicted),len(egyptNotsc_male_not)+len(egyptNotsc_female_not)]]
statw, pw, dofw, expectedw = chi2_contingency(egyptsc_sas_df)
print(egyptsc_sas_df)
print("egypt snapchat SAS addiction ChiSq p value=",statw, pw,"\n")

#egypt snapchat TAS chisq
egyptsc_alex=alex_df[(alex_df["SC"]==True) &(alex_df["nation_code"]==2)& (alex_df["TAS_total"] >=60)]
egyptsc_alex_neg= alex_df[(alex_df["SC"]==True) &(alex_df["nation_code"]==2)& (alex_df["TAS_total"] <60)]
egyptnosc_alex= alex_df[(alex_df["SC"] ==False) &(alex_df["nation_code"]==2)& (alex_df["TAS_total"] >=60)]
egyptnosc_alex_neg= alex_df[(alex_df["SC"]==False) &(alex_df["nation_code"]==2)& (alex_df["TAS_total"] <60)]

egyptsc_tas_df=[[len(egyptsc_alex),len(egyptsc_alex_neg)],
[len(egyptnosc_alex),len(egyptnosc_alex_neg)]]
statwa_tas, pwa_tas, dofwa_tas, expectedwa_tas = chi2_contingency(egyptsc_tas_df)
print(egyptsc_tas_df)
print("egypt snapchat  TAS ChiSq p value=",statwa_tas, pwa_tas,"\n")

#egypt snapchat DASS chisq
egyptsc_dass=alex_df[(alex_df["SC"]==True) &(alex_df["nation_code"]==2)& (alex_df["DASS_score"] >=60)]
egyptsc_dass_neg= alex_df[(alex_df["SC"]==True) &(alex_df["nation_code"]==2)& (alex_df["DASS_score"] <60)]
egyptnosc_dass= alex_df[(alex_df["SC"] ==False) &(alex_df["nation_code"]==2)& (alex_df["DASS_score"] >=60)]
egyptnosc_dass_neg= alex_df[(alex_df["SC"]==False)&(alex_df["nation_code"]==2) & (alex_df["DASS_score"] <60)]

egyptsc_dass_df=[[len(egyptsc_dass),len(egyptsc_dass_neg)],
[len(egyptnosc_dass),len(egyptnosc_dass_neg)]]
statwa_dass, pwa_dass, dofwa_dass, expectedwa_dass = chi2_contingency(egyptsc_dass_df)
print(egyptsc_dass_df)
print("egypt snapchat DASS severe ChiSq p value=",statwa_dass, pwa_dass,"\n")

#M/F SAS chisq test
male_addicted= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] >=31)]
male_not= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] <31)]
female_addicted= alex_df[(alex_df["Sex"] ==2) & (alex_df["SAS_total"] >=33)]
female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] <33)]

mf_df= [[len(male_addicted),len(male_not)], [len(female_addicted),len(female_not)]]
stat, p, dof, expected = chi2_contingency(mf_df)
print(mf_df)
print("M/F SAS addiction ChiSq p value=",stat, p,"\n")

#M/F TAS chisq test
male_alex=alex_df[alex_df["Sex"]==1 & (alex_df["TAS_total"] >=60)]
male_alex_neg= alex_df[alex_df["Sex"]==1 & (alex_df["TAS_total"] <60)]
female_alex= alex_df[(alex_df["Sex"] ==2) & (alex_df["TAS_total"] >=60)]
female_alex_neg= alex_df[(alex_df["Sex"]==2) & (alex_df["TAS_total"] <60)]

mf_alex_df= [[len(male_alex),len(male_alex_neg)], [len(female_alex),len(female_alex_neg)]]
print(mf_alex_df)
stat2, p2, dof2, expected2 = chi2_contingency(mf_alex_df)
print("M/F TAS total ChiSq p value=", stat2, p2,"\n")

#M/F DASS chisq test
male_DASS=alex_df[alex_df["Sex"]==1 & (alex_df["DASS_score"] >=60)]
male_DASS_neg= alex_df[alex_df["Sex"]==1 & (alex_df["DASS_score"] <60)]
female_DASS= alex_df[(alex_df["Sex"] ==2) & (alex_df["DASS_score"] >=60)]
female_DASS_neg= alex_df[(alex_df["Sex"]==2) & (alex_df["DASS_score"] <60)]

mf_DASS_df= [[len(male_DASS),len(male_DASS_neg)], [len(female_DASS),len(female_DASS_neg)]]
print(mf_DASS_df)
stat3, p3, dof3, expected3 = chi2_contingency(mf_DASS_df)
print("M/F DASS total ChiSq p value=", stat3, p3,"\n")

#Oman M/F SAS chisq test
oman_male_addicted= alex_df[(alex_df["Sex"]==1)&(alex_df["nation_code"]==1) & (alex_df["SAS_total"] >=31)]
oman_male_not= alex_df[(alex_df["Sex"]==1)&(alex_df["nation_code"]==1) & (alex_df["SAS_total"] <31)]
oman_female_addicted= alex_df[(alex_df["Sex"] ==2)&(alex_df["nation_code"]==1) & (alex_df["SAS_total"] >=33)]
oman_female_not= alex_df[(alex_df["Sex"]==2)&(alex_df["nation_code"]==1) & (alex_df["SAS_total"] <33)]

oman_mf_df= [[len(oman_male_addicted),len(oman_male_not)], [len(oman_female_addicted),len(oman_female_not)]]
stat, p, dof, expected = chi2_contingency(oman_mf_df)
print(oman_mf_df)
print("oman M/F SAS addiction ChiSq p value=",stat, p,"\n")

#oman M/F TAS chisq test
oman_male_alex=alex_df[(alex_df["Sex"]==1)&(alex_df["nation_code"]==1) & (alex_df["TAS_total"] >=60)]
oman_male_alex_neg= alex_df[(alex_df["Sex"]==1)&(alex_df["nation_code"]==1) & (alex_df["TAS_total"] <60)]
oman_female_alex= alex_df[(alex_df["Sex"] ==2)&(alex_df["nation_code"]==1) & (alex_df["TAS_total"] >=60)]
oman_female_alex_neg= alex_df[(alex_df["Sex"]==2)&(alex_df["nation_code"]==1) & (alex_df["TAS_total"] <60)]

oman_mf_alex_df= [[len(oman_male_alex),len(oman_male_alex_neg)], [len(oman_female_alex),len(oman_female_alex_neg)]]
print(oman_mf_alex_df)
stat2, p2, dof2, expected2 = chi2_contingency(oman_mf_alex_df)
print("Oman M/F TAS total ChiSq p value=", stat2, p2,"\n")

#oman M/F DASS chisq test
oman_male_DASS=alex_df[(alex_df["Sex"]==1)&(alex_df["nation_code"]==1) & (alex_df["DASS_score"] >=60)]
oman_male_DASS_neg= alex_df[(alex_df["Sex"]==1)&(alex_df["nation_code"]==1) & (alex_df["DASS_score"] <60)]
oman_female_DASS= alex_df[(alex_df["Sex"] ==2)&(alex_df["nation_code"]==1) & (alex_df["DASS_score"] >=60)]
oman_female_DASS_neg= alex_df[(alex_df["Sex"]==2)&(alex_df["nation_code"]==1) & (alex_df["DASS_score"] <60)]

oman_mf_DASS_df= [[len(oman_male_DASS),len(oman_male_DASS_neg)], [len(oman_female_DASS),len(oman_female_DASS_neg)]]
print(oman_mf_DASS_df)
stat3, p3, dof3, expected3 = chi2_contingency(oman_mf_DASS_df)
print("Oman M/F DASS total ChiSq p value=", stat3, p3,"\n")

#egypt M/F SAS chisq test
egypt_male_addicted= alex_df[(alex_df["Sex"]==1)&(alex_df["nation_code"]==2) & (alex_df["SAS_total"] >=31)]
egypt_male_not= alex_df[(alex_df["Sex"]==1)&(alex_df["nation_code"]==2) & (alex_df["SAS_total"] <31)]
egypt_female_addicted= alex_df[(alex_df["Sex"] ==2)&(alex_df["nation_code"]==2) & (alex_df["SAS_total"] >=33)]
egypt_female_not= alex_df[(alex_df["Sex"]==2)&(alex_df["nation_code"]==2) & (alex_df["SAS_total"] <33)]

egypt_mf_df= [[len(egypt_male_addicted),len(egypt_male_not)], [len(egypt_female_addicted),len(egypt_female_not)]]
stat, p, dof, expected = chi2_contingency(egypt_mf_df)
print(egypt_mf_df)
print("egypt M/F SAS addiction ChiSq p value=",stat, p,"\n")

#egypt M/F TAS chisq test
egypt_male_alex=alex_df[(alex_df["Sex"]==1)&(alex_df["nation_code"]==2) & (alex_df["TAS_total"] >=60)]
egypt_male_alex_neg= alex_df[(alex_df["Sex"]==1)&(alex_df["nation_code"]==2) & (alex_df["TAS_total"] <60)]
egypt_female_alex= alex_df[(alex_df["Sex"] ==2)&(alex_df["nation_code"]==2) & (alex_df["TAS_total"] >=60)]
egypt_female_alex_neg= alex_df[(alex_df["Sex"]==2)&(alex_df["nation_code"]==2) & (alex_df["TAS_total"] <60)]

egypt_mf_alex_df= [[len(egypt_male_alex),len(egypt_male_alex_neg)], [len(egypt_female_alex),len(egypt_female_alex_neg)]]
print(egypt_mf_alex_df)
stat2, p2, dof2, expected2 = chi2_contingency(egypt_mf_alex_df)
print("egyptM/F TAS total ChiSq p value=", stat2, p2,"\n")

#egypt M/F DASS chisq test
egypt_male_DASS=alex_df[(alex_df["Sex"]==1)&(alex_df["nation_code"]==2) & (alex_df["DASS_score"] >=60)]
egypt_male_DASS_neg= alex_df[(alex_df["Sex"]==1)&(alex_df["nation_code"]==2) & (alex_df["DASS_score"] <60)]
egypt_female_DASS= alex_df[(alex_df["Sex"] ==2)&(alex_df["nation_code"]==2) & (alex_df["DASS_score"] >=60)]
egypt_female_DASS_neg= alex_df[(alex_df["Sex"]==2)&(alex_df["nation_code"]==2) & (alex_df["DASS_score"] <60)]

egypt_mf_DASS_df= [[len(egypt_male_DASS),len(egypt_male_DASS_neg)], [len(egypt_female_DASS),len(egypt_female_DASS_neg)]]
print(egypt_mf_DASS_df)
stat3, p3, dof3, expected3 = chi2_contingency(egypt_mf_DASS_df)
print(" egypt M/F DASS total ChiSq p value=", stat3, p3,"\n")

#pakistan M/F SAS chisq test
pakistan_male_addicted= alex_df[(alex_df["Sex"]==1)&(alex_df["nation_code"]==3) & (alex_df["SAS_total"] >=31)]
pakistan_male_not= alex_df[(alex_df["Sex"]==1)&(alex_df["nation_code"]==3) & (alex_df["SAS_total"] <31)]
pakistan_female_addicted= alex_df[(alex_df["Sex"] ==2)&(alex_df["nation_code"]==3) & (alex_df["SAS_total"] >=33)]
pakistan_female_not= alex_df[(alex_df["Sex"]==2)&(alex_df["nation_code"]==3) & (alex_df["SAS_total"] <33)]

pakistan_mf_df= [[len(pakistan_male_addicted),len(pakistan_male_not)], [len(pakistan_female_addicted),len(pakistan_female_not)]]
stat, p, dof, expected = chi2_contingency(pakistan_mf_df)
print(pakistan_mf_df)
print("pakistan M/F SAS addiction ChiSq p value=",stat, p,"\n")

#pakistan M/F TAS chisq test
pakistan_male_alex=alex_df[(alex_df["Sex"]==1)&(alex_df["nation_code"]==3) & (alex_df["TAS_total"] >=60)]
pakistan_male_alex_neg= alex_df[(alex_df["Sex"]==1)&(alex_df["nation_code"]==3) & (alex_df["TAS_total"] <60)]
pakistan_female_alex= alex_df[(alex_df["Sex"] ==2)&(alex_df["nation_code"]==3) & (alex_df["TAS_total"] >=60)]
pakistan_female_alex_neg= alex_df[(alex_df["Sex"]==2)&(alex_df["nation_code"]==3) & (alex_df["TAS_total"] <60)]

pakistan_mf_alex_df= [[len(pakistan_male_alex),len(pakistan_male_alex_neg)], [len(pakistan_female_alex),len(pakistan_female_alex_neg)]]
print(pakistan_mf_alex_df)
stat2, p2, dof2, expected2 = chi2_contingency(pakistan_mf_alex_df)
print("pakistan M/F TAS total ChiSq p value=", stat2, p2,"\n")

#pakistan M/F DASS chisq test
pakistan_male_DASS=alex_df[(alex_df["Sex"]==1)&(alex_df["nation_code"]==3) & (alex_df["DASS_score"] >=60)]
pakistan_male_DASS_neg= alex_df[(alex_df["Sex"]==1)&(alex_df["nation_code"]==3) & (alex_df["DASS_score"] <60)]
pakistan_female_DASS= alex_df[(alex_df["Sex"] ==2)&(alex_df["nation_code"]==3) & (alex_df["DASS_score"] >=60)]
pakistan_female_DASS_neg= alex_df[(alex_df["Sex"]==2)&(alex_df["nation_code"]==3) & (alex_df["DASS_score"] <60)]

pakistan_mf_DASS_df= [[len(pakistan_male_DASS),len(pakistan_male_DASS_neg)], [len(pakistan_female_DASS),len(pakistan_female_DASS_neg)]]
print(pakistan_mf_DASS_df)
stat3, p3, dof3, expected3 = chi2_contingency(pakistan_mf_DASS_df)
print(" pakistan M/F DASS total ChiSq p value=", stat3, p3,"\n")


#Theo/Prac SAS chisq test
prac_male_addicted= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] >=31) & (alex_df["Faculty"]==2)]
prac_male_not= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] <31) & (alex_df["Faculty"]==2)]
prac_female_addicted= alex_df[(alex_df["Sex"] ==2) & (alex_df["SAS_total"] >=33)& (alex_df["Faculty"]==2)]
prac_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] <33)& (alex_df["Faculty"]==2)]
theo_male_addicted= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] >=31) & (alex_df["Faculty"]==1)]
theo_male_not= alex_df[alex_df["Sex"]==1 & (alex_df["SAS_total"] <31) & (alex_df["Faculty"]==1)]
theo_female_addicted= alex_df[(alex_df["Sex"] ==2) & (alex_df["SAS_total"] >=33) & (alex_df["Faculty"]==1)]
theo_female_not= alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] <33)& (alex_df["Faculty"]==1)]
faculty_df=[[len(prac_male_addicted)+len(prac_female_addicted),len(prac_male_not)+len(prac_female_not)],
[len(theo_male_addicted)+len(theo_female_addicted),len(theo_male_not)+len(theo_female_not)]]
print(faculty_df)
stat3, p3, dof3, expected3 = chi2_contingency(faculty_df)
print("Theo/Prac SAS total ChiSq p value=",stat3, p3,"\n")

#Theo/Prac TAS chisq test
theo_alex=alex_df[(alex_df["Faculty"]==1) & (alex_df["TAS_total"] >60)]
theo_alex_neg= alex_df[(alex_df["Faculty"]==1) & (alex_df["TAS_total"] <=60)]
prac_alex= alex_df[(alex_df["Faculty"] ==2) & (alex_df["TAS_total"] >60)]
prac_alex_neg= alex_df[(alex_df["Faculty"]==2) & (alex_df["TAS_total"] <=60)]
fac_alex_df=[[len(theo_alex),len(theo_alex_neg)],[len(prac_alex),len(prac_alex_neg)]]
print(fac_alex_df)
stat4, p4, dof4, expected4 = chi2_contingency(fac_alex_df)
print("Theo/Prac TAS total ChiSq p value=", stat4,p4,"\n")

#Theo/Prac DASS chisq test
theo_DASS=alex_df[(alex_df["Faculty"]==1) & (alex_df["DASS_score"] >60)]
theo_DASS_neg= alex_df[(alex_df["Faculty"]==1) & (alex_df["DASS_score"] <=60)]
prac_DASS= alex_df[(alex_df["Faculty"] ==2) & (alex_df["DASS_score"] >60)]
prac_DASS_neg= alex_df[(alex_df["Faculty"]==2) & (alex_df["DASS_score"] <=60)]
fac_DASS_df=[[len(theo_DASS),len(theo_DASS_neg)],[len(prac_DASS),len(prac_DASS_neg)]]
print(fac_DASS_df)
stat4, p4, dof4, expected4 = chi2_contingency(fac_DASS_df)
print("Theo/Prac DASS total ChiSq p value=", stat4,p4,"\n")

#Acad year SAS Chisq test
sas_add_y1m=alex_df[(alex_df["Sex"]==1) & (alex_df["SAS_total"] >=31) & (alex_df["Academic year"]==1)]
sas_add_y1f=alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] >=33) & (alex_df["Academic year"]==1)]
sas_add_y2m=alex_df[(alex_df["Sex"]==1) & (alex_df["SAS_total"] >=31) & (alex_df["Academic year"]==2)]
sas_add_y2f=alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] >=33) & (alex_df["Academic year"]==2)]
sas_add_y3m=alex_df[(alex_df["Sex"]==1) & (alex_df["SAS_total"] >=31) & (alex_df["Academic year"]==3)]
sas_add_y3f=alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] >=33) & (alex_df["Academic year"]==3)]
sas_add_y4m=alex_df[(alex_df["Sex"]==1) & (alex_df["SAS_total"] >=31) & (alex_df["Academic year"]==4)]
sas_add_y4f=alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] >=33) & (alex_df["Academic year"]==4)]
sas_add_y5m=alex_df[(alex_df["Sex"]==1) & (alex_df["SAS_total"] >=31) & (alex_df["Academic year"]==5)]
sas_add_y5f=alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] >=33) & (alex_df["Academic year"]==5)]
sas_add_y6m=alex_df[(alex_df["Sex"]==1) & (alex_df["SAS_total"] >=31) & (alex_df["Academic year"]==6)]
sas_add_y6f=alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] >=33) & (alex_df["Academic year"]==6)]
sas_add_y7m=alex_df[(alex_df["Sex"]==1) & (alex_df["SAS_total"] >=31) & (alex_df["Academic year"]==7)]
sas_add_y7f=alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] >=33) & (alex_df["Academic year"]==7)]
sas_not_y1m=alex_df[(alex_df["Sex"]==1) & (alex_df["SAS_total"] <31) & (alex_df["Academic year"]==1)]
sas_not_y1f=alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] <33) & (alex_df["Academic year"]==1)]
sas_not_y2m=alex_df[(alex_df["Sex"]==1) & (alex_df["SAS_total"] <31) & (alex_df["Academic year"]==2)]
sas_not_y2f=alex_df[(alex_df["Sex"]==2)& (alex_df["SAS_total"] <33) & (alex_df["Academic year"]==2)]
sas_not_y3m=alex_df[(alex_df["Sex"]==1) & (alex_df["SAS_total"] <31) & (alex_df["Academic year"]==3)]
sas_not_y3f=alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] <33) & (alex_df["Academic year"]==3)]
sas_not_y4m=alex_df[(alex_df["Sex"]==1) & (alex_df["SAS_total"] <31) & (alex_df["Academic year"]==4)]
sas_not_y4f=alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] <33) & (alex_df["Academic year"]==4)]
sas_not_y5m=alex_df[(alex_df["Sex"]==1) & (alex_df["SAS_total"] <31) & (alex_df["Academic year"]==5)]
sas_not_y5f=alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] <33) & (alex_df["Academic year"]==5)]
sas_not_y6m=alex_df[(alex_df["Sex"]==1) & (alex_df["SAS_total"] <31) & (alex_df["Academic year"]==6)]
sas_not_y6f=alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] <33) & (alex_df["Academic year"]==6)]
sas_not_y7m=alex_df[(alex_df["Sex"]==1) & (alex_df["SAS_total"] <31) & (alex_df["Academic year"]==7)]
sas_not_y7f=alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] <33) & (alex_df["Academic year"]==7)]
sas_add_year_df=[[len(sas_add_y1m)+len(sas_add_y1f),len(sas_add_y2m)+len(sas_add_y2f),len(sas_add_y3m)
+len(sas_add_y3f),len(sas_add_y4m)+len(sas_add_y4f),
len(sas_add_y5m)+len(sas_add_y5f),len(sas_add_y6m)+
len(sas_add_y6f),len(sas_add_y7m)+len(sas_add_y7f)],
[len(sas_not_y1m)+len(sas_not_y1f),len(sas_not_y2m)+
len(sas_not_y2f),len(sas_not_y3m)
+len(sas_not_y3f),len(sas_not_y4m)+len(sas_not_y4f),
len(sas_not_y5m)+len(sas_not_y5f),len(sas_not_y6m)+len(sas_not_y6f),
len(sas_not_y7m)+len(sas_not_y7f)]]
print(sas_add_year_df)
stat5, p5, dof5, expected5 = chi2_contingency(sas_add_year_df)
print("AcadYear SAS total ChiSq p value=", stat5,p5,"\n")

#Acad year TAS Chisq test
alex_y1=alex_df[(alex_df["TAS_total"] >=60)& (alex_df["Academic year"]==1)]
alex_y1_neg=alex_df[(alex_df["TAS_total"] <60)& (alex_df["Academic year"]==1)]
alex_y2=alex_df[(alex_df["TAS_total"] >=60)& (alex_df["Academic year"]==2)]
alex_y2_neg=alex_df[(alex_df["TAS_total"] <60)& (alex_df["Academic year"]==2)]
alex_y3=alex_df[(alex_df["TAS_total"] >=60)& (alex_df["Academic year"]==3)]
alex_y3_neg=alex_df[(alex_df["TAS_total"] <60)& (alex_df["Academic year"]==3)]
alex_y4=alex_df[(alex_df["TAS_total"] >=60)& (alex_df["Academic year"]==4)]
alex_y4_neg=alex_df[(alex_df["TAS_total"] <60)& (alex_df["Academic year"]==4)]
alex_y5=alex_df[(alex_df["TAS_total"] >=60)& (alex_df["Academic year"]==5)]
alex_y5_neg=alex_df[(alex_df["TAS_total"] <60)& (alex_df["Academic year"]==5)]
alex_y6=alex_df[(alex_df["TAS_total"] >=60)& (alex_df["Academic year"]==6)]
alex_y6_neg=alex_df[(alex_df["TAS_total"] <60)& (alex_df["Academic year"]==6)]
alex_y7=alex_df[(alex_df["TAS_total"] >=60)& (alex_df["Academic year"]==7)]
alex_y7_neg=alex_df[(alex_df["TAS_total"] <60)& (alex_df["Academic year"]==7)]
tas_year_df=[[len(alex_y1),len(alex_y2),len(alex_y3),len(alex_y4),
len(alex_y5),len(alex_y6),len(alex_y7)],
[len(alex_y1_neg),len(alex_y2_neg),len(alex_y3_neg),len(alex_y4_neg),
len(alex_y5_neg),len(alex_y6_neg),len(alex_y7_neg)]]
print(tas_year_df)
stat6, p6, dof6, expected6 = chi2_contingency(tas_year_df)
print("AcadYear TAS total ChiSq p value=", stat6,p6,"\n")

#Acad year DASS Chisq test
DASS_y1=alex_df[(alex_df["DASS_score"] >=60)& (alex_df["Academic year"]==1)]
DASS_y1_neg=alex_df[(alex_df["DASS_score"] <60)& (alex_df["Academic year"]==1)]
DASS_y2=alex_df[(alex_df["DASS_score"] >=60)& (alex_df["Academic year"]==2)]
DASS_y2_neg=alex_df[(alex_df["DASS_score"] <60)& (alex_df["Academic year"]==2)]
DASS_y3=alex_df[(alex_df["DASS_score"] >=60)& (alex_df["Academic year"]==3)]
DASS_y3_neg=alex_df[(alex_df["DASS_score"] <60)& (alex_df["Academic year"]==3)]
DASS_y4=alex_df[(alex_df["DASS_score"] >=60)& (alex_df["Academic year"]==4)]
DASS_y4_neg=alex_df[(alex_df["DASS_score"] <60)& (alex_df["Academic year"]==4)]
DASS_y5=alex_df[(alex_df["DASS_score"] >=60)& (alex_df["Academic year"]==5)]
DASS_y5_neg=alex_df[(alex_df["DASS_score"] <60)& (alex_df["Academic year"]==5)]
DASS_y6=alex_df[(alex_df["DASS_score"] >=60)& (alex_df["Academic year"]==6)]
DASS_y6_neg=alex_df[(alex_df["DASS_score"] <60)& (alex_df["Academic year"]==6)]
DASS_y7=alex_df[(alex_df["DASS_score"] >=60)& (alex_df["Academic year"]==7)]
DASS_y7_neg=alex_df[(alex_df["DASS_score"] <60)& (alex_df["Academic year"]==7)]
DASS_year_df=[[len(DASS_y1),len(DASS_y2),len(DASS_y3),len(DASS_y4),
len(DASS_y5),len(DASS_y6),len(DASS_y7)],
[len(DASS_y1_neg),len(DASS_y2_neg),len(DASS_y3_neg),len(DASS_y4_neg),
len(DASS_y5_neg),len(DASS_y6_neg),len(DASS_y7_neg)]]
print(DASS_year_df)
stat7, p7, dof7, expected7 = chi2_contingency(DASS_year_df)
print("AcadYear DASS total ChiSq p value=", stat7,p7,"\n")


#odds ratios of SAS predictors
sex_tab=pd.crosstab(index=alex_df['Sex'], columns=alex_df['smart_addicted'])
oddsratio, pvalue = fisher_exact(sex_tab)
print("Sex odds ratio:", oddsratio, pvalue,"\n")
faculty_tab=pd.crosstab(index=alex_df['Faculty'], columns=alex_df['smart_addicted'])
oddsratio2, pvalue2 = fisher_exact(faculty_tab)
print("Faculty odds ratio:", oddsratio2, pvalue2,"\n")
tas_tab=pd.crosstab(index=alex_df['alex_pos'], columns=alex_df['smart_addicted'])
oddsratio3, pvalue3 = fisher_exact(tas_tab)
print("alexythemia odds ratio:", oddsratio3, pvalue3,"\n")


#overall TAS, SAS & DASS mean sd
print("Overall SAS score mean & sd",alex_df['SAS_total'].mean(),alex_df['SAS_total'].std())
print("Overall TAS score mean & sd",alex_df['TAS_total'].mean(),alex_df['TAS_total'].std(),"\n")
print("Overall DASS score mean & sd",alex_df['DASS_score'].mean(),alex_df['DASS_score'].std(),"\n")

#whole population whatsapp SAS t-test
wa_sas=alex_df.query('WA == True')['SAS_total']
nowa_sas=alex_df.query('WA == False')['SAS_total']
print("whatsapp SAS mean, sd, n:", wa_sas.mean(),wa_sas.std(),wa_sas.count())
print("no whatsapp SAS mean, sd, n:",nowa_sas.mean(),nowa_sas.std(),nowa_sas.count())
wa_sas_res = pg.ttest(wa_sas, nowa_sas, correction=True)
print(wa_sas_res,"\n")

#whole population whatsapp TAS t-test
wa_tas=alex_df.query('WA == True')['TAS_total']
nowa_tas=alex_df.query('WA == False')['TAS_total']
print("whatsapp TAS mean, sd, n:", wa_tas.mean(),wa_tas.std(),wa_tas.count())
print("no whatsapp TAS mean, sd, n:",nowa_tas.mean(),nowa_tas.std(),nowa_tas.count())
wa_tas_res = pg.ttest(wa_tas, nowa_tas, correction=True)
print(wa_tas_res,"\n")

#whole population whatsapp DASS t-test
wa_dass=alex_df.query('WA == True')['DASS_score']
nowa_dass=alex_df.query('WA == False')['DASS_score']
print("whatsapp dass mean, sd, n:", wa_dass.mean(),wa_dass.std(),wa_dass.count())
print("no whatsapp dass mean, sd, n:",nowa_dass.mean(),nowa_dass.std(),nowa_dass.count())
wa_dass_res = pg.ttest(wa_dass, nowa_dass, correction=True)
print(wa_dass_res,"\n")

#Oman whatsapp SAS t-test
oman_wa_sas=alex_df.query('WA == True and nation_code==1')['SAS_total']
oman_nowa_sas=alex_df.query('WA == False and nation_code==1')['SAS_total']
print("oman whatsapp SAS mean, sd, n:", oman_wa_sas.mean(),oman_wa_sas.std(),oman_wa_sas.count())
print("oman no whatsapp SAS mean, sd, n:",oman_nowa_sas.mean(),oman_nowa_sas.std(),oman_nowa_sas.count())
oman_wa_sas_res = pg.ttest(oman_wa_sas, oman_nowa_sas, correction=True)
print(oman_wa_sas_res,"\n")

#Oman whatsapp TAS t-test
oman_wa_tas=alex_df.query('WA == True and nation_code==1')['TAS_total']
oman_nowa_tas=alex_df.query('WA == False and nation_code==1')['TAS_total']
print("oman whatsapp tAS mean, sd, n:", oman_wa_tas.mean(),oman_wa_tas.std(),oman_wa_tas.count())
print("oman no whatsapp tAS mean, sd, n:",oman_nowa_tas.mean(),oman_nowa_tas.std(),oman_nowa_tas.count())
oman_wa_tas_res = pg.ttest(oman_wa_tas, oman_nowa_tas, correction=True)
print(oman_wa_tas_res,"\n")

#Oman whatsapp DASS t-test
oman_wa_dass=alex_df.query('WA == True and nation_code==1')['DASS_score']
oman_nowa_dass=alex_df.query('WA == False and nation_code==1')['DASS_score']
print("oman whatsapp dass mean, sd, n:", oman_wa_dass.mean(),oman_wa_dass.std(),oman_wa_dass.count())
print("oman no whatsapp dass mean, sd, n:",oman_nowa_dass.mean(),oman_nowa_dass.std(),oman_nowa_dass.count())
oman_wa_dass_res = pg.ttest(oman_wa_dass, oman_nowa_dass, correction=True)
print(oman_wa_dass_res,"\n")

#Egypt whatsapp SAS t-test
egypt_wa_sas=alex_df.query('WA == True and nation_code==2')['SAS_total']
egypt_nowa_sas=alex_df.query('WA == False and nation_code==2')['SAS_total']
print("egypt whatsapp SAS mean, sd, n:", egypt_wa_sas.mean(),egypt_wa_sas.std(),egypt_wa_sas.count())
print("egypt no whatsapp SAS mean, sd, n:",egypt_nowa_sas.mean(),egypt_nowa_sas.std(),egypt_nowa_sas.count())
egypt_wa_sas_res = pg.ttest(egypt_wa_sas, egypt_nowa_sas, correction=True)
print(egypt_wa_sas_res,"\n")

#Egypt whatsapp TAS t-test
egypt_wa_tas=alex_df.query('WA == True and nation_code==2')['TAS_total']
egypt_nowa_tas=alex_df.query('WA == False and nation_code==2')['TAS_total']
print("egypt whatsapp tAS mean, sd, n:", egypt_wa_tas.mean(),egypt_wa_tas.std(),egypt_wa_tas.count())
print("egypt no whatsapp tAS mean, sd, n:",egypt_nowa_tas.mean(),egypt_nowa_tas.std(),egypt_nowa_tas.count())
egypt_wa_tas_res = pg.ttest(egypt_wa_tas, egypt_nowa_tas, correction=True)
print(egypt_wa_tas_res,"\n")

#Egypt whatsapp dass t-test
egypt_wa_dass=alex_df.query('WA == True and nation_code==2')['DASS_score']
egypt_nowa_dass=alex_df.query('WA == False and nation_code==2')['DASS_score']
print("egypt whatsapp dass mean, sd, n:", egypt_wa_dass.mean(),egypt_wa_dass.std(),egypt_wa_dass.count())
print("egypt no whatsapp dass mean, sd, n:",egypt_nowa_dass.mean(),egypt_nowa_dass.std(),egypt_nowa_dass.count())
egypt_wa_dass_res = pg.ttest(egypt_wa_dass, egypt_nowa_dass, correction=True)
print(egypt_wa_dass_res,"\n")

#Pakistan whatsapp SAS t-test
print("No respondents from Pakistan reported using Whatsapp \n")


#whole population IG SAS t-test
ig_sas=alex_df.query('IG == True')['SAS_total']
noig_sas=alex_df.query('IG == False')['SAS_total']
print("ig SAS mean, sd, n:", ig_sas.mean(),ig_sas.std(),ig_sas.count())
print("no ig SAS mean, sd, n:",noig_sas.mean(),noig_sas.std(),noig_sas.count())
ig_sas_res = pg.ttest(ig_sas, noig_sas, correction=True)
print(ig_sas_res,"\n")

#whole population IG TAS t-test
ig_tas=alex_df.query('IG == True')['TAS_total']
noig_tas=alex_df.query('IG == False')['TAS_total']
print("ig tAS mean, sd, n:", ig_tas.mean(),ig_tas.std(),ig_tas.count())
print("no ig tAS mean, sd, n:",noig_tas.mean(),noig_tas.std(),noig_tas.count())
ig_tas_res = pg.ttest(ig_tas, noig_tas, correction=True)
print(ig_tas_res,"\n")

#whole population IG dass t-test
ig_dass=alex_df.query('IG == True')['DASS_score']
noig_dass=alex_df.query('IG == False')['DASS_score']
print("ig dass mean, sd, n:", ig_dass.mean(),ig_dass.std(),ig_dass.count())
print("no ig dass mean, sd, n:",noig_dass.mean(),noig_dass.std(),noig_dass.count())
ig_dass_res = pg.ttest(ig_dass, noig_dass, correction=True)
print(ig_dass_res,"\n")

#Oman IG SAS t-test
oman_ig_sas=alex_df.query('IG == True and nation_code==1')['SAS_total']
oman_noig_sas=alex_df.query('IG == False and nation_code==1')['SAS_total']
print("oman ig SAS mean, sd, n:", oman_ig_sas.mean(),oman_ig_sas.std(),oman_ig_sas.count())
print("oman no ig SAS mean, sd, n:",oman_noig_sas.mean(),oman_noig_sas.std(),oman_noig_sas.count())
oman_ig_sas_res = pg.ttest(oman_ig_sas, oman_noig_sas, correction=True)
print(oman_ig_sas_res,"\n")

#Oman IG tAS t-test
oman_ig_tas=alex_df.query('IG == True and nation_code==1')['TAS_total']
oman_noig_tas=alex_df.query('IG == False and nation_code==1')['TAS_total']
print("oman ig tAS mean, sd, n:", oman_ig_tas.mean(),oman_ig_tas.std(),oman_ig_tas.count())
print("oman no ig tAS mean, sd, n:",oman_noig_tas.mean(),oman_noig_tas.std(),oman_noig_tas.count())
oman_ig_tas_res = pg.ttest(oman_ig_tas, oman_noig_tas, correction=True)
print(oman_ig_tas_res,"\n")

#Oman IG dass t-test
oman_ig_dass=alex_df.query('IG == True and nation_code==1')['DASS_score']
oman_noig_dass=alex_df.query('IG == False and nation_code==1')['DASS_score']
print("oman ig dass mean, sd, n:", oman_ig_dass.mean(),oman_ig_dass.std(),oman_ig_dass.count())
print("oman no ig dass mean, sd, n:",oman_noig_dass.mean(),oman_noig_dass.std(),oman_noig_dass.count())
oman_ig_dass_res = pg.ttest(oman_ig_dass, oman_noig_dass, correction=True)
print(oman_ig_dass_res,"\n")


#Egypt IG SAS t-test
egypt_ig_sas=alex_df.query('IG == True and nation_code==2')['SAS_total']
egypt_noig_sas=alex_df.query('IG == False and nation_code==2')['SAS_total']
print("egypt ig SAS mean, sd, n:", egypt_ig_sas.mean(),egypt_ig_sas.std(),egypt_ig_sas.count())
print("egypt no ig SAS mean, sd, n:",egypt_noig_sas.mean(),egypt_noig_sas.std(),egypt_noig_sas.count())
egypt_ig_sas_res = pg.ttest(egypt_ig_sas, egypt_noig_sas, correction=True)
print(egypt_ig_sas_res,"\n")

#Egypt IG tAS t-test
egypt_ig_tas=alex_df.query('IG == True and nation_code==2')['TAS_total']
egypt_noig_tas=alex_df.query('IG == False and nation_code==2')['TAS_total']
print("egypt ig tAS mean, sd, n:", egypt_ig_tas.mean(),egypt_ig_tas.std(),egypt_ig_tas.count())
print("egypt no ig tAS mean, sd, n:",egypt_noig_tas.mean(),egypt_noig_tas.std(),egypt_noig_tas.count())
egypt_ig_tas_res = pg.ttest(egypt_ig_tas, egypt_noig_tas, correction=True)
print(egypt_ig_tas_res,"\n")

#Egypt IG dass t-test
egypt_ig_dass=alex_df.query('IG == True and nation_code==2')['DASS_score']
egypt_noig_dass=alex_df.query('IG == False and nation_code==2')['DASS_score']
print("egypt ig dass mean, sd, n:", egypt_ig_dass.mean(),egypt_ig_dass.std(),egypt_ig_dass.count())
print("egypt no ig dass mean, sd, n:",egypt_noig_dass.mean(),egypt_noig_dass.std(),egypt_noig_dass.count())
egypt_ig_dass_res = pg.ttest(egypt_ig_dass, egypt_noig_dass, correction=True)
print(egypt_ig_dass_res,"\n")

#Pakistan IG SAS t-test
pakistan_ig_sas=alex_df.query('IG == True and nation_code==3')['SAS_total']
pakistan_noig_sas=alex_df.query('IG == False and nation_code==3')['SAS_total']
print("pakistan ig SAS mean, sd, n:", pakistan_ig_sas.mean(),pakistan_ig_sas.std(),pakistan_ig_sas.count())
print("pakistan no ig SAS mean, sd, n:",pakistan_noig_sas.mean(),pakistan_noig_sas.std(),pakistan_noig_sas.count())
pakistan_ig_sas_res = pg.ttest(pakistan_ig_sas, pakistan_noig_sas, correction=True)
print(pakistan_ig_sas_res,"\n")

#Pakistan IG tAS t-test
pakistan_ig_tas=alex_df.query('IG == True and nation_code==3')['TAS_total']
pakistan_noig_tas=alex_df.query('IG == False and nation_code==3')['TAS_total']
print("pakistan ig tAS mean, sd, n:", pakistan_ig_tas.mean(),pakistan_ig_tas.std(),pakistan_ig_tas.count())
print("pakistan no ig tAS mean, sd, n:",pakistan_noig_tas.mean(),pakistan_noig_tas.std(),pakistan_noig_tas.count())
pakistan_ig_tas_res = pg.ttest(pakistan_ig_tas, pakistan_noig_tas, correction=True)
print(pakistan_ig_tas_res,"\n")

#Pakistan IG dass t-test
pakistan_ig_dass=alex_df.query('IG == True and nation_code==3')['DASS_score']
pakistan_noig_dass=alex_df.query('IG == False and nation_code==3')['DASS_score']
print("pakistan ig dass mean, sd, n:", pakistan_ig_dass.mean(),pakistan_ig_dass.std(),pakistan_ig_dass.count())
print("pakistan no ig dass mean, sd, n:",pakistan_noig_dass.mean(),pakistan_noig_dass.std(),pakistan_noig_dass.count())
pakistan_ig_dass_res = pg.ttest(pakistan_ig_dass, pakistan_noig_dass, correction=True)
print(pakistan_ig_dass_res,"\n")

#whole population FB SAS t-test
fb_sas=alex_df.query('FB == True')['SAS_total']
nofb_sas=alex_df.query('FB == False')['SAS_total']
print("fb SAS mean, sd, n:", fb_sas.mean(),fb_sas.std(),fb_sas.count())
print("no fb SAS mean, sd, n:",nofb_sas.mean(),nofb_sas.std(),nofb_sas.count())
fb_sas_res = pg.ttest(fb_sas, nofb_sas, correction=True)
print(fb_sas_res,"\n")

#whole population FB tAS t-test
fb_tas=alex_df.query('FB == True')['TAS_total']
nofb_tas=alex_df.query('FB == False')['TAS_total']
print("fb tAS mean, sd, n:", fb_tas.mean(),fb_tas.std(),fb_tas.count())
print("no fb tAS mean, sd, n:",nofb_tas.mean(),nofb_tas.std(),nofb_tas.count())
fb_tas_res = pg.ttest(fb_tas, nofb_tas, correction=True)
print(fb_tas_res,"\n")

#whole population FB dass t-test
fb_dass=alex_df.query('FB == True')['DASS_score']
nofb_dass=alex_df.query('FB == False')['DASS_score']
print("fb dass mean, sd, n:", fb_dass.mean(),fb_dass.std(),fb_dass.count())
print("no fb dass mean, sd, n:",nofb_dass.mean(),nofb_dass.std(),nofb_dass.count())
fb_dass_res = pg.ttest(fb_dass, nofb_dass, correction=True)
print(fb_dass_res,"\n")

#Oman FB SAS t-test
oman_fb_sas=alex_df.query('FB == True and nation_code==1')['SAS_total']
oman_nofb_sas=alex_df.query('FB == False and nation_code==1')['SAS_total']
print("oman fb SAS mean, sd, n:", oman_fb_sas.mean(),oman_fb_sas.std(),oman_fb_sas.count())
print("oman no fb SAS mean, sd, n:",oman_nofb_sas.mean(),oman_nofb_sas.std(),oman_nofb_sas.count())
oman_fb_sas_res = pg.ttest(oman_fb_sas, oman_nofb_sas, correction=True)
print(oman_fb_sas_res,"\n")

#Oman FB tAS t-test
oman_fb_tas=alex_df.query('FB == True and nation_code==1')['TAS_total']
oman_nofb_tas=alex_df.query('FB == False and nation_code==1')['TAS_total']
print("oman fb tAS mean, sd, n:", oman_fb_tas.mean(),oman_fb_tas.std(),oman_fb_tas.count())
print("oman no fb tAS mean, sd, n:",oman_nofb_tas.mean(),oman_nofb_tas.std(),oman_nofb_tas.count())
oman_fb_tas_res = pg.ttest(oman_fb_tas, oman_nofb_tas, correction=True)
print(oman_fb_tas_res,"\n")

#Oman FB dass t-test
oman_fb_dass=alex_df.query('FB == True and nation_code==1')['DASS_score']
oman_nofb_dass=alex_df.query('FB == False and nation_code==1')['DASS_score']
print("oman fb dass mean, sd, n:", oman_fb_dass.mean(),oman_fb_dass.std(),oman_fb_dass.count())
print("oman no fb dass mean, sd, n:",oman_nofb_dass.mean(),oman_nofb_dass.std(),oman_nofb_dass.count())
oman_fb_dass_res = pg.ttest(oman_fb_dass, oman_nofb_dass, correction=True)
print(oman_fb_dass_res,"\n")

#Egypt FB SAS t-test
egypt_fb_sas=alex_df.query('FB == True and nation_code==2')['SAS_total']
egypt_nofb_sas=alex_df.query('FB == False and nation_code==2')['SAS_total']
print("egypt fb SAS mean, sd, n:", egypt_fb_sas.mean(),egypt_fb_sas.std(),egypt_fb_sas.count())
print("egypt no fb SAS mean, sd, n:",egypt_nofb_sas.mean(),egypt_nofb_sas.std(),egypt_nofb_sas.count())
egypt_fb_sas_res = pg.ttest(egypt_fb_sas, egypt_nofb_sas, correction=True)
print(egypt_fb_sas_res,"\n")

#Egypt FB tas t-test
egypt_fb_tas=alex_df.query('FB == True and nation_code==2')['TAS_total']
egypt_nofb_tas=alex_df.query('FB == False and nation_code==2')['TAS_total']
print("egypt fb tAS mean, sd, n:", egypt_fb_tas.mean(),egypt_fb_tas.std(),egypt_fb_tas.count())
print("egypt no fb tAS mean, sd, n:",egypt_nofb_tas.mean(),egypt_nofb_tas.std(),egypt_nofb_tas.count())
egypt_fb_tas_res = pg.ttest(egypt_fb_tas, egypt_nofb_tas, correction=True)
print(egypt_fb_tas_res,"\n")

#Egypt FB dass t-test
egypt_fb_dass=alex_df.query('FB == True and nation_code==2')['DASS_score']
egypt_nofb_dass=alex_df.query('FB == False and nation_code==2')['DASS_score']
print("egypt fb dass mean, sd, n:", egypt_fb_dass.mean(),egypt_fb_dass.std(),egypt_fb_dass.count())
print("egypt no fb dass mean, sd, n:",egypt_nofb_dass.mean(),egypt_nofb_dass.std(),egypt_nofb_dass.count())
egypt_fb_dass_res = pg.ttest(egypt_fb_dass, egypt_nofb_dass, correction=True)
print(egypt_fb_dass_res,"\n")

#Pakistan FB SAS t-test
print("No respondents from Pakistan reported using Facebook \n")

#whole population Snapchat SAS t-test
sc_sas=alex_df.query('SC == True')['SAS_total']
nosc_sas=alex_df.query('SC == False')['SAS_total']
print("sc SAS mean, sd, n:", sc_sas.mean(),sc_sas.std(),sc_sas.count())
print("no sc SAS mean, sd, n:",nosc_sas.mean(),nosc_sas.std(),nosc_sas.count())
sc_sas_res = pg.ttest(sc_sas, nosc_sas, correction=True)
print(sc_sas_res,"\n")

#whole population Snapchat tAS t-test
sc_tas=alex_df.query('SC == True')['TAS_total']
nosc_tas=alex_df.query('SC == False')['TAS_total']
print("sc tAS mean, sd, n:", sc_tas.mean(),sc_tas.std(),sc_tas.count())
print("no sc tAS mean, sd, n:",nosc_tas.mean(),nosc_tas.std(),nosc_tas.count())
sc_tas_res = pg.ttest(sc_tas, nosc_tas, correction=True)
print(sc_tas_res,"\n")

#whole population Snapchat dass t-test
sc_dass=alex_df.query('SC == True')['DASS_score']
nosc_dass=alex_df.query('SC == False')['DASS_score']
print("sc dass mean, sd, n:", sc_dass.mean(),sc_dass.std(),sc_dass.count())
print("no sc dass mean, sd, n:",nosc_dass.mean(),nosc_dass.std(),nosc_dass.count())
sc_dass_res = pg.ttest(sc_dass, nosc_dass, correction=True)
print(sc_dass_res,"\n")

#Oman snapchat SAS t-test
oman_sc_sas=alex_df.query('SC == True and nation_code==1')['SAS_total']
oman_nosc_sas=alex_df.query('SC == False and nation_code==1')['SAS_total']
print("oman sc SAS mean, sd, n:", oman_sc_sas.mean(),oman_sc_sas.std(),oman_sc_sas.count())
print("oman no sc SAS mean, sd, n:",oman_nosc_sas.mean(),oman_nosc_sas.std(),oman_nosc_sas.count())
oman_sc_sas_res = pg.ttest(oman_sc_sas, oman_nosc_sas, correction=True)
print(oman_sc_sas_res,"\n")

#Oman snapchat taS t-test
oman_sc_tas=alex_df.query('SC == True and nation_code==1')['TAS_total']
oman_nosc_tas=alex_df.query('SC == False and nation_code==1')['TAS_total']
print("oman sc tAS mean, sd, n:", oman_sc_tas.mean(),oman_sc_tas.std(),oman_sc_tas.count())
print("oman no sc tAS mean, sd, n:",oman_nosc_tas.mean(),oman_nosc_tas.std(),oman_nosc_tas.count())
oman_sc_tas_res = pg.ttest(oman_sc_tas, oman_nosc_tas, correction=True)
print(oman_sc_tas_res,"\n")

#Oman snapchat dass t-test
oman_sc_dass=alex_df.query('SC == True and nation_code==1')['DASS_score']
oman_nosc_dass=alex_df.query('SC == False and nation_code==1')['DASS_score']
print("oman sc dass mean, sd, n:", oman_sc_dass.mean(),oman_sc_dass.std(),oman_sc_dass.count())
print("oman no sc dass mean, sd, n:",oman_nosc_dass.mean(),oman_nosc_dass.std(),oman_nosc_dass.count())
oman_sc_dass_res = pg.ttest(oman_sc_dass, oman_nosc_dass, correction=True)
print(oman_sc_dass_res,"\n")

#Egypt snapchat SAS t-test
egypt_sc_sas=alex_df.query('SC == True and nation_code==2')['SAS_total']
egypt_nosc_sas=alex_df.query('SC == False and nation_code==2')['SAS_total']
print("egypt sc SAS mean, sd, n:", egypt_sc_sas.mean(),egypt_sc_sas.std(),egypt_sc_sas.count())
print("egypt no sc SAS mean, sd, n:",egypt_nosc_sas.mean(),egypt_nosc_sas.std(),egypt_nosc_sas.count())
egypt_sc_sas_res = pg.ttest(egypt_sc_sas, egypt_nosc_sas, correction=True)
print(egypt_sc_sas_res,"\n")

#Egypt snapchat tAS t-test
egypt_sc_tas=alex_df.query('SC == True and nation_code==2')['TAS_total']
egypt_nosc_tas=alex_df.query('SC == False and nation_code==2')['TAS_total']
print("egypt sc tAS mean, sd, n:", egypt_sc_tas.mean(),egypt_sc_tas.std(),egypt_sc_tas.count())
print("egypt no sc tAS mean, sd, n:",egypt_nosc_tas.mean(),egypt_nosc_tas.std(),egypt_nosc_tas.count())
egypt_sc_tas_res = pg.ttest(egypt_sc_tas, egypt_nosc_tas, correction=True)
print(egypt_sc_tas_res,"\n")

#Egypt snapchat dass t-test
egypt_sc_dass=alex_df.query('SC == True and nation_code==2')['DASS_score']
egypt_nosc_dass=alex_df.query('SC == False and nation_code==2')['DASS_score']
print("egypt sc dass mean, sd, n:", egypt_sc_dass.mean(),egypt_sc_dass.std(),egypt_sc_dass.count())
print("egypt no sc dass mean, sd, n:",egypt_nosc_dass.mean(),egypt_nosc_dass.std(),egypt_nosc_dass.count())
egypt_sc_dass_res = pg.ttest(egypt_sc_dass, egypt_nosc_dass, correction=True)
print(egypt_sc_dass_res,"\n")


#Pakistan SC SAS t-test
print("No respondents from Pakistan reported using Snapchat \n")

#whole population twitter SAS t-test
tw_sas=alex_df.query('tw == True')['SAS_total']
notw_sas=alex_df.query('tw == False')['SAS_total']
print("tw SAS mean, sd, n:", tw_sas.mean(),tw_sas.std(),tw_sas.count())
print("no tw SAS mean, sd, n:",notw_sas.mean(),notw_sas.std(),notw_sas.count())
tw_sas_res = pg.ttest(tw_sas, notw_sas, correction=True)
print(tw_sas_res,"\n")

#whole population twitter tAS t-test
tw_tas=alex_df.query('tw == True')['TAS_total']
notw_tas=alex_df.query('tw == False')['TAS_total']
print("tw tAS mean, sd, n:", tw_tas.mean(),tw_tas.std(),tw_tas.count())
print("no tw tAS mean, sd, n:",notw_tas.mean(),notw_tas.std(),notw_tas.count())
tw_tas_res = pg.ttest(tw_tas, notw_tas, correction=True)
print(tw_tas_res,"\n")

#whole population twitter dass t-test
tw_dass=alex_df.query('tw == True')['DASS_score']
notw_dass=alex_df.query('tw == False')['DASS_score']
print("tw dass mean, sd, n:", tw_dass.mean(),tw_dass.std(),tw_dass.count())
print("no tw dass mean, sd, n:",notw_dass.mean(),notw_dass.std(),notw_dass.count())
tw_dass_res = pg.ttest(tw_dass, notw_dass, correction=True)
print(tw_dass_res,"\n")

#Oman twitter SAS t-test
oman_tw_sas=alex_df.query('tw == True and nation_code==1')['SAS_total']
oman_notw_sas=alex_df.query('tw == False and nation_code==1')['SAS_total']
print("oman tw SAS mean, sd, n:", oman_tw_sas.mean(),oman_tw_sas.std(),oman_tw_sas.count())
print("oman no tw SAS mean, sd, n:",oman_notw_sas.mean(),oman_notw_sas.std(),oman_notw_sas.count())
oman_tw_sas_res = pg.ttest(oman_tw_sas, oman_notw_sas, correction=True)
print(oman_tw_sas_res,"\n")

#Oman twitter tAS t-test
oman_tw_tas=alex_df.query('tw == True and nation_code==1')['TAS_total']
oman_notw_tas=alex_df.query('tw == False and nation_code==1')['TAS_total']
print("oman tw tAS mean, sd, n:", oman_tw_tas.mean(),oman_tw_tas.std(),oman_tw_tas.count())
print("oman no tw tAS mean, sd, n:",oman_notw_tas.mean(),oman_notw_tas.std(),oman_notw_tas.count())
oman_tw_tas_res = pg.ttest(oman_tw_tas, oman_notw_tas, correction=True)
print(oman_tw_tas_res,"\n")

#Oman twitter dass t-test
oman_tw_dass=alex_df.query('tw == True and nation_code==1')['DASS_score']
oman_notw_dass=alex_df.query('tw == False and nation_code==1')['DASS_score']
print("oman tw dass mean, sd, n:", oman_tw_dass.mean(),oman_tw_dass.std(),oman_tw_dass.count())
print("oman no tw dass mean, sd, n:",oman_notw_dass.mean(),oman_notw_dass.std(),oman_notw_dass.count())
oman_tw_dass_res = pg.ttest(oman_tw_dass, oman_notw_dass, correction=True)
print(oman_tw_dass_res,"\n")


#Egypt twitter SAS t-test
egypt_tw_sas=alex_df.query('tw == True and nation_code==2')['SAS_total']
egypt_notw_sas=alex_df.query('tw == False and nation_code==2')['SAS_total']
print("egypt tw SAS mean, sd, n:", egypt_tw_sas.mean(),egypt_tw_sas.std(),egypt_tw_sas.count())
print("egypt no tw SAS mean, sd, n:",egypt_notw_sas.mean(),egypt_notw_sas.std(),egypt_notw_sas.count())
egypt_tw_sas_res = pg.ttest(egypt_tw_sas, egypt_notw_sas, correction=True)
print(egypt_tw_sas_res,"\n")

#Egypt twitter tAS t-test
egypt_tw_tas=alex_df.query('tw == True and nation_code==2')['TAS_total']
egypt_notw_tas=alex_df.query('tw == False and nation_code==2')['TAS_total']
print("egypt tw tAS mean, sd, n:", egypt_tw_tas.mean(),egypt_tw_tas.std(),egypt_tw_tas.count())
print("egypt no tw tAS mean, sd, n:",egypt_notw_tas.mean(),egypt_notw_tas.std(),egypt_notw_tas.count())
egypt_tw_tas_res = pg.ttest(egypt_tw_tas, egypt_notw_tas, correction=True)
print(egypt_tw_tas_res,"\n")

#Egypt twitter dass t-test
egypt_tw_dass=alex_df.query('tw == True and nation_code==2')['DASS_score']
egypt_notw_dass=alex_df.query('tw == False and nation_code==2')['DASS_score']
print("egypt tw dass mean, sd, n:", egypt_tw_dass.mean(),egypt_tw_dass.std(),egypt_tw_dass.count())
print("egypt no tw dass mean, sd, n:",egypt_notw_dass.mean(),egypt_notw_dass.std(),egypt_notw_dass.count())
egypt_tw_dass_res = pg.ttest(egypt_tw_dass, egypt_notw_dass, correction=True)
print(egypt_tw_dass_res,"\n")

#pakistan twitter SAS t-test
pakistan_tw_sas=alex_df.query('tw == True and nation_code==3')['SAS_total']
pakistan_notw_sas=alex_df.query('tw == False and nation_code==3')['SAS_total']
print("pakistan tw SAS mean, sd, n:", pakistan_tw_sas.mean(),pakistan_tw_sas.std(),pakistan_tw_sas.count())
print("pakistan no tw SAS mean, sd, n:",pakistan_notw_sas.mean(),pakistan_notw_sas.std(),pakistan_notw_sas.count())
pakistan_tw_sas_res = pg.ttest(pakistan_tw_sas, pakistan_notw_sas, correction=True)
print(pakistan_tw_sas_res,"\n")

#pakistan twitter tAS t-test
pakistan_tw_tas=alex_df.query('tw == True and nation_code==3')['TAS_total']
pakistan_notw_tas=alex_df.query('tw == False and nation_code==3')['TAS_total']
print("pakistan tw tAS mean, sd, n:", pakistan_tw_tas.mean(),pakistan_tw_tas.std(),pakistan_tw_tas.count())
print("pakistan no tw tAS mean, sd, n:",pakistan_notw_tas.mean(),pakistan_notw_tas.std(),pakistan_notw_tas.count())
pakistan_tw_tas_res = pg.ttest(pakistan_tw_tas, pakistan_notw_tas, correction=True)
print(pakistan_tw_tas_res,"\n")

#pakistan twitter dass t-test
pakistan_tw_dass=alex_df.query('tw == True and nation_code==3')['DASS_score']
pakistan_notw_dass=alex_df.query('tw == False and nation_code==3')['DASS_score']
print("pakistan tw dass mean, sd, n:", pakistan_tw_dass.mean(),pakistan_tw_dass.std(),pakistan_tw_dass.count())
print("pakistan no tw dass mean, sd, n:",pakistan_notw_dass.mean(),pakistan_notw_dass.std(),pakistan_notw_dass.count())
pakistan_tw_dass_res = pg.ttest(pakistan_tw_dass, pakistan_notw_dass, correction=True)
print(pakistan_tw_dass_res,"\n")

#whole population sex SAS ttest
male_sas = alex_df.query('Sex == 1')['SAS_total']
female_sas = alex_df.query('Sex == 2')['SAS_total']
print("Male SAS mean, sd, n:", male_sas.mean(),male_sas.std(),male_sas.count())
print("Female SAS mean, sd, n:",female_sas.mean(),female_sas.std(),female_sas.count())
res = pg.ttest(male_sas, female_sas, correction=True)
print(res,"\n")

#Oman sex SAS ttest
oman_male_sas = alex_df.query('Sex == 1 and nation_code==1')['SAS_total']
oman_female_sas = alex_df.query('Sex == 2 and nation_code==1')['SAS_total']
print("Oman Male SAS mean, sd, n:", oman_male_sas.mean(),oman_male_sas.std(),oman_male_sas.count())
print("Oman Female SAS mean, sd, n:",oman_female_sas.mean(),oman_female_sas.std(),oman_female_sas.count())
oman_sex_res = pg.ttest(oman_male_sas, oman_female_sas, correction=True)
print(oman_sex_res,"\n")

#Egypt sex SAS ttest
egypt_male_sas = alex_df.query('Sex == 1 and nation_code==2')['SAS_total']
egypt_female_sas = alex_df.query('Sex == 2 and nation_code==2')['SAS_total']
print("Egypt Male SAS mean, sd, n:", egypt_male_sas.mean(),egypt_male_sas.std(),egypt_male_sas.count())
print("egypt Female SAS mean, sd, n:",egypt_female_sas.mean(),egypt_female_sas.std(),egypt_female_sas.count())
egypt_sex_res = pg.ttest(egypt_male_sas, egypt_female_sas, correction=True)
print(egypt_sex_res,"\n")

#Pakistan sex SAS ttest
pakistan_male_sas = alex_df.query('Sex == 1 and nation_code==3')['SAS_total']
pakistan_female_sas = alex_df.query('Sex == 2 and nation_code==3')['SAS_total']
print("pakistan Male SAS mean, sd, n:", pakistan_male_sas.mean(),pakistan_male_sas.std(),pakistan_male_sas.count())
print("pakistan Female SAS mean, sd, n:",pakistan_female_sas.mean(),pakistan_female_sas.std(),pakistan_female_sas.count())
pakistan_sex_res = pg.ttest(pakistan_male_sas, pakistan_female_sas, correction=True)
print(pakistan_sex_res,"\n")

#overall sex TAS ttest
male_tas = alex_df.query('Sex == 1')['TAS_total']
female_tas = alex_df.query('Sex == 2')['TAS_total']
print("\nMale TAS mean, sd, n:", male_tas.mean(),male_tas.std(),male_tas.count())
print("Female TAS mean, sd, n:",female_tas.mean(),female_tas.std(),female_tas.count())
res2 = pg.ttest(male_tas, female_tas, correction=True)
print(res2)

#Oman sex TAS ttest
oman_male_tas = alex_df.query('Sex == 1 and nation_code==1')['TAS_total']
oman_female_tas = alex_df.query('Sex == 2 and nation_code==1')['TAS_total']
print("Oman Male TAS mean, sd, n:", oman_male_tas.mean(),oman_male_tas.std(),oman_male_tas.count())
print("Oman Female TAS mean, sd, n:",oman_female_tas.mean(),oman_female_tas.std(),oman_female_tas.count())
oman_sextas_res = pg.ttest(oman_male_tas, oman_female_tas, correction=True)
print(oman_sextas_res,"\n")

#Egypt sex TAS ttest
egypt_male_tas = alex_df.query('Sex == 1 and nation_code==2')['TAS_total']
egypt_female_tas = alex_df.query('Sex == 2 and nation_code==2')['TAS_total']
print("Egypt Male TAS mean, sd, n:", egypt_male_tas.mean(),egypt_male_tas.std(),egypt_male_tas.count())
print("Egypt Female TAS mean, sd, n:",egypt_female_tas.mean(),egypt_female_tas.std(),egypt_female_tas.count())
egypt_sextas_res = pg.ttest(egypt_male_tas, egypt_female_tas, correction=True)
print(egypt_sextas_res,"\n")

#Pakistan sex TAS ttest
pakistan_male_tas = alex_df.query('Sex == 1 and nation_code==3')['TAS_total']
pakistan_female_tas = alex_df.query('Sex == 2 and nation_code==3')['TAS_total']
print("Pakistan Male TAS mean, sd, n:", pakistan_male_tas.mean(),pakistan_male_tas.std(),pakistan_male_tas.count())
print("pakistan Female TAS mean, sd, n:",pakistan_female_tas.mean(),pakistan_female_tas.std(),pakistan_female_tas.count())
pakistan_sextas_res = pg.ttest(pakistan_male_tas, pakistan_female_tas, correction=True)
print(pakistan_sextas_res,"\n")

#sex DASS ttest
male_DASS = alex_df.query('Sex == 1')['DASS_score']
female_DASS = alex_df.query('Sex == 2')['DASS_score']
print("\nMale DASS mean, sd, n:", male_DASS.mean(),male_DASS.std(),male_DASS.count())
print("Female DASS mean, sd, n:",female_DASS.mean(),female_DASS.std(),female_DASS.count())
res3 = pg.ttest(male_DASS, female_DASS, correction=True)
print(res3)

#Oman sex DASS ttest
oman_male_dass = alex_df.query('Sex == 1 and nation_code==1')['DASS_score']
oman_female_dass = alex_df.query('Sex == 2 and nation_code==1')['DASS_score']
print("Oman Male DASS mean, sd, n:", oman_male_dass.mean(),oman_male_dass.std(),oman_male_dass.count())
print("Oman Female dass mean, sd, n:",oman_female_dass.mean(),oman_female_dass.std(),oman_female_dass.count())
oman_sexdass_res = pg.ttest(oman_male_dass, oman_female_dass, correction=True)
print(oman_sexdass_res,"\n")

#Egypt sex DASS ttest
egypt_male_dass = alex_df.query('Sex == 1 and nation_code==2')['DASS_score']
egypt_female_dass = alex_df.query('Sex == 2 and nation_code==2')['DASS_score']
print("Egypt Male DASS mean, sd, n:", egypt_male_dass.mean(),egypt_male_dass.std(),egypt_male_dass.count())
print("Egypt Female dass mean, sd, n:",egypt_female_dass.mean(),egypt_female_dass.std(),egypt_female_dass.count())
egypt_sexdass_res = pg.ttest(egypt_male_dass, egypt_female_dass, correction=True)
print(egypt_sexdass_res,"\n")

#Pakistan sex DASS ttest
pakistan_male_dass = alex_df.query('Sex == 1 and nation_code==3')['DASS_score']
pakistan_female_dass = alex_df.query('Sex == 2 and nation_code==3')['DASS_score']
print("Pakistan Male dass mean, sd, n:", pakistan_male_dass.mean(),pakistan_male_dass.std(),pakistan_male_dass.count())
print("pakistan Female dass mean, sd, n:",pakistan_female_dass.mean(),pakistan_female_dass.std(),pakistan_female_dass.count())
pakistan_sexdass_res = pg.ttest(pakistan_male_dass, pakistan_female_dass, correction=True)
print(pakistan_sexdass_res,"\n")

#Faculty SAS ttest
theo_sas = alex_df.query('Faculty == 1')['SAS_total']
prac_sas = alex_df.query('Faculty == 2')['SAS_total']
print("\nTheoretical SAS mean, sd, n:", theo_sas.mean(),theo_sas.std(),theo_sas.count())
print("Practical SAS mean, sd, n:",prac_sas.mean(),prac_sas.std(),prac_sas.count())
res4 = pg.ttest(theo_sas, prac_sas, correction=True)
print(res4)

#Faculty TAS ttest
theo_tas = alex_df.query('Faculty == 1')['TAS_total']
prac_tas = alex_df.query('Faculty == 2')['TAS_total']
print("\nTheoretical TAS mean, sd, n:", theo_tas.mean(),theo_tas.std(),theo_tas.count())
print("Practical TAS mean, sd, n:",prac_tas.mean(),prac_tas.std(),prac_tas.count())
res5 = pg.ttest(theo_tas, prac_tas, correction=True)
print(res5)

#Faculty DASS ttest
theo_DASS = alex_df.query('Faculty == 1')['DASS_score']
prac_DASS = alex_df.query('Faculty == 2')['DASS_score']
print("\nTheoretical DASS mean, sd, n:", theo_DASS.mean(),theo_DASS.std(),theo_DASS.count())
print("Practical DASS mean, sd, n:",prac_DASS.mean(),prac_DASS.std(),prac_DASS.count())
res6 = pg.ttest(theo_DASS, prac_DASS, correction=True)
print(res6)



#ACAD year SAS ANOVA
alex_df.rename(columns={"Academic year": "Acadyear"},inplace=True)
acad1_sas = alex_df.query("Acadyear == 1")['SAS_total']
acad2_sas = alex_df.query("Acadyear == 2")['SAS_total']
acad3_sas = alex_df.query("Acadyear == 3")['SAS_total']
acad4_sas = alex_df.query("Acadyear == 4")['SAS_total']
acad5_sas = alex_df.query("Acadyear == 5")['SAS_total']
acad6_sas = alex_df.query("Acadyear == 6")['SAS_total']
acad7_sas = alex_df.query("Acadyear == 7")['SAS_total']
print("\n1st year SAS mean sd n:",acad1_sas.mean(),acad1_sas.std(),acad1_sas.count())
print("2nd year SAS mean sd n:",acad2_sas.mean(),acad2_sas.std(),acad2_sas.count())
print("3rd year SAS mean sd n:",acad3_sas.mean(),acad3_sas.std(),acad3_sas.count())
print("4th year SAS mean sd n:",acad4_sas.mean(),acad4_sas.std(),acad4_sas.count())
print("5th year SAS mean sd n:",acad5_sas.mean(),acad5_sas.std(),acad5_sas.count())
print("6th year SAS mean sd n:",acad6_sas.mean(),acad6_sas.std(),acad6_sas.count())
print("7th year SAS mean sd n:",acad7_sas.mean(),acad7_sas.std(),acad7_sas.count())
print("Academic year SAS Anova",stats.f_oneway(acad1_sas,acad2_sas,acad3_sas,acad4_sas,acad5_sas,acad6_sas,acad7_sas))

#ACAD year TAS ANOVA
acad1_tas = alex_df.query("Acadyear == 1")['TAS_total']
acad2_tas = alex_df.query("Acadyear == 2")['TAS_total']
acad3_tas = alex_df.query("Acadyear == 3")['TAS_total']
acad4_tas = alex_df.query("Acadyear == 4")['TAS_total']
acad5_tas = alex_df.query("Acadyear == 5")['TAS_total']
acad6_tas = alex_df.query("Acadyear == 6")['TAS_total']
acad7_tas = alex_df.query("Acadyear == 7")['TAS_total']
print("\n1st year TAS mean sd n:",acad1_tas.mean(),acad1_tas.std(),acad1_tas.count())
print("2nd year TAS mean sd n:",acad2_tas.mean(),acad2_tas.std(),acad2_tas.count())
print("3rd year TAS mean sd n:",acad3_tas.mean(),acad3_tas.std(),acad3_tas.count())
print("4th year TAS mean sd n:",acad4_tas.mean(),acad4_tas.std(),acad4_tas.count())
print("5th year TAS mean sd n:",acad5_tas.mean(),acad5_tas.std(),acad5_tas.count())
print("6th year TAS mean sd n:",acad6_tas.mean(),acad6_tas.std(),acad6_tas.count())
print("7th year TAS mean sd n:",acad7_tas.mean(),acad7_tas.std(),acad7_tas.count())
print("Academic year TAS Anova",stats.f_oneway(acad1_tas,acad2_tas,acad3_tas,acad4_tas,acad5_tas,acad6_tas,acad7_tas))

#ACAD year DASS ANOVA
acad1_DASS = alex_df.query("Acadyear == 1")['DASS_score']
acad2_DASS = alex_df.query("Acadyear == 2")['DASS_score']
acad3_DASS = alex_df.query("Acadyear == 3")['DASS_score']
acad4_DASS = alex_df.query("Acadyear == 4")['DASS_score']
acad5_DASS = alex_df.query("Acadyear == 5")['DASS_score']
acad6_DASS = alex_df.query("Acadyear == 6")['DASS_score']
acad7_DASS = alex_df.query("Acadyear == 7")['DASS_score']
print("\n1st year DASS mean sd n:",acad1_DASS.mean(),acad1_DASS.std(),acad1_DASS.count())
print("2nd year DASS mean sd n:",acad2_DASS.mean(),acad2_DASS.std(),acad2_DASS.count())
print("3rd year DASS mean sd n:",acad3_DASS.mean(),acad3_DASS.std(),acad3_DASS.count())
print("4th year DASS mean sd n:",acad4_DASS.mean(),acad4_DASS.std(),acad4_DASS.count())
print("5th year DASS mean sd n:",acad5_DASS.mean(),acad5_DASS.std(),acad5_DASS.count())
print("6th year DASS mean sd n:",acad6_DASS.mean(),acad6_DASS.std(),acad6_DASS.count())
print("7th year DASS mean sd n:",acad7_DASS.mean(),acad7_DASS.std(),acad7_DASS.count())
print("Academic year DASS Anova",stats.f_oneway(acad1_DASS,acad2_DASS,acad3_DASS,acad4_DASS,acad5_DASS\
,acad6_DASS,acad7_DASS))


#test for independence of alexythemia pos/neg & SA pos/neg
#M/F SAS chisq test
male_sas_tas= alex_df[(alex_df["Sex"]==1) & (alex_df["SAS_total"] >=31) & (alex_df["TAS_total"]>=60)]
female_sas_tas= alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] >=33) & (alex_df["TAS_total"]>=60)]
male_sas_notas= alex_df[(alex_df["Sex"]==1) & (alex_df["SAS_total"] >=31) & (alex_df["TAS_total"]<60)]
female_sas_notas= alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] >=33) & (alex_df["TAS_total"]<60)]
male_nosas_tas= alex_df[(alex_df["Sex"]==1) & (alex_df["SAS_total"] <31) & (alex_df["TAS_total"]>=60)]
female_nosas_tas= alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] <33) & (alex_df["TAS_total"]>=60)]
male_nosas_notas= alex_df[(alex_df["Sex"]==1) & (alex_df["SAS_total"] <31) & (alex_df["TAS_total"]<60)]
female_nosas_notas= alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] <33) & (alex_df["TAS_total"]<60)]
tas_sas_df=[[len(male_sas_tas)+len(female_sas_tas),len(male_sas_notas)+len(female_sas_notas)],[len(male_nosas_tas)+len(female_nosas_tas),
len(male_nosas_notas)+len(female_nosas_notas)]]
print("\n",tas_sas_df)
stat_tas_sas, ptas_sas, doftas_sas, expectedtas_sas = chi2_contingency(tas_sas_df)
print("TAS SAS total ChiSq p value=",stat_tas_sas, ptas_sas,"\n")

#test for independence of  SA pos/neg & DASS severe
male_sas_dass= alex_df[(alex_df["Sex"]==1) & (alex_df["SAS_total"] >=31) & (alex_df["DASS_score"]>=60)]
female_sas_dass= alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] >=33) & (alex_df["DASS_score"]>=60)]
male_sas_nodass= alex_df[(alex_df["Sex"]==1) & (alex_df["SAS_total"] >=31) & (alex_df["DASS_score"]<60)]
female_sas_nodass= alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] >=33) & (alex_df["DASS_score"]<60)]
male_nosas_dass= alex_df[(alex_df["Sex"]==1) & (alex_df["SAS_total"] <31) & (alex_df["DASS_score"]>=60)]
female_nosas_dass= alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] <33) & (alex_df["DASS_score"]>=60)]
male_nosas_nodass= alex_df[(alex_df["Sex"]==1) & (alex_df["SAS_total"] <31) & (alex_df["DASS_score"]<60)]
female_nosas_nodass= alex_df[(alex_df["Sex"]==2) & (alex_df["SAS_total"] <33) & (alex_df["DASS_score"]<60)]
sas_dass_df=[[len(male_sas_dass)+len(female_sas_dass),len(male_sas_nodass)+len(female_sas_nodass)],\
[len(male_nosas_dass)+len(female_nosas_dass),
len(male_nosas_nodass)+len(female_nosas_nodass)]]
print("\n",sas_dass_df)
stat_sas_dass, psas_dass, dofsas_dass, expectedsas_dass = chi2_contingency(sas_dass_df)
print("SAS DASS total ChiSq p value=",stat_sas_dass, psas_dass,"\n")

#test for independence of  alexythemia pos/neg(TAS) & DASS severe
tas_dass= alex_df[(alex_df["TAS_total"]>=60) & (alex_df["DASS_score"]>=60)]
tas_nodass= alex_df[(alex_df["TAS_total"]>=60) & (alex_df["DASS_score"]<60)]
notas_dass= alex_df[(alex_df["TAS_total"]<60) & (alex_df["DASS_score"]>=60)]
notas_nodass= alex_df[(alex_df["TAS_total"]<60) & (alex_df["DASS_score"]<60)]
tas_dass_df=[[len(tas_dass),len(tas_nodass)],\
[len(notas_dass),len(notas_nodass)]]
print("\n",tas_dass_df)
stat_tas_dass, ptas_dass, doftas_dass, expectedtas_dass = chi2_contingency(tas_dass_df)
print("TAS DASS total ChiSq p value=",stat_tas_dass, ptas_dass,"\n")


#2nd article structure

#linreg age vs SAS
y = alex_df['SAS_total']
x = alex_df[['Age']]
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
print(model.summary())

#Linreg age vs TAS
y2=alex_df['TAS_total']
model2 = sm.OLS(y2, x).fit()
print(model2.summary())

#Linreg age vs DASS
y3=alex_df['DASS_score']
model3 = sm.OLS(y3, x).fit()
print(model3.summary())


#linreg SAS total vs tas Total
x2=alex_df[['SAS_total']]
x2 = sm.add_constant(x2)
model4=sm.OLS(y2,x2).fit()
print(model4.summary())

#linreg SAS total vs DASS Total
model5=sm.OLS(y3,x2).fit()
print(model5.summary())

#linreg TAS total vs DASS Total
x3=alex_df[['TAS_total']]
x3 = sm.add_constant(x3)
model6=sm.OLS(y3,x3).fit()
print(model6.summary())



sns.lmplot(data=alex_df,x='Age',y='SAS_total',
scatter_kws={"color": "dodgerblue"}, line_kws={"color": "black"})
plt.show()

sns.lmplot(data=alex_df,x='Age',y='TAS_total',
scatter_kws={"color": "dodgerblue"}, line_kws={"color": "black"})
plt.show()

sns.lmplot(data=alex_df,x='Age',y='DASS_score',
scatter_kws={"color": "dodgerblue"}, line_kws={"color": "black"})
plt.show()



sns.lmplot(data=alex_df,x='SAS_total',y='TAS_total',
scatter_kws={"color": "dodgerblue"}, line_kws={"color": "black"})
plt.show()

sns.lmplot(data=alex_df,x='SAS_total',y='DASS_score',
scatter_kws={"color": "dodgerblue"}, line_kws={"color": "black"})
plt.show()

sns.lmplot(data=alex_df,x='TAS_total',y='DASS_score',
scatter_kws={"color": "dodgerblue"}, line_kws={"color": "black"})
plt.show()


#place residence SAS ANOVA
alex_df.rename(columns={"Place of residence": "Plac_res"},inplace=True)
res1_sas = alex_df.query("Plac_res == 1")['SAS_total']
res2_sas = alex_df.query("Plac_res == 2")['SAS_total']
res3_sas = alex_df.query("Plac_res == 3")['SAS_total']
res4_sas = alex_df.query("Plac_res == 4")['SAS_total']
print("\nFamily res SAS mean sd n:",res1_sas.mean(),res1_sas.std(),res1_sas.count())
print("Friend res SAS mean sd n:",res2_sas.mean(),res2_sas.std(),res2_sas.count())
print("Alone res SAS mean sd n:",res3_sas.mean(),res3_sas.std(),res3_sas.count())
print("Hostel res SAS mean sd n:",res4_sas.mean(),res4_sas.std(),res4_sas.count())
print("Residesnce SAS Anova",stats.f_oneway(res1_sas,res2_sas,res3_sas,res4_sas))

#place residence TAS ANOVA
res1_tas = alex_df.query("Plac_res == 1")['TAS_total']
res2_tas = alex_df.query("Plac_res == 2")['TAS_total']
res3_tas = alex_df.query("Plac_res == 3")['TAS_total']
res4_tas = alex_df.query("Plac_res == 4")['TAS_total']
print("\nFamily res TAS mean sd n:",res1_tas.mean(),res1_tas.std(),res1_tas.count())
print("Friend res TAS mean sd n:",res2_tas.mean(),res2_tas.std(),res2_tas.count())
print("Alone res TAS mean sd n:",res3_tas.mean(),res3_tas.std(),res3_tas.count())
print("Hostel res TAS mean sd n:",res4_tas.mean(),res4_tas.std(),res4_tas.count())
print("Residesnce TAS Anova",stats.f_oneway(res1_tas,res2_tas,res3_tas,res4_tas))

#place residence DASS ANOVA
res1_dass = alex_df.query("Plac_res == 1")['DASS_score']
res2_dass = alex_df.query("Plac_res == 2")['DASS_score']
res3_dass = alex_df.query("Plac_res == 3")['DASS_score']
res4_dass = alex_df.query("Plac_res == 4")['DASS_score']
print("\nFamily res DASS mean sd n:",res1_dass.mean(),res1_dass.std(),res1_dass.count())
print("Friend res DASS mean sd n:",res2_dass.mean(),res2_dass.std(),res2_dass.count())
print("Alone res DASS mean sd n:",res3_dass.mean(),res3_dass.std(),res3_dass.count())
print("Hostel res DASS mean sd n:",res4_dass.mean(),res4_dass.std(),res4_dass.count())
print("Residesnce DASS Anova",stats.f_oneway(res1_dass,res2_dass,res3_dass,res4_dass))

#income SAS ANOVA
inc1_sas = alex_df.query("Income == 1")['SAS_total']
inc2_sas = alex_df.query("Income == 2")['SAS_total']
inc3_sas = alex_df.query("Income == 3")['SAS_total']
print("\nlow income SAS mean sd n:",inc1_sas.mean(),inc1_sas.std(),inc1_sas.count())
print("middle income SAS mean sd n:",inc2_sas.mean(),inc2_sas.std(),inc2_sas.count())
print("high income SAS mean sd n:",inc3_sas.mean(),inc3_sas.std(),inc3_sas.count())
print("Income SAS Anova",stats.f_oneway(inc1_sas,inc2_sas,inc3_sas))

#income TAS ANOVA
inc1_tas = alex_df.query("Income == 1")['TAS_total']
inc2_tas = alex_df.query("Income == 2")['TAS_total']
inc3_tas = alex_df.query("Income == 3")['TAS_total']
print("\nlow income TAS mean sd n:",inc1_tas.mean(),inc1_tas.std(),inc1_tas.count())
print("middle income TAS mean sd n:",inc2_tas.mean(),inc2_tas.std(),inc2_tas.count())
print("high income TAS mean sd n:",inc3_tas.mean(),inc3_tas.std(),inc3_tas.count())
print("Income TAS Anova",stats.f_oneway(inc1_tas,inc2_tas,inc3_tas))

#income DASS ANOVA
inc1_dass = alex_df.query("Income == 1")['DASS_score']
inc2_dass = alex_df.query("Income == 2")['DASS_score']
inc3_dass = alex_df.query("Income == 3")['DASS_score']
print("\nlow income DASS mean sd n:",inc1_dass.mean(),inc1_dass.std(),inc1_dass.count())
print("middle income DASS mean sd n:",inc2_dass.mean(),inc2_dass.std(),inc2_dass.count())
print("high income DASS mean sd n:",inc3_dass.mean(),inc3_dass.std(),inc3_dass.count())
print("Income DASS Anova",stats.f_oneway(inc1_dass,inc2_dass,inc3_dass))


#Monthly Bill SAS ANOVA
month1_sas = alex_df.query("month_bill == 1")['SAS_total']
month2_sas = alex_df.query("month_bill == 2")['SAS_total']
month3_sas = alex_df.query("month_bill == 3")['SAS_total']
month4_sas = alex_df.query("month_bill == 4")['SAS_total']
print("\nvery low bill SAS mean sd n:",month1_sas.mean(),month1_sas.std(),month1_sas.count())
print("low bill SAS mean sd n:",month2_sas.mean(),month2_sas.std(),month2_sas.count())
print("middle SAS mean sd n:",month3_sas.mean(),month3_sas.std(),month3_sas.count())
print("high SAS mean sd n:",month4_sas.mean(),month4_sas.std(),month4_sas.count())
print("monthly bill SAS Anova",stats.f_oneway(month1_sas,month2_sas,month3_sas,month4_sas))

#Monthly Bill TAS ANOVA
month1_tas = alex_df.query("month_bill == 1")['TAS_total']
month2_tas = alex_df.query("month_bill == 2")['TAS_total']
month3_tas = alex_df.query("month_bill == 3")['TAS_total']
month4_tas = alex_df.query("month_bill == 4")['TAS_total']
print("\nvery low bill TAS mean sd n:",month1_tas.mean(),month1_tas.std(),month1_tas.count())
print("low bill TAS mean sd n:",month2_tas.mean(),month2_tas.std(),month2_tas.count())
print("middle TAS mean sd n:",month3_tas.mean(),month3_tas.std(),month3_tas.count())
print("high TAS mean sd n:",month4_tas.mean(),month4_tas.std(),month4_tas.count())
print("monthly bill TAS Anova",stats.f_oneway(month1_tas,month2_tas,month3_tas,month4_tas))

#Monthly Bill DASS ANOVA
month1_dass = alex_df.query("month_bill == 1")['DASS_score']
month2_dass  = alex_df.query("month_bill == 2")['DASS_score']
month3_dass  = alex_df.query("month_bill == 3")['DASS_score']
month4_dass  = alex_df.query("month_bill == 4")['DASS_score']
print("\nvery low bill DASS mean sd n:",month1_dass.mean(),month1_dass.std(),month1_dass.count())
print("low bill DASS mean sd n:",month2_dass.mean(),month2_dass.std(),month2_dass.count())
print("middle DASS mean sd n:",month3_dass.mean(),month3_dass.std(),month3_dass.count())
print("high DASS mean sd n:",month4_dass.mean(),month4_dass.std(),month4_dass.count())
print("monthly bill DASS Anova",stats.f_oneway(month1_dass,month2_dass,month3_dass,month4_dass))


#Nation SAS ANOVA
oman_sas = alex_df.query("nation_code == 1")['SAS_total']
egypt_sas = alex_df.query("nation_code == 2")['SAS_total']
pakistan_sas = alex_df.query("nation_code == 3")['SAS_total']
print("\nOman SAS mean sd n:",oman_sas.mean(),oman_sas.std(),oman_sas.count())
print("Egypt SAS mean sd n:",egypt_sas.mean(),egypt_sas.std(),egypt_sas.count())
print("Pakistan SAS mean sd n:",pakistan_sas.mean(),pakistan_sas.std(),pakistan_sas.count())
print("Nation SAS Anova",stats.f_oneway(oman_sas,egypt_sas,pakistan_sas))

#Nation TAS ANOVA
oman_tas = alex_df.query("nation_code == 1")['TAS_total']
egypt_tas = alex_df.query("nation_code == 2")['TAS_total']
pakistan_tas = alex_df.query("nation_code == 3")['TAS_total']
print("\nOman TAS mean sd n:",oman_tas.mean(),oman_tas.std(),oman_tas.count())
print("Egypt TAS mean sd n:",egypt_tas.mean(),egypt_tas.std(),egypt_tas.count())
print("Pakistan TAS mean sd n:",pakistan_tas.mean(),pakistan_tas.std(),pakistan_tas.count())
print("Nation TAS Anova",stats.f_oneway(oman_tas,egypt_tas,pakistan_tas))

#Nation DASS ANOVA
oman_dass = alex_df.query("nation_code == 1")['DASS_score']
egypt_dass = alex_df.query("nation_code == 2")['DASS_score']
pakistan_dass = alex_df.query("nation_code == 3")['DASS_score']
print("\nOman DASS mean sd n:",oman_dass.mean(),oman_dass.std(),oman_dass.count())
print("Egypt DASS mean sd n:",egypt_dass.mean(),egypt_dass.std(),egypt_dass.count())
print("Pakistan DASS mean sd n:",pakistan_dass.mean(),pakistan_dass.std(),pakistan_dass.count())
print("Nation DASS Anova",stats.f_oneway(oman_dass,egypt_dass,pakistan_dass))


#Frequency of Smartphone use ANOVAs
freq=[]
for index, row in alex_df.iterrows():
    if '0' in row['Frequency']:
        freq.append(0)
    elif '1-' in row['Frequency']:
        freq.append(1)
    elif '2-' in row['Frequency']:
        freq.append(2)
    else:
        freq.append(3)
alex_df['freq_code'] = freq

lowest_freq_sas = alex_df.query("freq_code == 0")['SAS_total']
low_freq_sas = alex_df.query("freq_code == 1")['SAS_total']
med_freq_sas = alex_df.query("freq_code == 2")['SAS_total']
high_freq_sas = alex_df.query("freq_code == 3")['SAS_total']
print("\n0-1 SAS mean sd n:",lowest_freq_sas.mean(),lowest_freq_sas.std(),lowest_freq_sas.count())
print("1-2 SAS mean sd n:",low_freq_sas.mean(),low_freq_sas.std(),low_freq_sas.count())
print("2-3 SAS mean sd n:",med_freq_sas.mean(),med_freq_sas.std(),med_freq_sas.count())
print("4+ SAS mean sd n:",high_freq_sas.mean(),high_freq_sas.std(),high_freq_sas.count())
print("Frequency SAS Anova",stats.f_oneway(lowest_freq_sas,low_freq_sas,med_freq_sas,high_freq_sas))

lowest_freq_tas = alex_df.query("freq_code == 0")['TAS_total']
low_freq_tas = alex_df.query("freq_code == 1")['TAS_total']
med_freq_tas = alex_df.query("freq_code == 2")['TAS_total']
high_freq_tas = alex_df.query("freq_code == 3")['TAS_total']
print("\n0-1 TAS mean sd n:",lowest_freq_tas.mean(),lowest_freq_tas.std(),lowest_freq_tas.count())
print("1-2 TAS mean sd n:",low_freq_tas.mean(),low_freq_tas.std(),low_freq_tas.count())
print("2-3 TAS mean sd n:",med_freq_tas.mean(),med_freq_tas.std(),med_freq_tas.count())
print("4+ TAS mean sd n:",high_freq_tas.mean(),high_freq_tas.std(),high_freq_tas.count())
print("Frequency TAS Anova",stats.f_oneway(lowest_freq_tas,low_freq_tas,med_freq_tas,high_freq_tas))

lowest_freq_dass = alex_df.query("freq_code == 0")['DASS_score']
low_freq_dass = alex_df.query("freq_code == 1")['DASS_score']
med_freq_dass = alex_df.query("freq_code == 2")['DASS_score']
high_freq_dass = alex_df.query("freq_code == 3")['DASS_score']
print("\n0-1 DASS mean sd n:",lowest_freq_dass.mean(),lowest_freq_dass.std(),lowest_freq_dass.count())
print("1-2 DASS mean sd n:",low_freq_dass.mean(),low_freq_dass.std(),low_freq_dass.count())
print("2-3 DASS mean sd n:",med_freq_dass.mean(),med_freq_dass.std(),med_freq_dass.count())
print("4+ DASS mean sd n:",high_freq_dass.mean(),high_freq_dass.std(),high_freq_dass.count())
print("Frequency DASS Anova",stats.f_oneway(lowest_freq_dass,low_freq_dass,med_freq_dass,high_freq_dass))


#Academic performance ANOVAs
lowest_grade_sas = alex_df.query("grade == 1")['SAS_total']
low_grade_sas = alex_df.query("grade == 2")['SAS_total']
med_grade_sas = alex_df.query("grade == 3")['SAS_total']
high_grade_sas = alex_df.query("grade == 4")['SAS_total']
print("\npass SAS mean sd n:",lowest_grade_sas.mean(),lowest_grade_sas.std(),lowest_grade_sas.count())
print("good SAS mean sd n:",low_grade_sas.mean(),low_grade_sas.std(),low_grade_sas.count())
print("very good SAS mean sd n:",med_grade_sas.mean(),med_grade_sas.std(),med_grade_sas.count())
print("excellent SAS mean sd n:",high_grade_sas.mean(),high_grade_sas.std(),high_grade_sas.count())
print("Total grade SAS Anova",stats.f_oneway(lowest_grade_sas,low_grade_sas,med_grade_sas,high_grade_sas))

lowest_grade_tas = alex_df.query("grade == 1")['TAS_total']
low_grade_tas = alex_df.query("grade == 2")['TAS_total']
med_grade_tas = alex_df.query("grade == 3")['TAS_total']
high_grade_tas = alex_df.query("grade == 4")['TAS_total']
print("\npass TAS mean sd n:",lowest_grade_tas.mean(),lowest_grade_tas.std(),lowest_grade_tas.count())
print("good TAS mean sd n:",low_grade_tas.mean(),low_grade_tas.std(),low_grade_tas.count())
print("very good TAS mean sd n:",med_grade_tas.mean(),med_grade_tas.std(),med_grade_tas.count())
print("excellent TAS mean sd n:",high_grade_tas.mean(),high_grade_tas.std(),high_grade_tas.count())
print("Total grade TAS Anova",stats.f_oneway(lowest_grade_tas,low_grade_tas,med_grade_tas,high_grade_tas))

lowest_grade_dass = alex_df.query("grade == 1")['DASS_score']
low_grade_dass = alex_df.query("grade == 2")['DASS_score']
med_grade_dass = alex_df.query("grade == 3")['DASS_score']
high_grade_dass = alex_df.query("grade == 4")['DASS_score']
print("\npass DASS mean sd n:",lowest_grade_dass.mean(),lowest_grade_dass.std(),lowest_grade_dass.count())
print("good DASS mean sd n:",low_grade_dass.mean(),low_grade_dass.std(),low_grade_dass.count())
print("very good DASS mean sd n:",med_grade_dass.mean(),med_grade_dass.std(),med_grade_dass.count())
print("excellent DASS mean sd n:",high_grade_dass.mean(),high_grade_dass.std(),high_grade_dass.count())
print("Total grade DASS Anova",stats.f_oneway(lowest_grade_dass,low_grade_dass,med_grade_dass,high_grade_dass))


#Marital status t-tests
single_sas = alex_df.query("marital == 1")['SAS_total']
married_sas = alex_df.query("marital == 2")['SAS_total']
print("\nsingle SAS mean sd n:",single_sas.mean(),single_sas.std(),single_sas.count())
print("married SAS mean sd n:",married_sas.mean(),married_sas.std(),married_sas.count())
maternal_sas_t = pg.ttest(single_sas, married_sas, correction=True)
print(maternal_sas_t)

single_tas = alex_df.query("marital == 1")['TAS_total']
married_tas = alex_df.query("marital == 2")['TAS_total']
print("\nsingle TAS mean sd n:",single_tas.mean(),single_tas.std(),single_tas.count())
print("married TAS mean sd n:",married_tas.mean(),married_tas.std(),married_tas.count())
maternal_tas_t = pg.ttest(single_tas, married_tas, correction=True)
print(maternal_tas_t)

single_dass = alex_df.query("marital == 1")['DASS_score']
married_dass = alex_df.query("marital == 2")['DASS_score']
print("\nsingle DASS mean sd n:",single_dass.mean(),single_dass.std(),single_dass.count())
print("married DASS mean sd n:",married_dass.mean(),married_dass.std(),married_dass.count())
maternal_dass_t = pg.ttest(single_dass, married_dass, correction=True)
print(maternal_dass_t)

#environment ANOVAS
urban_sas=alex_df.query("Environment == 1")['SAS_total']
rural_sas=alex_df.query("Environment == 2")['SAS_total']
mountain_sas=alex_df.query("Environment == 3")['SAS_total']
print("\nurban SAS mean sd n", urban_sas.mean(),urban_sas.std(),urban_sas.count())
print("rural SAS mean sd n", rural_sas.mean(),rural_sas.std(),rural_sas.count())
print("mountain SAS mean sd n", mountain_sas.mean(),mountain_sas.std(),mountain_sas.count())
print("Total environment SAS Anova",stats.f_oneway(urban_sas,rural_sas,mountain_sas))

urban_tas=alex_df.query("Environment == 1")['TAS_total']
rural_tas=alex_df.query("Environment == 2")['TAS_total']
mountain_tas=alex_df.query("Environment == 3")['TAS_total']
print("\nurban TAS mean sd n", urban_tas.mean(),urban_tas.std(),urban_tas.count())
print("rural TAS mean sd n", rural_tas.mean(),rural_tas.std(),rural_tas.count())
print("mountain TAS mean sd n", mountain_tas.mean(),mountain_tas.std(),mountain_tas.count())
print("Total environment TAS Anova",stats.f_oneway(urban_tas,rural_tas,mountain_tas))

urban_dass=alex_df.query("Environment == 1")['DASS_score']
rural_dass=alex_df.query("Environment == 2")['DASS_score']
mountain_dass=alex_df.query("Environment == 3")['DASS_score']
print("\nurban DASS mean sd n", urban_dass.mean(),urban_dass.std(),urban_dass.count())
print("rural DASS mean sd n", rural_dass.mean(),rural_dass.std(),rural_dass.count())
print("mountain DASS mean sd n", mountain_dass.mean(),mountain_dass.std(),mountain_dass.count())
print("Total environment DASS Anova",stats.f_oneway(urban_dass,rural_dass,mountain_dass))

#social media frequency ANOVAS
never_sas=alex_df.query("socmed_freq == 1")['SAS_total']
rarely_sas=alex_df.query("socmed_freq == 2")['SAS_total']
occasionally_sas=alex_df.query("socmed_freq == 3")['SAS_total']
frequent_sas=alex_df.query("socmed_freq == 4")['SAS_total']
print("\nnever SAS mean sd n", never_sas.mean(),never_sas.std(),never_sas.count())
print("rarely SAS mean sd n", rarely_sas.mean(),rarely_sas.std(),rarely_sas.count())
print("occasionally SAS mean sd n", occasionally_sas.mean(),occasionally_sas.std(),occasionally_sas.count())
print("frequent SAS mean sd n",frequent_sas.mean(),frequent_sas.std(),frequent_sas.count())
print("Social Media frequency SAS Anova",stats.f_oneway(never_sas,rarely_sas,occasionally_sas,frequent_sas))

never_tas=alex_df.query("socmed_freq == 1")['TAS_total']
rarely_tas=alex_df.query("socmed_freq == 2")['TAS_total']
occasionally_tas=alex_df.query("socmed_freq == 3")['TAS_total']
frequent_tas=alex_df.query("socmed_freq == 4")['TAS_total']
print("\nnever TAS mean sd n", never_tas.mean(),never_tas.std(),never_tas.count())
print("rarely TAS mean sd n", rarely_tas.mean(),rarely_tas.std(),rarely_tas.count())
print("occasionally TAS mean sd n", occasionally_tas.mean(),occasionally_tas.std(),occasionally_tas.count())
print("frequent TAS mean sd n",frequent_tas.mean(),frequent_tas.std(),frequent_tas.count())
print("Social Media frequency TAS Anova",stats.f_oneway(never_tas,rarely_tas,occasionally_tas,frequent_tas))

never_dass=alex_df.query("socmed_freq == 1")['DASS_score']
rarely_dass=alex_df.query("socmed_freq == 2")['DASS_score']
occasionally_dass=alex_df.query("socmed_freq == 3")['DASS_score']
frequent_dass=alex_df.query("socmed_freq == 4")['DASS_score']
print("\nnever DASS mean sd n", never_dass.mean(),never_dass.std(),never_dass.count())
print("rarely DASS mean sd n", rarely_dass.mean(),rarely_dass.std(),rarely_dass.count())
print("occasionally DASS mean sd n", occasionally_dass.mean(),occasionally_dass.std(),occasionally_dass.count())
print("frequent DASS mean sd n",frequent_dass.mean(),frequent_dass.std(),frequent_dass.count())
print("Social Media frequency DASS Anova",stats.f_oneway(never_dass,rarely_dass,occasionally_dass,frequent_dass))

#
#Hours usage LinRegs
hourlinreg_df=alex_df[["hour_usage","SAS_total","TAS_total","DASS_score"]].copy()
hourlinreg_df.drop([1279],inplace=True)
hourlinreg_df["hour_usage"] = pd.to_numeric(hourlinreg_df["hour_usage"])

y3=hourlinreg_df['SAS_total']
x3 = hourlinreg_df[['hour_usage']]
x3 = sm.add_constant(x3)
model3 = sm.OLS(y3,x3).fit()
print("\n",model3.summary())

y4=hourlinreg_df["TAS_total"]
model4 = sm.OLS(y4, x3).fit()
print(model4.summary())

y5=hourlinreg_df["DASS_score"]
model5 = sm.OLS(y5, x3).fit()
print(model5.summary())

sns.lmplot(data=hourlinreg_df,x='hour_usage',y='SAS_total',
scatter_kws={"color": "dodgerblue"}, line_kws={"color": "black"})
plt.show()

sns.lmplot(data=hourlinreg_df,x='hour_usage',y='TAS_total',
scatter_kws={"color": "dodgerblue"}, line_kws={"color": "black"})
plt.show()

sns.lmplot(data=hourlinreg_df,x='hour_usage',y='DASS_score',
scatter_kws={"color": "dodgerblue"}, line_kws={"color": "black"})
plt.show()




#Paying for internet attractions T-tests
pay_sas=alex_df.query("pay_attract == 1")['SAS_total']
no_pay_sas=alex_df.query("pay_attract == 2")['SAS_total']
print("\npay SAS mean sd n:",pay_sas.mean(),pay_sas.std(),pay_sas.count())
print("no pay SAS mean sd n:",no_pay_sas.mean(),no_pay_sas.std(),no_pay_sas.count())
pay_attract_sas_t = pg.ttest(pay_sas, no_pay_sas, correction=True)
print(pay_attract_sas_t)

pay_tas=alex_df.query("pay_attract == 1")['TAS_total']
no_pay_tas=alex_df.query("pay_attract == 2")['TAS_total']
print("\npay TAS mean sd n:",pay_tas.mean(),pay_tas.std(),pay_tas.count())
print("no pay TAS mean sd n:",no_pay_tas.mean(),no_pay_tas.std(),no_pay_tas.count())
pay_attract_tas_t = pg.ttest(pay_tas, no_pay_tas, correction=True)
print(pay_attract_tas_t)

pay_dass=alex_df.query("pay_attract == 1")['DASS_score']
no_pay_dass=alex_df.query("pay_attract == 2")['DASS_score']
print("\npay DASS mean sd n:",pay_dass.mean(),pay_dass.std(),pay_dass.count())
print("no pay DASS mean sd n:",no_pay_dass.mean(),no_pay_dass.std(),no_pay_dass.count())
pay_attract_dass_t = pg.ttest(pay_dass, no_pay_dass, correction=True)
print(pay_attract_dass_t)

prof = ProfileReport(alex_df)
prof.to_file(output_file='output.html')

#Multiregs
alex_df.drop([1279],inplace=True)
SAS_df=alex_df['SAS_total']
TAS_df=alex_df['TAS_total']
DASS_df=alex_df['DASS_score']

alex_df.drop(columns=['platforms','Evaluation of the purpose of phone use',
'smart_addicted','Nationality','alex_pos','Frequency','(ALX)_1','(ALX)_2','(ALX)_3','(ALX)_4','(ALX)_5',
'(ALX)_6','(ALX)_7','(ALX)_8','(ALX)_9','(ALX)_10','(ALX)_11','(ALX)_12','(ALX)_13','(ALX)_14','(ALX)_15',
'(ALX)_16','(ALX)_17','(ALX)_18','(ALX)_19','(ALX)_20','(SAS)_1','(SAS)_2','(SAS)_3','(SAS)_4','(SAS)_5','(SAS)_6',
'(SAS)_7','(SAS)_8','(SAS)_9','(SAS)_10','SAS_total','TAS_total','Acadyear','month_bill','(DASS)_1', '(DASS)_2', '(DASS)_3', '(DASS)_4', '(DASS)_5', '(DASS)_6', '(DASS)_7', '(DASS)_8',
'(DASS)_9', '(DASS)_10', '(DASS)_11', '(DASS)_12', '(DASS)_13', '(DASS)_14', '(DASS)_15', '(DASS)_16',
 '(DASS)_17', '(DASS)_18', '(DASS)_19', '(DASS)_20', '(DASS)_21','DASS_score','DASS_severe',\
 'platforms',"WA","IG","FB","SC","tw",'socmed_freq','Environment', 'Plac_res',
 'Income', 'pay_attract', 'freq_code','Faculty', 'grade'],inplace=True)
print(alex_df.columns)
alex_df['hour_usage'] = pd.to_numeric(alex_df['hour_usage'])

alex_df['Sex'] = alex_df['Sex'].astype('category')
alex_df['marital'] = alex_df['marital'].astype('category')
alex_df['nation_code'] = alex_df['nation_code'].astype('category')
alex_df=pd.get_dummies(alex_df)
alex_df.drop(columns=['marital_3','marital_4'],inplace=True)
#alex_df.drop(['nation_code'],inplace=True)
#alex_df = pd.merge(left=alex_df,right=hc_nation,left_index=True,right_index=True)
scaler = StandardScaler()
alex_df[["Age","hour_usage"]] = scaler.fit_transform(alex_df[["Age","hour_usage"]])
#alex_df.drop(columns=['nation_code'],inplace=True)
print(alex_df.head())

#TAS LinRegs
tas_linreg = LinearRegression()
tas_linreg.fit(alex_df, TAS_df)
tas_linreg_pred=tas_linreg.predict(alex_df)
tas_linreg_rmse=np.sqrt(mean_squared_error(tas_linreg_pred, TAS_df))
print("RMSE for linear regression model:",tas_linreg_rmse)

TAS_DT_reg = DecisionTreeRegressor()
TAS_DT_reg.fit(alex_df, TAS_df)
print("Cross validation scores:")
TAS_DTscores = cross_val_score(TAS_DT_reg, alex_df, TAS_df,scoring="neg_mean_squared_error", cv=10)
TAS_DTresults = np.sqrt(-TAS_DTscores)
print("Decision Tree RMSE:",TAS_DTresults.mean(),"\nDecision Tree Standard deviation", TAS_DTresults.std())

TAS_RF_reg = RandomForestRegressor()
TAS_RF_reg.fit(alex_df, TAS_df)
TAS_RFscores = cross_val_score(TAS_RF_reg, alex_df, TAS_df,scoring="neg_mean_squared_error", cv=10)
TAS_RFresults = np.sqrt(-TAS_RFscores)
print("Random Forest RMSE is:",TAS_RFresults.mean(),"\nRandom Forest standard deviation:", TAS_RFresults.std())
#print(TAS_RF_reg.get_params())

param_grid = [ {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
                {'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4],} ]
TAS_RF_reg2 = RandomForestRegressor()
grid_search = GridSearchCV(TAS_RF_reg2, param_grid, cv=5, scoring="neg_mean_squared_error", \
  return_train_score=True)
grid_search.fit(alex_df, TAS_df)
print(grid_search.best_params_)
print(grid_search.best_estimator_)
cvres = grid_search.cv_results_
#for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
#    print(np.sqrt(-mean_score), params)
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)

# summarize feature importance
for i,v in enumerate(feature_importances):
	print('Feature: %0d, Score: %.5f' % (i,v))
print(alex_df.columns)


print(alex_df.columns)
dass_linreg = LinearRegression()
dass_linreg.fit(alex_df, DASS_df)
dass_linreg_pred=dass_linreg.predict(alex_df)
dass_linreg_rmse=np.sqrt(mean_squared_error(dass_linreg_pred, DASS_df))
print("RMSE for linear regression model:",dass_linreg_rmse)

DASS_DT_reg = DecisionTreeRegressor()
DASS_DT_reg.fit(alex_df, DASS_df)
print("Cross validation scores:")
DASS_DTscores = cross_val_score(DASS_DT_reg, alex_df, DASS_df,scoring="neg_mean_squared_error", cv=10)
DASS_DTresults = np.sqrt(-DASS_DTscores)
print("Decision Tree RMSE:",DASS_DTresults.mean(),"\nDecision Tree Standard deviation", DASS_DTresults.std())

DASS_RF_reg = RandomForestRegressor()
DASS_RF_reg.fit(alex_df, DASS_df)
DASS_RFscores = cross_val_score(DASS_RF_reg, alex_df, DASS_df,scoring="neg_mean_squared_error", cv=10)
DASS_RFresults = np.sqrt(-DASS_RFscores)
print("Random Forest RMSE is:",DASS_RFresults.mean(),"\nRandom Forest standard deviation:", DASS_RFresults.std())

#print(SAS_RF_reg.get_params())
param_grid = [ {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
                {'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4],} ]
DASS_RF_reg2 = RandomForestRegressor()
grid_search = GridSearchCV(DASS_RF_reg2, param_grid, cv=5, scoring="neg_mean_squared_error", \
  return_train_score=True)
grid_search.fit(alex_df, DASS_df)
print(grid_search.best_params_)
print(grid_search.best_estimator_)
cvres = grid_search.cv_results_
#for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
#    print(np.sqrt(-mean_score), params)
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)
for i,v in enumerate(feature_importances):
	print('Feature: %0d, Score: %.5f' % (i,v))
print(alex_df.columns)


#SAS LinRegs
alex_df.drop(columns =['hour_usage','platform_count'],inplace=True)
print(alex_df.columns)
sas_linreg = LinearRegression()
sas_linreg.fit(alex_df, SAS_df)
sas_linreg_pred=sas_linreg.predict(alex_df)
sas_linreg_rmse=np.sqrt(mean_squared_error(sas_linreg_pred, SAS_df))
print("RMSE for linear regression model:",sas_linreg_rmse)

SAS_DT_reg = DecisionTreeRegressor()
SAS_DT_reg.fit(alex_df, SAS_df)
print("Cross validation scores:")
SAS_DTscores = cross_val_score(SAS_DT_reg, alex_df, SAS_df,scoring="neg_mean_squared_error", cv=10)
SAS_DTresults = np.sqrt(-SAS_DTscores)
print("Decision Tree RMSE:",SAS_DTresults.mean(),"\nDecision Tree Standard deviation", SAS_DTresults.std())

SAS_RF_reg = RandomForestRegressor()
SAS_RF_reg.fit(alex_df, SAS_df)
SAS_RFscores = cross_val_score(SAS_RF_reg, alex_df, SAS_df,scoring="neg_mean_squared_error", cv=10)
SAS_RFresults = np.sqrt(-SAS_RFscores)
print("Random Forest RMSE is:",SAS_RFresults.mean(),"\nRandom Forest standard deviation:", SAS_RFresults.std())

#print(SAS_RF_reg.get_params())
param_grid = [ {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
                {'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4],} ]
SAS_RF_reg2 = RandomForestRegressor()
grid_search = GridSearchCV(SAS_RF_reg2, param_grid, cv=5, scoring="neg_mean_squared_error", \
  return_train_score=True)
grid_search.fit(alex_df, SAS_df)
print(grid_search.best_params_)
print(grid_search.best_estimator_)
cvres = grid_search.cv_results_
#for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
#    print(np.sqrt(-mean_score), params)
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)
for i,v in enumerate(feature_importances):
	print('Feature: %0d, Score: %.5f' % (i,v))
print(alex_df.columns)
