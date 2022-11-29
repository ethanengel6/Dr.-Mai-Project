import pandas as pd

alex_df=pd.read_csv("ASD_responses.csv")

TAS_list=['(ALX)_1','(ALX)_2','(ALX)_3','(ALX)_4','(ALX)_5','(ALX)_6','(ALX)_7','(ALX)_8',
'(ALX)_9','(ALX)_10','(ALX)_11','(ALX)_12','(ALX)_13','(ALX)_14','(ALX)_15','(ALX)_16','(ALX)_17',
'(ALX)_18','(ALX)_19','(ALX)_20']

DASS_list=['(DASS)_1', '(DASS)_2', '(DASS)_3', '(DASS)_4', '(DASS)_5',
       '(DASS)_6', '(DASS)_7', '(DASS)_8', '(DASS)_9', '(DASS)_10',
       '(DASS)_11', '(DASS)_12', '(DASS)_13', '(DASS)_14', '(DASS)_15',
       '(DASS)_16', '(DASS)_17', '(DASS)_18', '(DASS)_19', '(DASS)_20',
       '(DASS)_21']

alex_df['TAS_total'] = alex_df[TAS_list].sum(axis=1)
alex_df['DASS_total']=alex_df[DASS_list].sum(axis=1)

TAS_move = alex_df.pop('TAS_total')
alex_df.insert(37,'TAS_total', TAS_move)

print(len(alex_df["TAS_total"]),alex_df["TAS_total"].mean(),alex_df["TAS_total"].std())
