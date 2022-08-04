import pandas as pd
import sys

output = []
sheet_mapper = {"KAI-ADAPTED 13B": "13B", "KAI-ADAPTED 6B": "6B", 'KAI-CUSTOM': 'Custom'}

for file in ['KoboldAI Settings (6B).xlsx', 'KoboldAI Settings (13B).xlsx', 'KoboldAI Settings (Custom).xlsx', 'KoboldAI Settings (Original).xlsx']:
    presets = pd.read_excel("preset Files/{}".format(file), None)
    for sheet in presets:
        df = presets[sheet]
        if sheet in sheet_mapper:
            sheet = sheet_mapper[sheet]
        df = df.dropna(axis=1, how='all')
        df = df.rename(columns={"Unnamed: 0": "setting"})
        df = pd.melt(df, id_vars=['setting'])
        df = df.rename(columns={"variable": "preset"})
        df['fix'] = df['value'].str.replace(" (KAI)", "", regex=False)
        df.loc[~df['fix'].isnull(), 'value'] = df['fix']
        df = df.drop(columns=['fix'])
        df.loc[df['setting']=='Samplers Order', 'value'] = df['value'].str.replace("Temp", "5", regex=False)
        df.loc[df['setting']=='Samplers Order', 'value'] = df['value'].str.replace("K", "0", regex=False)
        df.loc[df['setting']=='Samplers Order', 'value'] = df['value'].str.replace("TFS", "3", regex=False)
        df.loc[df['setting']=='Samplers Order', 'value'] = df['value'].str.replace("A", "1", regex=False)
        df.loc[df['setting']=='Samplers Order', 'value'] = df['value'].str.replace("Typ", "4", regex=False)
        df.loc[df['setting']=='Samplers Order', 'value'] = df['value'].str.replace("P", "2", regex=False)


        settings_mapper = {'Temperature': 'temp', 'Output Length': 'genamt', 'Repetition Penalty': 'rep_pen',
                           'Top P': 'top_p', 'Top K': 'top_k', 'Tail-Free': 'tfs', 'Repetition Penalty Range': 'rep_pen_range',
                           'Repetition Penalty Slope': 'rep_pen_slope', 'Typical': 'typical', 'Top A': 'top_a',
                           'Samplers Order': 'sampler_order', 'Description of settings from the author': 'description', 
                           'Author': 'Author', 'Model Type': 'Model Type', 
                           'Description of settings from NovelAI': 'description', 'Model Size': "Model Size"
                          }
        df['setting'] = df['setting'].map(settings_mapper)
        
        try:
            df = df.pivot(index='preset', columns='setting', values='value')
        except:
            print(file)
            display(df)
            raise
        
        df['Model Type'] = df['Model Type'].str.replace(", ", ",").str.split(",")
        
        df.loc[:, 'Model Category'] = sheet
        
        output.append(df)

        #output[sheet] = df.to_json(orient="index")

df = pd.concat(output)
df = df.reset_index(drop=False)
df['uid'] = df.index
df = df.explode("Model Type")
df['description'] = df['description'].str.strip()

        
with open("official.presets", "w") as f:
    f.write(df.reset_index(drop=True).to_json(orient='records'))