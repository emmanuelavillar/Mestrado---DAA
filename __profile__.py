import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv("dados/training_data.csv", encoding="latin1")

profile = ProfileReport(df, title="Relatório", explorative=True)
profile.to_file("relatorio.html")
