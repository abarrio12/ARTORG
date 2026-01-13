import pandas as pd

url = "https://unibe365-my.sharepoint.com/:x:/r/personal/gaia_stievano_unibe_ch/_layouts/15/Doc.aspx?sourcedoc=%7B8D69FBBC-1D47-4E00-A93E-CE6D81FC97A2%7D&file=edge_geometry_coordinates.csv&action=default&mobileredirect=true"

df = pd.read_csv(url, nrows=5)
print(df.columns)
print(df.head())
