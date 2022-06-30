import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("/Users/christopherhempel/Desktop/score_df.csv")

models = df["model"].unique()
dtypes = df["datatype"].unique()
ranks = df["rank"].unique()

for model in models:
    for dtype in dtypes:
        name = "{0}_{1}".format(model, dtype)
        modeldf= df[(df["model"]==model) & (df["datatype"]==dtype)]
        fig = px.line(modeldf, x="rank", y="test_score_mcc", color='seqtype', title=name, markers=True)
        fig.update_yaxes(range=[-1, 1])
        #fig.add_hline(y=0.2)
        fig.add_shape(type="line", x0=ranks[0], y0=0, x1=ranks[-1], y1=0, line_width=2, line_dash="dash")
        fig.show()



df_test = px.data.medals_wide(indexed=True)
fig = px.imshow(df_test)
fig.show()
