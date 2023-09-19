import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# fig = go.Figure()
# go.marker.colorbar()

# fig.update_traces(
#     cmin=cmin,
#     cmax=cmax,
#     colorbar_x=1 + 0.2 * bar_count,
#     colorbar_title=cbar.title,
#     colorbar_title_font_size=40,
#     colorbar_title_side="right",
# )

# fig = make_subplots(rows=1, cols=2, shared_yaxes=True)

# # create traces using plotly.graph_objects
# b1 = go.Scatter(
#     x=[1, 2, 3],
#     y=[1, 2, 3],
#     mode="markers",
#     marker={"color": [35, 67, 89], "colorscale": "agsunset", "showscale": True},
# )
# b2 = go.Scatter(
#     x=[5, 2, 6], y=[9, 2, 4], mode="markers", marker={"color": [22, 5, 111]}
# )

# # add traces to figure
# fig.add_traces(b1, rows=1, cols=1)
# fig.add_traces(b2, rows=1, cols=2)

# fig.update_traces(
#     {
#         "marker": {
#             "colorbar": {
#                 "orientation": "h",
#                 "title": "Layer with best correlation (relative %)",
#                 "y": -1.0,
#             },
#             "colorscale": "agsunset",
#         }
#     }
# )


fig = go.Figure()

# Create list from 0 to 39 to use as x, y, and color
values = list(range(1))

fig.add_trace(
    go.Scatter(
        x=values,
        y=values,
        marker=dict(
            size=16,
            cmax=1,
            cmin=0,
            color=values,
            colorbar=dict(
                title="Layer with best correlation (relative %)",
                orientation="h",
                y=-1.0,
            ),
            colorscale="agsunset",
        ),
        mode="markers",
    )
)

fig.show()


fig.write_image("bar.svg", scale=6, width=1200, height=500)
