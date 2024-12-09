import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State, ALL
import pandas as pd
import json
import os
import dash_leaflet as dl
import plotly.graph_objs as go

def get_all_trajectories():
    """Get all trajectories for initial display."""
    # data_path = '../data/AIS'
    data_path = 'D:/CodeProject/python/TERI/data/AIS'
    data_name ='AIS_2023_1112'

    folder = os.path.join(data_path, data_name)

    df = pd.read_csv(os.path.join(folder, 'RMSE_point.csv'))

    return df

# Initialize the Dash app
app = dash.Dash(__name__)

# 预加载所有轨迹数据到缓存
initial_data = get_all_trajectories()
total_rmse = initial_data.iloc[0]['RMSE']
total_rmse_base = initial_data.iloc[0]['RMSE_base']

# 起始点的中心和缩放
initial_center = [initial_data['LAT_Label'].iloc[0], initial_data['LON_Label'].iloc[0]]
initial_zoom =14

# 按 mmsi 列进行分组
grouped = initial_data.groupby('MMSI')

mmsi_options = initial_data['MMSI'].unique()

trajectory_points = [
    dl.CircleMarker(
        id={'index': idx, 'type': 'Point_Pred', 'mmsi': str(group['MMSI'].iloc[0])},  # 动态设置 id
        center=[lat, lon],
        radius=5,  # 点的半径
        color='red',  # 预测点颜色
        weight=0,  # 点的边框粗细
        opacity=0.8,
        fillColor='red',  # 填充颜色
        fillOpacity=1,
    )
    for id, group in grouped
    for idx, (lat, lon) in enumerate(zip(group['LAT_Pred'], group['LON_Pred']))
    if group['Pred'].iloc[idx] == 1
] + [
    dl.CircleMarker(
        id={'index': idx, 'type': 'Point', 'mmsi': str(group['MMSI'].iloc[0])},  # 动态设置 id
        center=[lat, lon],
        radius=3,  # 点的半径
        color='black',  # 点的颜色
        weight=0,  # 点的边框粗细
        opacity=1,
        fillColor='black',  # 填充颜色
        fillOpacity=1,
    )
    for id, group in grouped
    for idx, (lat, lon) in enumerate(zip(group['LAT_Label'], group['LON_Label']))
    if group['Pred'].iloc[idx] == 0
]

trajectory_lines = [
   dl.Polyline(
        id={'type': 'Path_Pred', 'mmsi': str(group['MMSI'].iloc[0])},  # 动态设置 id
        positions=list(zip(group['LAT_Pred'], group['LON_Pred'])),
        color="red",
        weight=2,
        opacity=1,
        n_clicks=0  # 允许点击
   )
   for id, group in grouped
]


app.layout = html.Div([
    dcc.Store(id='overall-flag', data=True),  # Store for caching fetched data
    dcc.Store(id='cached-data'),  # Store for caching fetched data
    dcc.Store(id='mmsi-map', data=mmsi_options),  # 当前地图上展示的轨迹mmsi
    html.Header([
        html.H2([html.Span("STAR: Spatio-Temporal Trajectory Recovery for Sparse and Uncertain Marine Trajectories")], className='title'),
    ], style={
        # 'backgroundColor': 'black',  # 顶部背景颜色
        'color': 'black',  # 字体颜色
        'padding': '20px',
        'textAlign': 'center',
        'fontWeight': 'bold',
        'borderRadius': '8px 8px 0 0'  # 圆角顶部
    }),
    # Flex container
    html.Div([
        # Left column for input elements
        html.Div([
            html.P('Select trajectory and click Search to get RMSE.', className='subtitle'),
            html.Label('Selected Trajectory', htmlFor='trajectory-dropdown'),
            dcc.Dropdown(
                id='trajectory-dropdown',
                options=mmsi_options,
                value=mmsi_options[0] if len(mmsi_options) > 0 else None,
                searchable=True,
                placeholder='Select MMSI',
                className='input',
            ),
            html.Button(
                'Reset',
                id='reset-val',
                n_clicks=0,
                className='reset',
                style={'margin-bottom': '20px',  # 设置底部外边距
                        'width': '48%',
                        'padding': '10px',
                        'backgroundColor': '#ffcc00',
                        'border': 'none',
                        'color': 'white',
                        'fontSize': '1em',
                        'borderRadius': '5px',
                        'cursor': 'pointer',
                }
            ),
            html.Button(
                'Search',
                id='search-val',
                n_clicks=0,
                className='search',
                style={'margin-bottom': '20px',  # 设置底部外边距
                        'width': '48%',
                        'padding': '10px',
                        'backgroundColor': '#4CAF50',
                        'border': 'none',
                        'color': 'white',
                        'fontSize': '1em',
                        'borderRadius': '5px',
                        'cursor': 'pointer',
                        'marginLeft': '4%'  # 调整按钮之间的间距
                }
            ),
            # 柱状图
            dcc.Graph(
                id="bar-chart",
                style={'width': '100%', 'height': '500px', 'display': 'inline-block'}
            ),
            html.Label('RMSE: the root mean square error',
                       style={'fontWeight': 'normal'}),
            html.Label('[1]Chen Y, Cong G, Anda C. Teri: An effective framework for trajectory recovery with irregular time intervals[J]. Proceedings of the VLDB Endowment, 2023, 17(3): 414-426.',
                       style={'fontWeight': 'normal'}),
            html.Div(id="performance-info"),
        ], className='form',
            style={
                'backgroundColor': 'transparent',
                'border': 'none',
                'boxShadow': 'none'
            }
        ),


        # Right column for the map
        html.Div([
            dl.Map(
                id="map",
                center=initial_center,
                zoom=initial_zoom,
                children=[
                    dl.TileLayer(),  # 地图瓦片
                    dl.LayerGroup(
                        id="map-display",
                        children=trajectory_lines + trajectory_points
                    ),  # 轨迹图层
                ],
                style={'width': '100%', 'height': '800px', 'position': 'relative'}
            ),

            # 创建显示轨迹颜色标识的部分
            html.Div(id='trajectory-legend', children=[
                html.Div(children=[
                    'Original trajectories:',
                     html.Div(style={
                         'width': '3px',  # 线条宽度
                         'height': '3px',  # 线条高度
                         'backgroundColor': 'black',  # 线条颜色
                         'margin-left': '33px',  # 线条与文字之间的间距
                         # 'display': 'inline-block'  # 使线条与文本在同一行
                     }),
                    html.Div(style={
                         'width': '3px',  # 线条宽度
                         'height': '3px',  # 线条高度
                         'backgroundColor': 'black',  # 线条颜色
                         'margin-left': '5px',  # 线条与文字之间的间距
                         # 'display': 'inline-block'  # 使线条与文本在同一行
                     }),
                    html.Div(style={
                         'width': '3px',  # 线条宽度
                         'height': '3px',  # 线条高度
                         'backgroundColor': 'black',  # 线条颜色
                         'margin-left': '5px',  # 线条与文字之间的间距
                         # 'display': 'inline-block'  # 使线条与文本在同一行
                     }),
                    html.Div(style={
                         'width': '3px',  # 线条宽度
                         'height': '3px',  # 线条高度
                         'backgroundColor': 'black',  # 线条颜色
                         'margin-left': '5px',  # 线条与文字之间的间距
                         # 'display': 'inline-block'  # 使线条与文本在同一行
                     }),
                    html.Div(style={
                         'width': '3px',  # 线条宽度
                         'height': '3px',  # 线条高度
                         'backgroundColor': 'black',  # 线条颜色
                         'margin-left': '5px',  # 线条与文字之间的间距
                         # 'display': 'inline-block'  # 使线条与文本在同一行
                     })
                ],
                style={
                    'color': 'black',
                    'fontSize': '1em',
                    'padding': '5px',
                    'border': 'none',
                    'borderRadius': '5px',
                    'backgroundColor': 'white',
                    'marginBottom': '10px',
                    'display': 'flex',  # 使用 flexbox 布局
                    'alignItems': 'center'  # 垂直居中对齐
                }),
                html.Div(children=[
                    'Recovered trajectories:',
                    html.Div(style={
                         'width': '5px',  # 线条宽度
                         'height': '2px',  # 线条高度
                         'backgroundColor': 'red',  # 线条颜色
                         'margin-left': '10px',  # 线条与文字之间的间距
                         # 'display': 'inline-block'  # 使线条与文本在同一行
                     }),
                    html.Div(style={
                         'width': '3px',  # 线条宽度
                         'height': '3px',  # 线条高度
                         'backgroundColor': 'red',  # 线条颜色
                         'margin-left': '0px',  # 线条与文字之间的间距o
                         # 'display': 'inline-block'  # 使线条与文本在同一行
                     }),
                    html.Div(style={
                         'width': '5px',  # 线条宽度
                         'height': '2px',  # 线条高度
                         'backgroundColor': 'red',  # 线条颜色
                         'margin-left': '0px',  # 线条与文字之间的间距
                         # 'display': 'inline-block'  # 使线条与文本在同一行
                     }),
                    html.Div(style={
                         'width': '3px',  # 线条宽度
                         'height': '3px',  # 线条高度
                         'backgroundColor': 'red',  # 线条颜色
                         'margin-left': '0px',  # 线条与文字之间的间距
                         # 'display': 'inline-block'  # 使线条与文本在同一行
                     }),
                    html.Div(style={
                         'width': '5px',  # 线条宽度
                         'height': '2px',  # 线条高度
                         'backgroundColor': 'red',  # 线条颜色
                         'margin-left': '0px',  # 线条与文字之间的间距
                         # 'display': 'inline-block'  # 使线条与文本在同一行
                     }),
                    html.Div(style={
                         'width': '3px',  # 线条宽度
                         'height': '3px',  # 线条高度
                         'backgroundColor': 'red',  # 线条颜色
                         'margin-left': '0px',  # 线条与文字之间的间距
                         # 'display': 'inline-block'  # 使线条与文本在同一行
                     }),
                    html.Div(style={
                         'width': '5px',  # 线条宽度
                         'height': '2px',  # 线条高度
                         'backgroundColor': 'red',  # 线条颜色
                         'margin-left': '0px',  # 线条与文字之间的间距
                         # 'display': 'inline-block'  # 使线条与文本在同一行
                     }),
                    html.Div(style={
                         'width': '3px',  # 线条宽度
                         'height': '3px',  # 线条高度
                         'backgroundColor': 'red',  # 线条颜色
                         'margin-left': '0px',  # 线条与文字之间的间距
                         # 'display': 'inline-block'  # 使线条与文本在同一行
                     }),
                    html.Div(style={
                         'width': '5px',  # 线条宽度
                         'height': '2px',  # 线条高度
                         'backgroundColor': 'red',  # 线条颜色
                         'margin-left': '0px',  # 线条与文字之间的间距
                         # 'display': 'inline-block'  # 使线条与文本在同一行
                     })
                ],
                style={
                    'color': 'red',
                    'fontSize': '1em',
                    'padding': '5px',
                    'border': 'none',
                    'borderRadius': '5px',
                    'backgroundColor': 'white',
                    'marginBottom': '10px',
                    'display': 'flex',  # 使用 flexbox 布局
                    'alignItems': 'center'  # 垂直居中对齐
                })
            ], style={
                'position': 'absolute',  # 定位为绝对位置
                'bottom': '20px',
                'left': '20px',
                'backgroundColor': 'white',
                'padding': '10px',
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
                'borderRadius': '8px',
                'zIndex': 1000  # 确保它显示在地图上方
            })

        ], className='graph', style={'position': 'relative'})
    ], className='row'),
])

def prepare_performance_info(df):
    data_points = len(df)

    return [
        html.Div(f"Data points: {data_points}")
    ]

@app.callback(
    [Output('cached-data', 'data'),
     Output('overall-flag', 'data'),
     Output("bar-chart", "figure"),
     Output('trajectory-dropdown', 'value')],
    [Input({'type': 'Path', 'mmsi': ALL}, 'n_clicks'),  # 动态监听所有轨迹
     Input({'type': 'Path_Pred', 'mmsi': ALL}, 'n_clicks'),
     Input({'type': 'Point', 'mmsi': ALL, 'index': ALL}, 'n_clicks'),  # 动态监听所有轨迹点
     Input({'type': 'Point_Pred', 'mmsi': ALL, 'index': ALL}, 'n_clicks'),
     Input('reset-val', 'n_clicks'),
     Input('search-val', 'n_clicks')],
    [State('trajectory-dropdown', 'value')]
)
def display_rmse_on_click(clickMap, clickMap_pred, clickPoint, clickPoint_pred, reset, clickmmsi, mmsi):
    # 获取当前回调触发的事件
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    if triggered_id != 'reset-val':
        # 如果点击的不是重置
        if triggered_id == 'search-val':
            # 数据请求逻辑
            data = initial_data[initial_data['MMSI'] == mmsi]
            print(f"Requests data length: {len(data)}")

            data_rmse = data.loc[:, ['single_RMSE', 'RMSE', 'single_RMSE_base', 'RMSE_base']]
            selected_rmse = data_rmse.iloc[0]['single_RMSE'] if data_rmse.iloc[0]['single_RMSE'] != -1 else 'N/A'
            selected_rmse_base = data_rmse.iloc[0]['single_RMSE_base'] if data_rmse.iloc[0]['single_RMSE_base'] != -1 else 'N/A'

            fig = go.Figure(data=[
                go.Bar(
                    x=["Selected(Ours)", "Selected(SOTA[1])", "Overall(Ours)", "Overall(SOTA)"],
                    y=[selected_rmse, selected_rmse_base, total_rmse, total_rmse_base],
                    marker=dict(color='skyblue')
                )
            ])
            fig.update_layout(
                title="RMSE Comparison ",
                xaxis_title="MMSI",
                yaxis_title="RMSE",
                template="plotly_white"
            )

            return {
                'data': json.loads(data.to_json()),
                'cache_key': f"{mmsi}"
            }, False, fig, mmsi

        if triggered_id is not None:
            clicked_mmsi = int(triggered_id.split("\"mmsi\":")[1].split("\"")[1])  # 根据你 ID 的命名格式提取 mmsi

            data = initial_data[initial_data['MMSI'] == clicked_mmsi]
            print(f"Requests data length: {len(data)}")

            # 获取对应 mmsi 的 RMSE
            data_rmse = data.loc[:, ['single_RMSE', 'RMSE', 'single_RMSE_base', 'RMSE_base']]
            selected_rmse = data_rmse.iloc[0]['single_RMSE'] if data_rmse.iloc[0]['single_RMSE'] != -1 else 'N/A'
            selected_rmse_base = data_rmse.iloc[0]['single_RMSE_base'] if data_rmse.iloc[0]['single_RMSE_base'] != -1 else 'N/A'

            fig = go.Figure(data=[
                go.Bar(
                    x=["Selected(Ours)", "Selected(SOTA[1])", "Overall(Ours)", "Overall(SOTA)"],
                    y=[selected_rmse, selected_rmse_base, total_rmse, total_rmse_base],
                    marker=dict(color='skyblue')
                )
            ])
            fig.update_layout(
                title="RMSE Comparison Chart",
                xaxis_title="MMSI",
                yaxis_title="RMSE",
                template="plotly_white"
            )

            return {
                'data': json.loads(data.to_json()),
                'cache_key': f"{clicked_mmsi}"
            }, False, fig, clicked_mmsi

    fig = go.Figure(data=[
        go.Bar(
            x=["Overall(Ours)", "Overall(SOTA[1])"],
            y=[total_rmse, total_rmse_base],
            marker=dict(color='skyblue')
        )
    ])
    fig.update_layout(
        title="RMSE Comparison Chart",
        xaxis_title="MMSI",
        yaxis_title="RMSE",
        template="plotly_white",
        bargap=0.6,  # 设置柱子之间的间隔为 0
    )

    # 如果未点击任何内容或点击重置，返回默认值
    return {}, True, fig, 0

@app.callback(
    [Output('map-display', 'children'),
     Output("performance-info", "children"),
     Output('map', 'center'),
     Output('map', 'zoom')],
    [Input('cached-data', 'data'),
     Input('overall-flag', 'data')],
    [State('map', 'center'),
     State('map', 'zoom')]
)
def update_map(cached_data, overall_flag, current_center, current_zoom):
    if overall_flag is True:
        return trajectory_lines + trajectory_points, '', current_center, current_zoom
    else:
        if not cached_data or 'data' not in cached_data or not cached_data['data']:
            return trajectory_lines + trajectory_points, '', current_center, current_zoom

        print(f"Cache Key: {cached_data.get('cache_key')}")
        df = pd.DataFrame(cached_data['data'])
        mmsi = df['MMSI'].iloc[0]

        performance_info = prepare_performance_info(df)

        selected_points = [
            dl.CircleMarker(
                id={'index': idx, 'type': 'Point_Pred', 'mmsi': str(mmsi)},
                # 动态设置 id
                center=[lat, lon],
                radius=5,  # 点的半径
                color='red',  # 预测点颜色
                weight=0,  # 点的边框粗细
                opacity=0.8,
                fillColor='red',  # 填充颜色
                fillOpacity=1,
            )
            for idx, (lat, lon) in enumerate(zip(df['LAT_Pred'], df['LON_Pred']))
            if df['Pred'].iloc[idx] == 1
        ] + [
            dl.CircleMarker(
                id={'index': idx, 'type': 'Point', 'mmsi': str(mmsi)},  # 动态设置 id
                center=[lat, lon],
                radius=3,  # 点的半径
                color='black',  # 点的颜色
                weight=0,  # 点的边框粗细
                opacity=1,
                fillColor='black',  # 填充颜色
                fillOpacity=1,
            )
            for idx, (lat, lon) in enumerate(zip(df['LAT_Label'], df['LON_Label']))
            if df['Pred'].iloc[idx] == 0
        ]

        selected_trajectory = [
            dl.Polyline(
               id={'type': 'Path_Pred', 'mmsi': str(mmsi)},  # 动态设置 id
               positions=list(zip(df['LAT_Pred'], df['LON_Pred'])),
               color="red",
               weight=1,
               opacity=1,
               n_clicks=0  # 允许点击
            )
        # ] + [
        #     dl.Polyline(
        #        id={'type': 'Path', 'mmsi': str(mmsi)},  # 动态设置 id
        #        positions=list(zip(df['LAT_Label'], df['LON_Label'])),
        #        color="black",  # 预测轨迹颜色
        #        weight=1,
        #        opacity=0.8,
        #        n_clicks=0  # 允许点击
        #     )
        ]
        df_center = int(len(df)/2)

        new_center = [df['LAT_Label'].iloc[df_center], df['LON_Label'].iloc[df_center]]
        new_zoom=14
        return selected_trajectory + selected_points, html.Div(performance_info, className='info'), new_center, current_zoom




if __name__ == '__main__':
    app.run_server(debug=True)
