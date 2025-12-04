#from typing import Any
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib
from numpy.typing import NDArray

#import plotly.tools as tls
#import plotly.io as pio
#import plotly.graph_objects as go

#import mpld3
#from mpld3 import plugins

matplotlib.use('TkAgg')

# 라이다 점들을 생성하는 함수
def generate_lidar_points(distance) -> tuple[NDArray, NDArray, int]:
    if distance > 300.0:
        return (NDArray(), NDArray(), 0)

    v_list_per_meter = np.array([
        0.354118573,
        0.328783272,
        0.303823231,
        0.279205018,
        0.254896805,
        0.230868192,
        0.207090045,
        0.183534345,
        0.160174049,
        0.136982961,
        0.113935609,
        0.09100713,
        0.068173161,
        0.045409735,
        0.022693175,
        0,
        -0.022693175,
        -0.045409735,
        -0.068173161,
        -0.09100713,
        -0.113935609,
        -0.136982961,
        -0.160174049,
        -0.183534345,
        -0.207090045,
        -0.230868192,
        -0.254896805,
        -0.279205018,
        -0.303823231,
        -0.328783272,
        -0.354118573,
        -0.379864366])
    h_list_per_meter = []
    for i in range(-1000, 1000):
        h_list_per_meter.append(np.tan((0.09 * i) * np.pi / 180.0))

    #h_list_per_meter = np.tan(h_list_per_meter)
    
    (h_grid, v_grid) = np.meshgrid(h_list_per_meter, v_list_per_meter)
    # 조건 계산
    condition = distance * np.sqrt(1 + v_grid**2 + h_grid**2) < 300

    # 조건을 만족하는 값들 필터링
    h_grid = distance * h_grid[condition]
    v_grid = distance * v_grid[condition]
    return (h_grid, v_grid, np.sum(condition))

# 초기 값 설정
working_distance = 1
effective_distance = 300
# 플롯 설정
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.08, right=0.98, top=1., bottom=0., wspace=0.1, hspace=0.1)

# 무작위 점들을 생성하고 플롯
x, y, remain = generate_lidar_points(working_distance)
scatter = ax.scatter(x, y, s=0.05)

# 슬라이더 설정
ax_wd = plt.axes([0.16, 0.1, 0.78, 0.03])
wd_slider = Slider(ax_wd, 'working distance', 0.1, 300.0, valinit = working_distance)

text = ax.text(0.02, 0.95, f'valid points : {remain}', transform=ax.transAxes, va='top')
#ax_ed = plt.axes([0.16, 0.05, 0.78, 0.03])
#ed_slider = Slider(ax_ed, 'effective distance', 0.1, 300.0, valinit = effective_distance)

# 슬라이더 변화에 따른 업데이트 함수
def update(val):
    x, y, remain = generate_lidar_points(val)

    x_length = 0 if x.size == 0 else np.max(x) - np.min(x)
    y_length = 0 if y.size == 0 else np.max(y) - np.min(y)
    text.set_text(f'valid points : {remain}\ndimension(x, y) : ({x_length}, {y_length})\narea : {x_length * y_length} m^2')

    scatter.set_offsets(np.c_[x, y])
    fig.canvas.draw_idle()

#def update_color(val):
#    condition2 = np.sqrt(wd_slider.val**2 + x**2 + y**2) <= val
#    colors = np.where(condition2, 'blue', 'red')
#    scatter.set_color(colors)

# 슬라이더 이벤트 연결
wd_slider.on_changed(update)
#ed_slider.on_changed(update_color)

# 축 설정 및 그래프 보여주기
ax.set_xlim(-300, 300)
ax.set_xlabel('horizontal (x)')
ax.set_ylim(-120, 120)
ax.set_ylabel('vertical (y)')

ax.set_aspect('equal')

# plotly_fig = tls.mpl_to_plotly(plt.gcf())
# # Plotly 그래프 생성
# #plotly_fig = go.Figure()

# # 초기 점 추가
# plotly_fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=2), name='Lidar Points'))

# # 슬라이더 설정
# sliders = [{
#     "active": 0,
#     "currentvalue": {"prefix": "Working Distance: "},
#     "pad": {"b": 10},
#     "steps": [
#         {
#             "args": [
#                 ["x", "y"],  # 업데이트할 축 데이터
#                 {
#                     "x": [generate_lidar_points(i)[0]],
#                     "y": [generate_lidar_points(i)[1]],
#                 }
#             ],
#             "label": f"{i:.1f}",
#             "method": "restyle",  # 그래프 데이터를 실시간으로 변경
#         }
#         for i in np.arange(0.1, 300.1, 1.0)
#     ]
# }]

# # 슬라이더에 따른 프레임 정의 (working_distance 값 변경 시 점 데이터 갱신)
# frames = [
#     go.Frame(
#         data=[go.Scatter(x=generate_lidar_points(i)[0], y=generate_lidar_points(i)[1], mode='markers', marker=dict(size=2))],
#         name=f"{i:.1f}",
#     )
#     for i in np.arange(0.1, 300.1, 1.0)
# ]

# # 프레임과 슬라이더 추가
# plotly_fig.frames = frames
# plotly_fig.update_layout(
#     sliders=sliders,
#     title="Lidar Point Cloud",
#     xaxis_title="X",
#     yaxis_title="Y",
#     height=600, width=1000
# )
# pio.write_html(plotly_fig, file="interactive_matplotlib_plot.html", auto_open=True)

plt.show()
input("Press Enter to close the plot...")