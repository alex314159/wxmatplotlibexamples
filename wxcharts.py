"""
Templates for charting

Written by Alexandre Almosni   alexandre.almosni@gmail.com
(C) 2018 Alexandre Almosni
Released under Apache 2.0 license. More info at http://www.apache.org/licenses/LICENSE-2.0

"""


import pandas
import wx
import wx.lib.colourdb
import wx.adv

import matplotlib
matplotlib.use("WxAgg")
import datetime
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib import rcParams
#rcParams['font.family'] = 'Helvetica'

# from cycler import cycler

investec_palette = {'Sea': '#326295',
                   'Forrest': '#74AA50',
                   'Cerise': '#CE0F69',
                   'Sunrise': '#DE7C00',
                   'Violet': '#93328E',
                   'Sky': '#4298B5',
                   'Teal': '#009681',
                   'Terracotta': '#A50034',
                   'Sea_m': '#ADC0D5',
                   'Forrest_m': '#C7DDB9',
                   'Cerise_m': '#EB9FC3',
                   'Sunrise_m': '#F2CB99',
                   'Violet_m': '#D4ADD2',
                   'Sky_m': '#B3D6E1',
                   'Teal_m': '#99D5CD',
                   'Terracotta_m': '#DB99AE'}

DEFAULT_FIG_SIZE = (4.0, 4.0)


class BaseChart:
    def __init__(self, titlelabel, xlabel, ylabel, labeltextsize, canvas_panel=None, PDF=False):
        self.titlelabel = titlelabel
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.labeltextsize = labeltextsize
        if PDF:
            self.fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
            plt.clf()
            self.ax = self.fig.add_axes((0.1, 0.1, 0.8, 0.8))
        else:
            self.fig = canvas_panel.figure
            self.ax = canvas_panel.ax
        # self.ax.set_prop_cycle(cycler('color', investec_palette.values()))
        # self.ax.set_prop_cycle(None)
        self.ax.set_prop_cycle('color', list(investec_palette.values()))
        self.ax.axis('auto')  # needed as is reset by pie chart
        self.ax.set_frame_on(True)  # needed as is reset by pie chart
        self.ax.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True)  # needed as is reset by stacked chart
        self.ax.set_title(titlelabel, fontsize=labeltextsize, pad=15)
        self.ax.set_xlabel(xlabel, fontsize=labeltextsize)
        self.ax.set_ylabel(ylabel, fontsize=labeltextsize)
        self.ax.tick_params(axis='both', which='major', labelsize=labeltextsize)

    def scatter(self, df, xc, yc, labels, xtickslabeldic=False, force_zero_axe=False):
        self.ax.scatter(df[xc], df[yc], marker='.')
        for label, x, y in zip(labels, df[xc], df[yc]):
            self.ax.annotate(label, xy=(x, y), xytext=(0, -10), textcoords='offset points', ha='center', va='center', size=(self.labeltextsize - 2))
        self.ax.grid(b=True, which='major', axis='y', linestyle='--')
        if xtickslabeldic:
            xvals = range(df[xc].min(), df[xc].max() + 1)
            self.ax.set_xticks(xvals)
            self.ax.set_xticklabels(list(map(lambda t: xtickslabeldic[t] if t in xtickslabeldic.keys() else '', xvals)), fontsize=8, rotation=0)
        if force_zero_axe:
            xmin, xmax = self.ax.get_xlim()
            ymin, ymax = self.ax.get_ylim()
            self.ax.set_xlim((max(0, xmin), xmax))
            self.ax.set_ylim((max(0, ymin), ymax))

    def line(self, df):
        df.plot(ax=self.ax)
        x = self.ax.get_xlim()
        self.ax.set_xlim(x[1], x[0])
        self.ax.grid(b=True, which='major', axis='both', linestyle='--')

    def histogram(self, df, bins=10):
        df.hist(ax=self.ax, figure=self.fig, bins=bins)
        self.ax.grid(b=True, which='major', axis='both', linestyle='--')

    def regression_scatter(self, df, prediction):
        self.ax.axis('equal')
        df.plot.scatter(ax=self.ax, x='Benchmark', y='Fund')
        self.ax.plot(df['Benchmark'].values, prediction, color=investec_palette['Forrest'])
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.grid(b=True, which='major', axis='both', linestyle='--')

    def single_stacked_bar(self, df, labels, index_name='volatility'):
        outdf = pandas.DataFrame(columns=labels, index=[index_name], data=pandas.np.reshape(df.values, (1, len(labels))))
        outdf.plot.bar(stacked=True, ax=self.ax, legend=False, color=investec_palette.values())
        self.ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        for (i, rec) in enumerate(self.ax.patches):
            self.ax.text(rec.get_x() + rec.get_width() / 2., rec.get_y() + rec.get_height() / 2., outdf.columns[i], ha="center", va="center", fontsize=8)

    def double_stacked_bar(self, df):
        df.plot.bar(stacked=True, ax=self.ax, legend=False, color=investec_palette.values())
        self.ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        self.ax.set_xticklabels(df.index, fontsize=8, rotation=0)
        self.ax.yaxis.set_tick_params(labelsize=8)
        for (i, rec) in enumerate(self.ax.patches):
            self.ax.text(rec.get_x() + rec.get_width() / 2., rec.get_y() + rec.get_height() / 2., df.columns[int(i / 2.)], ha="center", va="center", fontsize=8)

    def pie(self, data, labels):
        self.ax.axis('equal')
        self.ax.pie(data, labels=labels, autopct='%1.0f%%', textprops={'fontsize': self.labeltextsize})

    def bar(self, df, labels, fb):
        cols = df.columns
        df = df.rename(columns=dict(zip(cols, fb)))
        df.plot.bar(stacked=False, ax=self.ax, legend=True, rot=0, color=investec_palette.values())
        self.ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}%'))
        for p in self.ax.patches:
            w, h = p.get_width(), p.get_height()
            x, y = p.get_xy()
            self.ax.annotate('{:.0f}%'.format(h), (x + 0.5 * w, h + 0.5), horizontalalignment='center', fontsize=8)
        self.ax.set_xticklabels(labels, fontsize=8)
        self.ax.set_xlabel(self.xlabel)

    def bar_deviation(self, df, labels):
        df.plot.bar(stacked=False, ax=self.ax, legend=False, rot=0, color=investec_palette.values())
        self.ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}%'))
        self.ax.axhline(0, linewidth=0.5, color='black')
        for p in self.ax.patches:
            w, h = p.get_width(), p.get_height()
            x, y = p.get_xy()
            self.ax.annotate('{:.0f}%'.format(h), (x + 0.5 * w, h + 0.2), horizontalalignment='center', fontsize=8)
        self.ax.set_xticklabels(labels, fontsize=8, visible=True)
        self.ax.set_xlabel(self.xlabel)

    def simple_bar(self, df, labels, lower=False):
        df.plot.bar(stacked=False, ax=self.ax, legend=False, rot=0, color=investec_palette.values())
        self.ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
        try:
            hmax = float(df.abs().max().iloc[0])
        except AttributeError:
            hmax = float(df.abs().max())

        for p in self.ax.patches:
            w, h = p.get_width(), p.get_height()
            x, y = p.get_xy()
            if lower:
                self.ax.annotate('{:,.0f}'.format(h), (x + 0.5 * w, h - hmax / 20.), horizontalalignment='center', fontsize=8)
            else:
                self.ax.annotate('{:,.0f}'.format(h), (x + 0.5 * w, h + hmax / 20.), horizontalalignment='center', fontsize=8)
        self.ax.set_xticklabels(labels)

    def h_performance_bar(self, df, labels, inside=False):
        df.plot.barh(stacked=False, ax=self.ax, legend=False, rot=0, color=investec_palette['Sea'])
        self.ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}%'))
        xmin, xmax = self.ax.get_xlim()
        db = (xmax - xmin) / 100.
        self.ax.axvline(0, linewidth=0.5, color='black')
        for p in self.ax.patches:
            w, h = p.get_width(), p.get_height()
            x, y = p.get_xy()
            if inside:
                self.ax.annotate('{:,.1f}'.format(w), (x + 0.5 * w, h - db / 20.), verticalalignment='center', fontsize=8)
            else:
                if w >= 0:
                    # print(x,w,h,y)
                    self.ax.annotate('{:,.1f}%'.format(w), (x + w + 1.5 * db, h * 0.5 + y), verticalalignment='center', fontsize=8)
                else:
                    self.ax.annotate('{:,.1f}%'.format(w), (x + w - 4 * db, h * 0.5 + y), verticalalignment='center', fontsize=8)
        self.ax.yaxis.set_tick_params(labelsize=8)
        self.ax.xaxis.set_tick_params(labelsize=8)
        self.ax.set_xlabel('')
        self.ax.set_ylabel('')

    def h_momentum_bar(self, df, labels, inside=False):
        df.plot.barh(stacked=False, ax=self.ax, legend=False, rot=0, color=investec_palette['Sea'])
        self.ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}x'))
        xmin, xmax = self.ax.get_xlim()
        db = (xmax - xmin) / 100.
        self.ax.axvline(0, linewidth=0.5, color='black')
        for p in self.ax.patches:
            w, h = p.get_width(), p.get_height()
            x, y = p.get_xy()
            if inside:
                self.ax.annotate('{:,.1f}'.format(w), (x + 0.5 * w, h - db / 20.), verticalalignment='center', fontsize=8)
            else:
                if w >= 0:
                    # print(x,w,h,y)
                    self.ax.annotate('{:,.1f}x'.format(w), (x + w + 1.5 * db, h * 0.5 + y), verticalalignment='center', fontsize=8)
                else:
                    self.ax.annotate('{:,.1f}x'.format(w), (x + w - 4 * db, h * 0.5 + y), verticalalignment='center', fontsize=8)
        self.ax.yaxis.set_tick_params(labelsize=8)
        self.ax.xaxis.set_tick_params(labelsize=8)
        self.ax.set_xlabel('')
        self.ax.set_ylabel('')

    def h_range_bar(self, df, labels, inside=False):
        df.plot.barh(stacked=True, ax=self.ax, legend=False, rot=0, color=[investec_palette['Sea'], investec_palette['Forrest']])
        self.ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
        self.ax.axvline(0, linewidth=0.5, color='black')
        self.ax.yaxis.set_tick_params(labelsize=8)
        self.ax.xaxis.set_tick_params(labelsize=8)
        self.ax.set_xlabel('')
        self.ax.set_ylabel('')

    def scatter_z_circle(self, df, xc, yc, labels):
        self.ax.scatter(df[xc], df[yc], marker='.')
        for label, x, y in zip(labels, df[xc], df[yc]):
            self.ax.annotate(label, xy=(x, y), xytext=(0, -10), textcoords='offset points', ha='center', va='center', size=(self.labeltextsize - 2))
        xn = max(df[xc].abs().max(), 2.5)
        self.ax.set_xlim((-xn, xn))
        yn = max(df[yc].abs().max(), 2.5)
        self.ax.set_ylim((-yn, yn))
        self.fig.gca().set_aspect('equal', adjustable='box')  # plt.axis('equal')
        self.ax.grid(b=True, which='major', axis='y', linestyle='--')
        self.ax.grid(b=True, which='major', axis='x', linestyle='--')
        circle1 = plt.Circle((0, 0), 1, color=investec_palette['Forrest'], fill=False)
        circle2 = plt.Circle((0, 0), 2, color=investec_palette['Cerise'], fill=False)
        self.fig.gca().add_artist(circle1)
        self.fig.gca().add_artist(circle2)
        self.ax.set_xlabel(xc)
        self.ax.set_ylabel(yc)


class CanvasPanel(wx.Panel):

    def __init__(self, parent, figsize=DEFAULT_FIG_SIZE, toolbar=True, style=wx.NO_BORDER):
        wx.Panel.__init__(self, parent, style=style)
        self.toolbar = None
        self.SetBackgroundColour(wx.Colour("WHITE"))
        self.figure = Figure(figsize=figsize, dpi=100) #WE NEED TO INITIALIZE OTHERWISE IT CAN BREAK
        self.ax = self.figure.add_axes((0.1, .15, 0.8, .7))
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.EXPAND)
        if toolbar:
            self.add_toolbar()
        self.SetSizer(self.sizer)
        self.Fit()

    def add_toolbar(self):
        """Copied verbatim from embedding_wx2.py"""
        self.toolbar = NavigationToolbar2Wx(self.canvas)
        self.toolbar.Realize()
        tw, th = self.toolbar.GetSize()
        fw, fh = self.canvas.GetSize()
        self.toolbar.SetSize(wx.Size(fw, th))
        self.sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        self.toolbar.update()


class TestFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "wxcharts", size=(800, 600))
        sizer = wx.BoxSizer(wx.VERTICAL)
        btn = wx.Button(self, label="Change chart")
        btn.Bind(wx.EVT_BUTTON, self.on_change_chart)
        self.bottom_panel = CanvasPanel(self, None)
        sizer.Add(btn, 1, wx.EXPAND, 1)
        sizer.Add(self.bottom_panel, 5, wx.EXPAND, 1)
        self.SetSizer(sizer)
        self.pool = cycle(range(0,3))

    def on_change_chart(self, event):
        chart_counter = next(self.pool)
        if chart_counter == 0:
            df = pandas.DataFrame(index=range(0, 10), columns=['a'], data=list(range(0, 10)))
            self.bottom_panel.ax.clear()
            BaseChart('Line chart', 'x', 'y', 8, self.bottom_panel, False).line(df)
            self.bottom_panel.canvas.draw()
        elif chart_counter == 1:
            df = pandas.DataFrame(index=range(0, 10), columns=['a'], data=list(range(0, 10)))
            self.bottom_panel.ax.clear()
            BaseChart('Pie chart', 'x', 'y', 8, self.bottom_panel, False).pie(df, list(range(0, 10)))
            self.bottom_panel.canvas.draw()
        elif chart_counter == 2:
            df = pandas.DataFrame(index=range(0, 3), columns=['a'], data=[4, 8, 3])
            self.bottom_panel.ax.clear()
            BaseChart('Bar chart', '', '', 8, self.bottom_panel, False).single_stacked_bar(df, ['a', 'b', 'c'])
            self.bottom_panel.canvas.draw()
            pass
        else:
            pass


if __name__ == "__main__":
    app = wx.App()
    frame = TestFrame().Show()
    app.MainLoop()
