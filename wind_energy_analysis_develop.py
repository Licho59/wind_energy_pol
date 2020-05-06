#coding: utf-8
import os, re, csv, random
from glob import glob
from datetime import date, datetime, timedelta
import webcolors
import numpy as np
import pandas as pd
from plotly.offline import plot
import plotly.graph_objs as go
import plotly.express as px
from plotly import subplots
import fire


PATHNAME = os.getcwd()

def get_file(template):
    """Function checks if there is only 1 file for a given path template and returns path of file in /Data/wind_csv/ folder/. Returns None.
    """
    # glob function creates a list of files appropriate with template
    filenames = glob(template)
    if len(filenames) == 1:
        return filenames[0]
    elif len(filenames) > 1:
        raise Exception(
            'There are more than one files in /Data/wind_csv/ folder - leave only 1 and repeat!')

def get_raw_file_path(year, month):
    """Return: path template of the raw(source) file for a given year and month.
    """
    return get_file(PATHNAME + f'./Data/wind_csv/{year}/PL_GEN_WIATR_{year}{month:02}*.csv')

def get_raw_files_list(year):
    """Return: Two lists: first one for downloaded from PSE webpage files, second one for months numbers of appropriate files. When there are no files downloaded or duplicates - error message is generated.
    """
    raw_files = []
    month_numbers = []
    for month_num in range(1, 13):
        file = get_raw_file_path(year, month_num)
        if file:
            raw_files.append(file)
            month_numbers.append(month_num)
    if raw_files == []: return
    return raw_files, month_numbers

def transformed_file_path(year, month):
    """Return: path to the file which is ready to use with plot functions.
    """
    if not os.path.exists(f'./Data/wind_ready/{year}/'):
        os.makedirs(f'./Data/wind_ready/{year}')
    return PATHNAME + f'./Data/wind_ready/{year}/{year}{month:02}.csv'

def get_transformed_file(year, month):
    """Return: file ready to use with plot functions after checking if there is only 1 file for given year and month (no copies).
    """
    return get_file(transformed_file_path(year, month))

def files_list(year):
    """
    Return: List of transformed 'csv' files for given year or message when the data is lacking.
    """
    ready_list = sorted(glob(f'./Data/wind_ready/{year}/*.csv'))
    raw_files = get_raw_files_list(year)
    month_numbers = raw_files[1]
    if len(ready_list) == 0:
        if raw_files is None:
            if year <= datetime.now().year and year >= 2012:
                print(f'There is no data for {year}. You should try to download it from PSE webpage.')
            return
        for month_number in month_numbers:
            save_clean_data(year, month_number)
        ready_list = sorted(glob(f'./Data/wind_ready/{year}/*.csv'))
    month_numbers.append('all')
    return ready_list, month_numbers

def save_clean_data(year, month_num):
    """Sends transformed file to the right folder with files ready for using in plotting functions. Returns None.
    """
    df = pd.read_csv(get_raw_file_path(year, month_num),
                     encoding='iso 8859-1',
                     sep=';',
                     skiprows=[0],
                     usecols=[0, 1, 2],
                     names=['Date', 'Time', 'Total_Wind_Power(MWh)'],
                     index_col='Date',
                     converters={1: lambda x: x.replace('2A', '2'),
                                 2: lambda x: x.replace(',', '.')})
    df.to_csv(transformed_file_path(year, month_num), header=True)

def get_clean_data(year, month):
    """Return: Pandas dataframe for given month of year."""
    if not os.path.exists(f'./Data/wind_ready/{year}/{year}{month}.csv'):
        save_clean_data(year, month)
    return pd.read_csv(get_transformed_file(year, month))

def wind_hourly(year, month_num):
    """Input: function takes the year and the number of month in range between 1 and number of months for given year 
    Return: dataframe either for month's wind power generation or for all months - to use for visualization, analysis, modelling.
    """
    months_numbers = files_list(year)[1]
    if month_num == 'all':
        df_all = [get_clean_data(year, month_num)
                  for month_num in months_numbers[:-1]]
        df = pd.concat(df_all)
    elif month_num in months_numbers:
        df = get_clean_data(year, month_num)
    else:
        print(f"Data for {month_name(month_num)} of {year} year is not available.")
        return
    df['Date'] = pd.to_datetime(df['Date'].astype('str'))
    df.set_index('Date', inplace=True)
    return df

def wind_daily(year, month_num):
    """Return: dataframe of wind generation where indexing is by day values not by hours.
    """
    if month_num not in files_list(year)[1]:
        print("Data for that month is not available or wrong parameter was given for month number.")
        return
    df_days = wind_hourly(year, month_num).resample('D').sum().iloc[:, [1]]
    df_days.rename(
        columns={'Total_Wind_Power(MWh)': 'Wind_Daily(MWh)'}, inplace=True)
    return df_days

def month_name(month_num):
    """Return: name of month to use it while formatting strings for plotting labels.
    """
    return date(1990, int(month_num), 1).strftime('%B')

def month_names(year):
    """Return: list of all month names."""
    return [month_name(month_num) for month_num in files_list(year)[1][:-1]]

def years_list():
    """Return: list of subsequent years with data files ready for graph plotting.
    """
    y_list = os.listdir(PATHNAME + './Data/wind_csv')
    if y_list[0] == 'ipynb':
        return y_list[1:]
    else:
        return y_list

def checking_year(year):
    """Helpful function to check if year argument is correct or to return random from the list if year is None."""
    if year == None:
        year = random.choice(years_list())
    elif str(year) not in years_list():
        return
    return year

def get_random_colors():
    """Return: random chosen set of colors to use in plotting functions."""
    colors = webcolors.CSS3_HEX_TO_NAMES
    aborted_colors = ['white', 'mintcream', 'snow', 'lightyellow', 'whitesmoke', 'linen','beige','seashell','floralwhite','oldlace', 'lavenderblush','ivory','ghostwhite','mediumslateblue','aliceblue', 'lightgoldenrodyellow', 'honeydew', 'azure', 'cornsilk', 'black']
    palette = [colors[key] for key in colors]
    random.shuffle(palette)
    chosen_palette = [palette[i]
                      for i in range(16) if palette[i] not in aborted_colors]
    return chosen_palette

def current_year(year):
    """
    Return: cumulative dataframe for all months of the current year but with None values for months to come(future monts until December).
    """
    wind_grow_ = wind_daily(year, 'all')
    # preparing dataframe for days in future where date is not available yet
    last_date = wind_grow_.index[-1]
    last_day_of_year = datetime(int(year), 12, 31)
    no_data_days = (last_day_of_year - last_date).days
    no_data_range = pd.date_range(
        last_date + timedelta(days=1), periods=no_data_days, freq='D')
    data_vals = np.array([None] * no_data_days)
    df = pd.DataFrame({'Date': no_data_range, 'Wind_Daily(MWh)': data_vals})
    df.set_index('Date', inplace=True)
    wind_grow = pd.concat([wind_grow_, df.iloc[:]], axis=0)
    return wind_grow

    
def wind_1(year=None, month_number=None):
    """
    Input: number of month in range of 1-12 (or number for passed months of present year), by default it is random chosen month when function is being called without argument.
    Return: graph presenting daily total wind generation in Poland for given or random chosen month. 
    """
    year_ = checking_year(year)
    if year_ is None:
        print(f"No data for a given year: {year}")
        return
    else:
        year = year_

    if month_number is None:
        month_number = random.choice(files_list(year)[1][:-1])
    elif month_number not in files_list(year)[1]:
        print(f"Data for that month is not available or wrong parameter ({month_number}) was given for month number.")
        return
    else:
        month_number = month_number

    month = month_name(month_number)
    wind_m = wind_daily(year, month_number) / 10**3
    m_avg = wind_m.iloc[:, 0].mean()
    
    # using plotly.graph_obs object with iplot method of plotly
    data = [go.Bar(x=wind_m.index, y=wind_m['Wind_Daily(MWh)'].values,
             marker={'color': 'orange'})]
    layout = {'xaxis': {'title': 'Days'},
            'yaxis': {'title': 'Total Power (GWh)'},
            'shapes': [{'type': 'line', 'x0': wind_m.index[0], 'x1':wind_m.index[-1], 'y0':m_avg, 'y1':m_avg,
            'line':{'color': 'green', 'width': 2, 'dash': 'longdash'}}],
            'annotations': [{'x': wind_m.index[-3], 'y': m_avg,
                    'text': 'Avg Power=' + str(round(m_avg, 1)) + ' GWh',
                    'showarrow':True, 'arrowhead':1, 'ax':0, 'ay':-30}],
            'autosize': True,
            'title': f'Generation of Wind Power in {month} of {year}'}
    plot(go.Figure(data=data, layout=layout))

def wind_2(year=None):
    """
    Input: Random chosen number of year from available in Data if year not given;
    Return: plot presenting wind power generation for each day of given year.
    """
    year_ = checking_year(year)
    if not year_:
        print(f"No data for a given year: {year}")
        return
        
    months = 'all'  # enables dataframe for all months of year
    wind_y = wind_daily(year_, months)/10**3 # to get TeraWattHours
    wind_y.rename(columns={"Wind_Daily(MWh)":"dailyMwh"}, inplace=True)
    y_avg = wind_y['dailyMwh'].mean()
    
    # implementing plotly.express lib being a counterpart for cufflinks lib
    fig = px.bar(wind_y,
                x = wind_y.index,
                y='dailyMwh',
                #color='dailyMwh',
                title=dict(text="Day by day generation of wind power in {}".format(year_), x=0.5, y=0.95),
                labels=dict(dailyMwh="Daily Generation (MWh)", x="Days"))
    fig.update_layout(
                    annotations=[dict(x=wind_y.index[-2], y=y_avg, text='Average Power=' + str(round(
                    wind_y.iloc[:, 0].mean(), 1)) + ' GWh',
                    textangle=0, showarrow=True, arrowhead=8,
                    ax=-60, ay=-20)],
                    shapes=[dict(type='line', line={'dash':'dot', 'color':'orange'}, xref='paper', x0=0, x1=1, yref='y', y0=y_avg, y1=y_avg, )])
    fig.show()

def wind_3(year=None):
    """
    Return: plot presenting wind power generation for each month of given year
    """
    year_ = checking_year(year)
    if not year_:
        print(f"No data for a given year: {year}")
        return

    months = 'all'
    m_names = month_names(year_)

    # dataframe resampled to months with index being months names
    wind_monthly = wind_daily(year_, months).resample('M').sum() / 10**3
    colors = random.choice(get_random_colors())

    data = [go.Bar(x=m_names, y=round(wind_monthly.iloc[:, 0], 3),
            marker=dict(color=colors), name='Power by Month')]
    layout = go.Layout(legend=dict(x=0.9, y=0.95),
            xaxis=dict(title='Months'),
            yaxis=dict(title='Total Power (GWh)'),
            title=dict(text="Monthly Wind Power Generation in {}".format(
                   year_), x= 0.5, y=0.9),
            showlegend=True,
            #width=1000, height=600,
            )
    plot(go.Figure(data=data, layout=layout))


def wind_4(year=None, plot='line'):
    """
    Return: linear or bar graph for cumulative amount of wind energy generated from the beginning of the year.
    Bar chart is returned with any kind of argument put after year number( for example: (2019, 1) or (2018,'bar'))
    """
    year_ = checking_year(year)
    if not year_:
        print(f"No data for a given year: {year}")
        return
    year = year_

    months = 'all'
    # dataframe resampled to months with index being months names
    w_daily = wind_hourly(year, months).resample('D').sum().iloc[:, [1]]
    w_daily.rename(columns={'Total_Wind_Power(MWh)': 'dailyMwh'},inplace=True)
    wind_grow = w_daily.cumsum() / 10**3
    # total value for the last day of the plot
    last_day = round(wind_grow.max()[0], 1)
    if plot == 'line':
        plot_f = px.scatter
    else:
        plot_f = px.bar
    
    fig = plot_f(wind_grow,
                x=wind_grow.index,
                y='dailyMwh',
                color='dailyMwh',
                title=dict(text=f"Wind Power Cumulation in {year}",
                x=0.5, y=0.9),
                labels=dict(dailyMwh="Total Power (Gwh)", x="Days"),
                template='plotly_dark'
                )
    fig.update_layout(
        annotations=[dict(x=wind_grow.index[-5], y=last_day,
                    text='Total=' + str(last_day) + ' GWh',
                    textangle=0, showarrow=True, arrowhead=1,
                    ax=0, ay=-20)]
                        )
    fig.show()       
    
def wind_4a():
    """
    Return: Separate linear graphs for each year in data folder with cumulative amount of wind energy generated).
    """
    years = years_list()

    for year in years:
        wind_4(year)

def wind_4b():
    """
    Return: A single graph of number of plots for every year of wind generation's cumulative value
    """
    data = []
    years = years_list()
    cur_year = years[-1]
    # 2012 year moved to the end of list to facilitate plotting disturbed by the lack of data for first months of 2012
    years.append(years.pop(0))
    for year in years:
        wind_grow = wind_daily(year, 'all')
        # preparing data for current year to facilitate correct plotting
        if year == cur_year:
            wind_grow = current_year(year)

        # code used for leap years(February with 29 days)
        if wind_grow.shape[0] == 366:
            wind_grow.drop(wind_grow.index[59], inplace=True)
        # common adjustments in data for all years
        wind_grow = wind_grow.cumsum() / 10**6
        wind_grow.rename(columns={'Wind_Daily(MWh)': year}, inplace=True)
        wind_grow.index = wind_grow.index.strftime('%b %d')
        last_data = round(wind_grow.max()[0], 1)

        trace = go.Scatter(x=wind_grow.index, y=wind_grow[year].values,
                           name=f'{year}: {last_data} TWh')
        data.append(trace)

    layout = {'xaxis': {'title': 'Days of year', 'nticks': 25, 'tickangle': -45, 'ticks': 'inside'},
              'yaxis': {'title': 'Total Power (TWh)'},
              'title': 'Cumulative Wind Power Generation in Years',
              #'width': 900, 'height': 550
             }
    fig = go.Figure(data=data, layout=layout)
    plot(fig)


def wind_4c():
    """
    Return: Total yearly production of wind energy in GWh.
    """
    years = years_list()
    df = pd.DataFrame()
    for year in years:
        wind_year = (wind_daily(year, 'all') / 10**6).resample('Y').sum()
        df[year] = wind_year['Wind_Daily(MWh)'].values
    df_tidy = pd.melt(df, var_name='Year', value_name='Total Power')
    fig = px.bar(df_tidy, x='Year', y='Total Power', color='Year',
             labels={'Total Power': 'Total Power in TWh'},
             title=dict(text='Wind Energy Total Production per Year', x=0.5, y=0.9))
    fig.show()

def wind_5(year=None):
    """
    Return: plot showing an average hour wind generation for given year
    """
    year_ = checking_year(year)
    if not year_:
        print(f"No data for a given year: {year}")
        return
    else:
        year = year_
        
    df = wind_hourly(year_, 'all')
    h_wind = df.pivot_table(index='Date', columns='Time',
                            values='Total_Wind_Power(MWh)').mean()
    hour_avg = h_wind.mean()

    data = [go.Bar(x=h_wind.index, y=h_wind.values)]
    layout = {'shapes': [{'type': 'line',
                          'x0': h_wind.index[0], 'y0': hour_avg, 'x1': len(h_wind.index), 'y1': hour_avg,
                          'line': {'color': 'red', 'width': 2, 'dash': 'longdash'}}],
              'showlegend': False,
              'annotations': [{'x': h_wind.index[-10], 'y': hour_avg,
                               'text': 'Avg Power=' + str(round(hour_avg, 1)) + ' MWh',
                               'showarrow':True, 'arrowhead':1, 'ax':0, 'ay':-30}],
              'xaxis': {'title': 'Hours'},
              'yaxis': {'title': 'Generation by Hour (MWh)'},
              'title': f"Average Wind Generation per Hour in {year}",
              #'width': 800, 'height': 400
              }
    plot(go.Figure(data=data, layout=layout))

def wind_6(year=None):
    """
    Return: each month subplots for hour wind generation in given year
    """
    year_ = checking_year(year)
    if not year_:
        print(f"No data for a given year: {year}")
        return
    else:
        year = year_

    rows, cols = (4, 3)
    m_names = [month_name(m_num) for m_num in range(1, 13)]
    fig = subplots.make_subplots(rows=rows, cols=cols,
                            shared_xaxes=True, shared_yaxes=True,
                            subplot_titles=m_names,
                            print_grid=False)
    row, col = 1, 0
    for month_num in files_list(year)[1][:-1]:
        # month dataframe with energy values for each hour
        df = wind_hourly(year, month_num)
        h_wind = df.pivot_table(
            index='Date', columns='Time', values='Total_Wind_Power(MWh)').mean()
        # average value of wind energy for all day
        day_avg = h_wind.mean()
        m_name = month_name(month_num)
        trace_month_num = go.Bar(x=h_wind.index, y=h_wind.values)
        trace_avg = go.Scatter(
            x=[h_wind.index[0], len(h_wind.index)],
            y=[day_avg] * 2,
            mode='lines+text',
            line={'width': 0.8},
            text=[None, 'avg=' + str(int(day_avg)) + ' MWh'],
            textposition='middle left')
        if year != 2012:
            if month_num <= (row * cols): col += 1
            else:
                row += 1
                col = 1
        else:
            if month_num <= (row * cols): col += 1
            else:
                row += 1
                col = month_num % cols

        fig.append_trace(trace_month_num, row=row, col=col)
        fig.append_trace(trace_avg, row=row, col=col)

    fig.layout.update({'title':'Average Hour Wind Generation in ' + f'{year}',
                       'xaxis': {'title': 'Hours'},
                       'yaxis': {'title': 'Avg Power'},
                       'showlegend': False,
                       #'width': 800, 'height': 700
                       }, autosize=True)
    plot(fig)


def run(year=None, month=None):
    wind_1(year, month)
    wind_2(year)
    wind_3(year)
    wind_4(year)
    wind_5(year)
    wind_6(year)
    wind_4b()
    wind_4c()

if __name__ == "__main__":
    fire.Fire(run)
