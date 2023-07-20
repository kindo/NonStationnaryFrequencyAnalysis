import os
import numpy as np
import pandas as pd



def load_climate_index(p=None):
    
  
        
    if p is None:
        return "Pleease provide a path"

    name_indice = os.path.basename(p).split(".")[0]

    colnames = ["year", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    climate_indice_ts_ = pd.read_table(os.path.normpath(p),  skipfooter=3, skiprows=1, engine='python', sep='\s+', header=None,)
    col_replace = {old:new for old, new in zip(climate_indice_ts_.columns, colnames)}
    climate_indice_ts_ = climate_indice_ts_.rename(columns = col_replace)
    
    climate_indice_ts = (climate_indice_ts_.set_index("year").melt(ignore_index=False, var_name="month", value_name="indice")
                        .assign(indice = lambda x: x.indice.astype(float).round(2))
    .assign(Month_num = lambda x: 
            x.month.map({"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6, "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}))
    .assign(Date = lambda x: pd.DatetimeIndex(pd.to_datetime(x.index.astype(str) + "-" + x.Month_num.astype(str), format="%Y-%m",)))
    .set_index("Date")
    .drop(columns=["month", "Month_num"])

    )

    return climate_indice_ts.rename(columns={"indice":name_indice})


def get_date_Month(df):
    return (pd.to_datetime(df.dt.year.astype(str) +
                           '-' + df.dt.month.astype(str) 
                           + '-' + '01',
                            format = r'%Y-%m-%d')
                            )

def get_date_time(df):
    return (
        pd.to_datetime(df.Year.astype(str) +
                            '-' + df.Month.astype(str) + '-' + df.Day.astype(str) 
                            + '-' + df.Time.astype(str), format = '%Y-%m-%d-%H')
                            )