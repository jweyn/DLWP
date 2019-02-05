#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Test retrieval and processing of CFSReforecast data.
"""

from DLWP.data import CFSReanalysis, CFSReforecast
from datetime import datetime
import pandas as pd


start_date = datetime(2003, 1, 1)
end_date = datetime(2009, 12, 31)
dates = list(pd.date_range(start_date, end_date, freq='D').to_pydatetime())
variables = ['z500']
max_f_hour = 144

meta_cfs = CFSReanalysis(root_directory='/home/disk/wave2/jweyn/Data/CFSR', file_id='dlwp_')
meta_cfs.set_dates(dates)
meta_cfs.open()

cfs = CFSReforecast(root_directory='/home/disk/wave2/jweyn/Data/CFSR/reforecast', file_id='dlwp_')

cfs.retrieve(dates, variables=variables, verbose=True)
cfs.set_dates(dates)
cfs.write(variables=variables, forecast_hours=max_f_hour, interpolate=(meta_cfs.lat, meta_cfs.lon), verbose=True)

meta_cfs.close()
