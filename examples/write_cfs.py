#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Test retrieval and processing of CFSReanalysis data.
"""

from DLWP.data import CFSReanalysis
from datetime import datetime
import pandas as pd


start_date = datetime(1979, 1, 1)
end_date = datetime(2010, 12, 31)
dates = list(pd.date_range(start_date, end_date, freq='D').to_pydatetime())
variables = ['TMP', 'R H', 'HGT', 'U GRD', 'V GRD', 'V VEL']
levels = [200, 250, 300, 500, 700, 850, 925, 1000]

cfs = CFSReanalysis(root_directory='/home/disk/wave2/jweyn/Data/CFSR', file_id='analysis_', run_type='nl')

cfs.retrieve(dates, verbose=True)
cfs.set_dates(dates)
cfs.write(variables=variables, levels=levels, verbose=True, n_proc=8)
