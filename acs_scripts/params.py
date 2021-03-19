# the geographical level at which the data needs to be retrieved
# must be one of ['us', 'state', 'county', 'tract', 'block group', 'block']]
geo_level_name = 'state legislative district (upper chamber)'

# data_source (string): which type of census data to use,
#             must be one of ['acs5', 'acs3', 'acs1', 'sf1']
data_source = 'acs5'

year = 2018
tabletype = 'detail'
# tabletype='subject'

variable_names = ['race']

# the following parameters specify which real area you want to gather the data for
# if you want some specific areas, name them
# else specify the area as None, and all areas at that level will be returned

# list of names of US states (e.g. 'Pennsylvania'), or None
state_names = ['Illinois']
# list of list of names of counties in these states (e.g. 'York County'), or None
# entries in the outer list correspond to the different states mentioned as state_names
# and each inner list holds the names of the counties in that state
# If None, we take all counties in every state
# If a specific element of outer list is None, we will take all counties in that state
county_names = None
