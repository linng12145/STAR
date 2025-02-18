from meshing import *
from trip_count import *
from trip2trips import *
from trips2new import *
from trips_drop import *
from trips_split import *
from trips_graph import *
from test_delete_graph import *

data_name = 'AIS_2023_4month'

print(data_name)


meshing('csv', data_name)
trip_count('csv', data_name)
trip2trips('csv', data_name)
trips2new('csv', data_name)
trips_drop('csv', data_name)
trips_split('csv', data_name)
trips_graph('csv', data_name)
test_delete_graph('csv', data_name)