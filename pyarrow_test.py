import pyarrow.csv
import pyarrow.compute as pc
import math

table = pyarrow.csv.read_csv("system_state_at_time_100.000000.csv");
#len(table.column_names)=200

for name in table.column_names:
	column = []
	for i in range(0, 600, 3):
		vec1 = table['system_0'][i:i+3]
		for j in range(i, 600, 3):
			vec2 = table['system_0'][j:j+3]
			column.append(pc.sqrt(pc.sum(pc.power(pc.subtract(vec1, vec2), 2))))
	print(column)

