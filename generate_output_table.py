import pyarrow as pa
import pyarrow.csv
import pyarrow.compute as pc
import numpy as np
import math
import pandas as pd

def write_o_table(m_num_species, thres):
	table = pyarrow.csv.read_csv("system_state_at_t_100.000000.csv");
	threshold = thres
	num_species = m_num_species
	parameters_condition = table.num_columns
	initial_condition = int(table.num_rows / num_species)
	stabilities, shannons, diversities = [], [], []
	for name in table.column_names:
		vec = {i: table[name][num_species*i:num_species*(i+1)] for i in range(0,initial_condition)}
		tag = {i: -1 for i in range(0,initial_condition)}
		counter = 0
		vec_sum = np.zeros(num_species)
		for i in range(0, initial_condition):
			if tag[i] == -1:
				vec1 = vec[i]
				for j in range(i+1, initial_condition):
					if tag[j] == -1:
						vec2 = vec[j]
						norm2 = np.sqrt(np.sum(np.power(np.subtract(vec1, vec2), 2)))
						if norm2 < threshold:
							tag[j] = counter
							tag[i] = counter
				counter += 1
				vec_sum = np.add(vec_sum, vec[i])
		res = pa.array(tag.values())
		modes = pc.mode(res, num_species) #here 3 bc num_species equals 3
		#print(pc.count_distinct(res))
		stabilities.append(pc.count_distinct(res))
		counts = [mode['count'].cast(pa.float32()) for mode in modes]
		props = [pc.divide(count, pc.sum(counts)) for count in counts]
		multiplication = [pc.multiply(prop, pc.ln(prop)) for prop in props]
		shannon = pc.power(math.e, pc.multiply(pc.sum(multiplication), -1.0))
		#print(shannon)
		shannons.append(shannon)
		vec_props = [vec/np.sum(vec_sum) for vec in vec_sum]
		multiplication = [vec_prop * np.log(vec_prop) for vec_prop in vec_props]
		diversity = math.exp(-np.sum(multiplication))
		diversities.append(diversity)
	pd.concat([pd.DataFrame(stabilities, index=table.column_names, columns=['stability']), 
							pd.DataFrame(shannons, index=table.column_names, columns=['shannon']), 
							pd.DataFrame(diversities, index=table.column_names, columns=['diversity'])], 
							axis=1).to_csv("glv_output.csv")
