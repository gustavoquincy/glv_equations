import pandas as pd
import pyarrow
import pyarrow.csv
import numpy as np

def write_p_table(m_num_species):
#	table = pyarrow.csv.read_csv("system_state_at_t_1.0.csv");
	interaction_table = pyarrow.csv.read_csv("interaction.csv");
	num_species = m_num_species;
	interaction_output = pd.DataFrame(columns=[f'interaction{i}' for i in range(num_species*num_species)]);
	growth_table = pyarrow.csv.read_csv("growth_rate.csv");
	growth_output = pd.DataFrame(columns=[f'growth rate{i}' for i in range(num_species)]);
	sigma_table = pyarrow.csv.read_csv("sigma.csv");
	sigma_output = pd.DataFrame(columns=[f'sigma{i}' for i in range(num_species)]);
	for i in range(0, growth_table.num_rows, num_species):
		slice_index = np.arange(i, i+num_species)
		panda_slice = growth_table.take(slice_index).to_pandas()
		slice_transpose = panda_slice.T
		slice_transpose.columns = [f'growth rate{i}' for i in range(num_species)]
		growth_output = pd.concat([growth_output, slice_transpose], ignore_index=True)
	for i in range(0, interaction_table.num_rows, num_species*num_species):
		slice_index = np.arange(i, i+num_species*num_species)
		panda_slice = interaction_table.take(slice_index).to_pandas()
		slice_transpose = panda_slice.T
		slice_transpose.columns = [f'interaction{i}' for i in range(num_species*num_species)]
		interaction_output = pd.concat([interaction_output, slice_transpose], ignore_index=True)
	for i in range(0, sigma_table.num_rows, num_species):
		slice_index = np.arange(i, i+num_species)
		panda_slice = growth_table.take(slice_index).to_pandas()
		slice_transpose = panda_slice.T
		slice_transpose.columns = [f'sigma{i}' for i in range(num_species)]
		sigma_output = pd.concat([sigma_output, slice_transpose], ignore_index=True)
	dilution_table = pyarrow.csv.read_csv("dilution.csv").to_pandas();
#	growth_output.index = table.column_names
#	sigma_output.index = table.column_names
#	interaction_output.index = table.column_names
#	dilution_table.index = table.column_names
	pd.concat([growth_output, interaction_output, sigma_output, dilution_table], axis=1).to_csv('parameters.csv');
