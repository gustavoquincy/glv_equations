from generate_parameter_table import write_p_table
from generate_output_table import write_o_table 
import pandas as pd

num_species = 3
threshold = 0.01
write_p_table(num_species)
#write_o_table(num_species, threshold)

