import pickle

# List of file names
file_list = []

for i in range(9):
    str_fname = 'cane_frame_analysis_dec19_seq'+str(i)+'.pkl'
    file_list.append(str_fname)

# List to hold the data from each file
data_list = []

# Loop through the list of file names
for file in file_list:
    # Open the file and load the data
    with open(file, 'rb') as f:
        data = pickle.load(f)
    # Add the data to the list
    data_list.append(data)

# Combine all the data into a single object
combined_data = {'data': data_list}

print(data_list)
# Save the combined data to a new file
with open('cane_frame_analysis_dec19_combined_data.pkl', 'wb') as f:
    pickle.dump(combined_data, f)