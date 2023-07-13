import csv

# Define the input and output file paths
input_file = '/Users/claire/Desktop/Practical/udacity_autonomous_driving/data/driving_log.csv'
output_file = '/Users/claire/Desktop/Practical/udacity_autonomous_driving/data/driving_log.csv'


# Function to remove leading spaces from each element in a list
def remove_leading_spaces(elements):
    return [element.strip() for element in elements]


# Function to delete characters from each element in a list
def delete_characters(elements, characters):
    return [element.replace(characters, '') for element in elements]

# Read the CSV file and store its contents in a list
data = []
with open(input_file, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        modified_row = delete_characters(row, '/Users/claire/Desktop/')
        modified_row = remove_leading_spaces(modified_row)
        data.append(modified_row)

# Write the modified list back to a new CSV file
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print('Character deletion complete. Modified data saved to', output_file)
