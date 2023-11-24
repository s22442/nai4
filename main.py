from ucimlrepo import fetch_ucirepo
import csv

mushroom = fetch_ucirepo(id=73)

mushroom_X = mushroom.data.features
mushroom_y = mushroom.data.targets

print(mushroom.metadata)
# print(mushroom.variables)

with open('sonar.csv') as csvfile:
    reader = csv.reader(csvfile)
    sonar = []
    for row in reader:
        sonar.append(row)

print(sonar)
