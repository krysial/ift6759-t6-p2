import os
for file in os.listdir():
     if os.path.isfile(file):
         with open(file) as f:
             lines = f.readlines()
         size = int(len(lines)*0.75)
         print(size, type(size))
         with open(os.path.join("new", "Train", file), 'w') as f:
             f.writelines(lines[:size])
         with open(os.path.join("new", "Valid", file), 'w') as f:
             f.writelines(lines[size:])
