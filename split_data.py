import os

ids = []
for f in os.listdir('./data'):
  if f.find('.gui') != -1:
    ids.append(f[:-4])

for i in range(len(ids)):
  if i < 1500:
    os.rename(ids[i] + '.gui', 'train/' + ids[i] + '.gui')
    os.rename(ids[i] + '.png', 'train/' + ids[i] + '.png')
  elif i < 1625:
    os.rename(ids[i] + '.gui', 'dev/' + ids[i] + '.gui')
    os.rename(ids[i] + '.png', 'dev/' + ids[i] + '.png')
  else:
    os.rename(ids[i] + '.gui', 'test/' + ids[i] + '.gui')
    os.rename(ids[i] + '.png', 'test/' + ids[i] + '.png')
  
