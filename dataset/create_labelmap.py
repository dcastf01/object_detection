def label_map_v1(itemsname,filename):
  i=1
  with open(filename, 'w') as the_file:
    for item in itemsname:
      
      the_file.write('item\n')
      the_file.write('{\n')
      the_file.write('id :{}'.format(int(i)))
      the_file.write('\n')
      the_file.write("name :'{0}'".format(str(item)))
      the_file.write('\n')
      the_file.write('}\n') 
      i+=1
