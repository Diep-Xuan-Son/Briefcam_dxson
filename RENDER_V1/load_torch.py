import xlrd
import json
# loc = ("H:\\Attribute_predict\\car_attribute\\Dataset_v2.xlsx")
# wb = xlrd.open_workbook(loc)
# sheet = wb.sheet_by_index(0)
# # For row 0 and column 0
# label = sheet.row_values(0)[1:]
# print(len(label))
# data = dict()
# for i in range(1, sheet.nrows):
# 	ids = sheet.row_values(i)[0]
# 	arr = []
# 	for j in sheet.row_values(i)[1:]:
# 		if j == '':
# 			arr.append(0)
# 		else:
# 			arr.append(1)
# 	data[ids] = arr
# 	print(len(arr))
# with open("H:\\Attribute_predict\\car_attribute\\Dataset_v2.json", "w") as fi:
# 	json.dump(data, fi, indent = 4)