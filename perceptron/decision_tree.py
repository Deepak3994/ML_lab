import numpy as np
import warnings


def cal_entropy(pos,neg):
	total = pos + neg
	if pos != 0:
		positive = -(pos/total) * np.log2(pos/total)
	else:
		positive = 0
	if neg !=0:
		negative = - (neg/total) * np.log2(neg/total)
	else:
		negative = 0
	entropy = positive + negative
	#print(entropy)
	return entropy

def cal_gain(field,y,total_entropy):
	pos_field_value = {}
	neg_field_value = {}
	positive_list = []
	negative_list = []

	for i in range(len(field)):
		if not field[i]  in pos_field_value:
			pos_field_value[field[i]] = 0
			neg_field_value[field[i]] = 0
			if y[i] == 1:
				pos_field_value[field[i]] = 1
			else:
				neg_field_value[field[i]] = 1
		else:
			if y[i] == 1:
				pos_field_value[field[i]] = 1 + int(pos_field_value[field[i]])
			else:
				neg_field_value[field[i]] = 1 + int(neg_field_value[field[i]])
	
	#print(pos_field_value)
	#print(neg_field_value)

	total = len(y)
	for key in pos_field_value:
		positive_list.append(pos_field_value[key])
		negative_list.append(neg_field_value[key])
	sum = 0

	for i in range(len(positive_list)):
		entropy = cal_entropy(positive_list[i],negative_list[i])
		sum = sum + ((pos_field_value[i]+neg_field_value[i])/total) * entropy
		##print(sum)
	gain = total_entropy - sum
	#print(gain)
	return(gain)
	
	
def calculate_total_entropy(Y):
	pos = 0
	neg = 0
	for i in range(len(y)):
		if Y[i] == 1:
			pos = pos + 1
		else:
			neg = neg + 1
	##print(pos,neg)
	total_entropy = cal_entropy(pos,neg)
	return total_entropy


data = [[0,0,1,2,2,2,1,0,0,2,0,1,1,2],
		[0,0,0,1,2,2,2,1,2,1,1,1,0,1],
		[0,0,0,0,1,1,1,0,1,1,1,0,1,0],
		[0,1,0,0,0,1,1,0,0,0,1,1,0,1]]

y = [0,0,1,1,1,0,1,0,1,1,1,1,1,0]
total_entropy =  calculate_total_entropy(y)
##print(total_entropy)
gain = []
for i in range(len(data)):
	gain.append(cal_gain(data[i],y,total_entropy))
print(gain)