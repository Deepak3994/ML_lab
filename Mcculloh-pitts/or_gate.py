class or_gate:
	def __init__(self):
		self.w1 = 3
		self.w2 = 3
		self.x1 = None
		self.x2 = None
		

	def activation_function(self):
		yin = self.w1*self.x1+self.w2*self.x2
		if yin >= 3:
			return 1
		else:
			return 0

	def or_ops(self,x1,x2):
		self.x1 = x1
		self.x2 = x2
		Y = self.activation_function()
		print('%d\t%d\t%d'% (x1,x2,Y))
		return Y


def main():
	or1 = or_gate()
	Y = or1.or_ops(1,1)
	Y = or1.or_ops(1,0)
	Y = or1.or_ops(0,1)
	Y = or1.or_ops(0,0)
	
main()