class XOR_gate:
	def __init__(self):
		self.w1 = 1
		self.w2 = -1
		self.w3 = -1
		self.w4 = 1
		self.w11 = 1
		self.w22 = 1
		self.x1 = None
		self.x2 = None
		self.x11 = None
		self.x22 = None


	def andnot_activation_function(self,w1,w2):
		yin = w1*self.x1+w2*self.x2
		if yin >= 1:
			return 1
		else:
			return 0

	def or_activation_function(self):
		yin = self.w11*self.x11+self.w22*self.x22
		if yin >=1:
			return 1
		else:
			return 0

	def XOR_ops(self,x1,x2):
		self.x1 = x1
		self.x2 = x2
		self.x11 = self.andnot_activation_function(self.w1,self.w2)
		self.x22 = self.andnot_activation_function(self.w3,self.w4)
		Y = self.or_activation_function()
		print('%d\t%d\t%d'% (x1,x2,Y))
		return Y


def main():
	XOR1 = XOR_gate()
	Y = XOR1.XOR_ops(1,1)
	Y = XOR1.XOR_ops(1,0)
	Y = XOR1.XOR_ops(0,1)
	Y = XOR1.XOR_ops(0,0)
	
main()