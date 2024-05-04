class A:
	def __init__(self):
		pass

	def forward(self, ):
		print("A haha")
		return 100

class B(A):
	def __init__(self):
		pass

	def forward(self, ):
		print("B haha")



a = []
print(a.__class__)

# a2 = A()
# b = B()
# a_forward = getattr(A, 'forward')
# a2_forward = getattr(A, 'forward')
#
# print(a_forward)
# print(a2_forward)
# b_forward = getattr(B, 'forward')
# print(b_forward)
# print(a_forward == b_forward)