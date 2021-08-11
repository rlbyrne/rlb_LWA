pro test_idl

	args = Command_Line_Args(count=nargs)
	arg1 = args[0]
	arg2 = args[1]
	sum = fix(arg1)+fix(arg2)
	print, sum
end
