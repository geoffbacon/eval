lint:
	@echo "Running pylint"
	pylint evaluate

test:
	@echo "Running tests"
	py.test evaluate
