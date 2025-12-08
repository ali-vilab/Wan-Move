.PHONY: format

format:
	isort generate.py gradio wan
	yapf -i -r *.py generate.py gradio wan
