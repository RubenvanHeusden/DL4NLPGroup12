# DL4NLP
There seems to be a little bug in the 'torchtext' module (which is required in the current version):
change '<python install folder>\Lib\site-packages\torchtext\utils.py' line 130 'csv.field_size_limit(sys.maxsize)' to 'csv.field_size_limit(maxInt)'
