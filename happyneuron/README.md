# HAPPYNeurons app library

## Development

All sub-packages on this folder must be name based on the third-part software that it will wrap.

The files inside should both be importable or executed ina terminal on stand-alone fashion.

Suggested template:

* Imports
* def parse_function()
* def main_function(args)
* main()
* * args = parse()
* * main_function(args)