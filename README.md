# APS program for string matching using OpenCL
Simple searching string in large text files. User can enter file and pattern he is searching. Program is using Knuth-Morris-Pratt algorithm for searching. 
## Features
* Counting occurances
* Printing offset of occurance
* Printing line and line number
* Multi-threading 

## Usage 
```
aps [-tlodh] [-p pattern] [-f file]
```
* -t - multithreading option (default is without)
* -l - ouputs number of line and line itself
* -d - debug output at the end
* -p <pattern> - input needle (required)
* -f <file> - input file haystack (required)
* -h - help output

## Installation with clone
```
git clone https://github.com/simonharvan/aps-search-string.git
cd aps-search-string/aps
gcc -framework OpenCL main.c -o <output-file>
```

## Testing 
Comparison of several string-matching programs (APS v1.0, APS v2.0, GNU Grep and BSD Grep)

### Comparison with different file sizes seaching for one-word pattern
![](aps/2.0graf_5.png "Comparison of several string-matching programs (APS v1.0, APS v2.0, GNU Grep and BSD Grep)")


### Comparison with different file sizes seaching for two-word pattern
![](aps/2.0graf_6.png "Comparison of several string-matching programs (APS v1.0, APS v2.0, GNU Grep and BSD Grep)")
