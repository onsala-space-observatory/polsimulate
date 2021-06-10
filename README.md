# POLSIMULATE

## Installation

Steps to install the `polsimulate` task into `casa`

 1. Clone the git repository into a directory of your choice
 (e.g., $HOME/.casa/NordicTools)

``` shell
cd $HOME/.casa/NordicTools
git clone <repository url>
cd polsimulate
buildmytasks
```
 2. Edit the file `$HOME/.casa/init.py`. Add the line:

``` shell
execfile('$HOME/.casa/NordicTools/polsimulate/mytasks.py')
```

That's it! You should be able to run the new task in CASA! Just doing:

``` shell
tget polsimulate
```

inside `casa` should load the task. To get help, just type `help polsimulate`
