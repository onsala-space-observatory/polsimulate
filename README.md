# POLSIMULATE

## Installation

Steps to install the `polsimulate` task into `casa`

 1. Clone the git repository into a directory of your choice
 (e.g., $HOME/.casa/NordicTools)

``` shell
cd $HOME/.casa/NordicTools
git clone <repository url>
cd polsimulate
buildmytasks      # in your shell, not casa!
```

This assumes that the directory containing the `buildmytasks` binary
(the same where the `casa` binary is residing) is part of your
PATH. You can check this by typing

``` shell
which buildmytasks
```

If you don't get any output, you should replace the last line of the
first shell code block above with

``` shell
PATH=<directory where your casa binary lives> buildmytasks
```

 2. Edit the file `$HOME/.casa/init.py`. Add the line:

``` shell
execfile('$HOME/.casa/NordicTools/polsimulate/mytasks.py')
```

Alternatively, just run that command inside `casa` before trying to
run the `polsimulate` task.


That's it! You should be able to run the new task in `casa`! Just doing:

``` shell
tget polsimulate
```

inside `casa` should load the task. To get help, just type `help
polsimulate` and to run it use `polsimulate()`.
