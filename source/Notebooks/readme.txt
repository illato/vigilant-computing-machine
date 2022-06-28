How to get R kernel in Jupyter Notebook

-RStudio/R > terminal > ".libPaths()" -> (e.g. "C:/Users/Bob/Documents/R/R-4.1.3/library")
-Anaconda Prompt > "cd '/users/bob/documents/r/r-4.1.3/library'"
-Anaconda Prompt > "cd .."
-Anaconda Prompt > "cd bin"
-Anaconda Prompt > "R"
-R > "install.packages('IRkernel')"
-R > "IRkernel::installspec()"
# -R > "quit()"
# -Anaconda Prompt > "jupyter notebook" > "new" > notice R is an option now!