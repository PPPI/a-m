@echo off
setlocal
doskey python=D:\Python\Python35\bin\python.exe
set PYTHONPATH=D:\OneDrive\TopicalLinker\
python "%~dp0/__main__.py" %*
endlocal
