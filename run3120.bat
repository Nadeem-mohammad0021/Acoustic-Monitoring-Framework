@echo off

REM Save current directory
set "orig_dir=%CD%"

REM Activate virtual environment directly
call C:\Envs\py311env\Scripts\activate.bat

REM Return to original directory
cd /d "%orig_dir%"

cmd