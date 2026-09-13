@echo off
setlocal

pushd "%~dp0"

docker build --tag henry-local-validate .
if errorlevel 1 goto :error

docker run --rm -it -v "%~dp0.env:/app/.env:ro" henry-local-validate
set "exit_code=%ERRORLEVEL%"
popd
exit /b %exit_code%

:error
set "exit_code=%ERRORLEVEL%"
popd
exit /b %exit_code%