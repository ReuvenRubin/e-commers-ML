@echo off
REM Push to GitHub
REM דחיפה ל-GitHub

echo ========================================
echo Pushing to GitHub
echo ========================================
echo.

cd /d "C:\Users\Reuven\Desktop\ML"

REM Check if Git is available
where git >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Git not found in PATH!
    echo Please run this from Command Prompt (not PowerShell)
    echo.
    pause
    exit /b 1
)

echo Current status:
git status --short
echo.

echo Checking if we need to push...
git status -sb
echo.

echo.
echo Pushing to GitHub...
echo (This may take a moment and may ask for credentials)
echo.
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Successfully pushed to GitHub!
    echo ========================================
    echo.
    echo Repository: https://github.com/ReuvenRubin/e-commers-ML
    echo.
) else (
    echo.
    echo ========================================
    echo Error occurred!
    echo ========================================
    echo.
    echo If you got an error about "unrelated histories", run:
    echo   git pull origin main --allow-unrelated-histories
    echo   git push -u origin main
    echo.
    echo If you need authentication, you may need to:
    echo   1. Create a Personal Access Token at: https://github.com/settings/tokens
    echo   2. Use your username and the token as password
    echo.
)

pause

