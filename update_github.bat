@echo off
REM Quick update to GitHub
REM עדכון מהיר ל-GitHub

cd /d "C:\Users\Reuven\Desktop\ML-eCommers-GitHub"

if not exist .git (
    echo Error: Not a git repository!
    pause
    exit /b 1
)

echo Updating GitHub...
echo.

REM Set git user identity (if not already set globally)
git config user.name "ReuvenRubin" 2>nul
git config user.email "reuven542rub@gmail.com" 2>nul

REM Add all changes
git add .

REM Check if there are changes to commit
git diff --cached --quiet
if %ERRORLEVEL% EQU 0 (
    echo No changes to commit.
    git status
) else (
    REM Commit with date/time
    for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set mydate=%%c-%%a-%%b
    for /f "tokens=1-2 delims=: " %%a in ('time /t') do set mytime=%%a:%%b
    set mytime=%mytime: =0%
    git commit -m "Update project - %mydate% %mytime%"
    
    REM Push to GitHub
    echo Pushing to GitHub...
    git push origin main
    
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo Successfully updated GitHub!
        echo Repository: https://github.com/ReuvenRubin/e-commers-ML
    ) else (
        echo.
        echo Error occurred during push. Please check the message above.
        echo.
        echo Common issues:
        echo - Make sure you have internet connection
        echo - Check if you have push permissions to the repository
        echo - Verify git credentials are set correctly
    )
)

pause

