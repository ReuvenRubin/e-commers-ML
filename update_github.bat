@echo off
setlocal enabledelayedexpansion
REM Quick update to GitHub
REM עדכון מהיר ל-GitHub

echo ========================================
echo GitHub Update Script
echo ========================================
echo.

cd /d "C:\Users\Reuven\Desktop\ML-eCommers-GitHub" 2>nul
if not "%CD%"=="C:\Users\Reuven\Desktop\ML-eCommers-GitHub" (
    echo Error: Failed to change directory!
    echo Current directory: %CD%
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

echo Current directory: %CD%
echo.

if not exist .git (
    echo Error: Not a git repository!
    echo Current directory: %CD%
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
echo Checking for changes...
git diff --cached --quiet
set check_result=%ERRORLEVEL%
echo DEBUG: git diff returned: %check_result%
if %check_result% EQU 0 (
    echo.
    echo No changes to commit.
    echo Current status:
    git status
    echo.
    echo DEBUG: About to reach end of script (no changes branch)
) else (
    echo Changes found. Committing...
    echo DEBUG: About to commit...
    REM Commit with date/time
    for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set mydate=%%c-%%a-%%b
    for /f "tokens=1-2 delims=: " %%a in ('time /t') do set mytime=%%a:%%b
    set mytime=%mytime: =0%
    git commit -m "Update project - %mydate% %mytime%"
    
    REM Push to GitHub
    echo Pushing to GitHub...
    git push origin main
    set push_result=%ERRORLEVEL%
    
    if %push_result% EQU 0 (
        echo.
        echo ========================================
        echo Successfully updated GitHub!
        echo Repository: https://github.com/ReuvenRubin/e-commers-ML
        echo ========================================
    ) else (
        echo.
        echo ========================================
        echo Error occurred during push. Please check the message above.
        echo.
        echo Common issues:
        echo - Make sure you have internet connection
        echo - Check if you have push permissions to the repository
        echo - Verify git credentials are set correctly
        echo ========================================
    )
)

echo.
echo ========================================
echo Script completed.
echo ========================================
echo.
endlocal
echo Press any key to exit...
timeout /t 1 /nobreak >nul
pause

