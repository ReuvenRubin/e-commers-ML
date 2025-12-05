@echo off
REM Simple Git Setup Script
REM סקריפט פשוט להגדרת Git

echo ========================================
echo Git Setup for GitHub
echo ========================================
echo.

REM Navigate to project directory
cd /d "C:\Users\Reuven\Desktop\ML"
echo Current directory: %CD%
echo.

REM Check if Git is available
echo Checking for Git...
where git >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Git found!
    git --version
    echo.
    goto :git_available
) else (
    echo Git not found in PATH.
    echo.
    echo Please make sure Git is installed and added to PATH.
    echo Or restart Command Prompt after installing Git.
    echo.
    pause
    exit /b 1
)

:git_available
echo.
echo ========================================
echo Setting up Git repository...
echo ========================================
echo.

REM Initialize Git repository
if exist .git (
    echo Git repository already exists.
) else (
    echo Initializing Git repository...
    git init
    echo Done!
)

echo.
echo Adding files to Git...
git add .

echo.
echo Creating commit...
git commit -m "Initial commit: E-Commerce ML Recommendation System with optimized user categorization"

echo.
echo Setting up remote repository...
git remote remove origin >nul 2>&1
git remote add origin https://github.com/ReuvenRubin/e-commers-ML.git
echo Remote added: https://github.com/ReuvenRubin/e-commers-ML.git

echo.
echo Setting branch to main...
git branch -M main

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To push to GitHub, run:
echo   git push -u origin main
echo.
echo If you get an error, you may need to run:
echo   git pull origin main --allow-unrelated-histories
echo   git push -u origin main
echo.
pause

