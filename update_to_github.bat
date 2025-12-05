@echo off
REM Batch Script to Update Project to GitHub
REM סקריפט לעדכון הפרויקט ל-GitHub

echo ========================================
echo GitHub Update Script
echo ========================================
echo.

REM Check if Git is available
where git >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Git not found in PATH!
    echo Please add Git to your PATH or restart Command Prompt after installing Git.
    echo.
    echo Trying to find Git in common locations...
    
    REM Try common Git installation paths
    set "GIT_PATH="
    if exist "C:\Program Files\Git\cmd\git.exe" (
        set "GIT_PATH=C:\Program Files\Git\cmd"
        set "PATH=%PATH%;%GIT_PATH%"
        echo Found Git at: C:\Program Files\Git\cmd\git.exe
        goto :git_found
    )
    if exist "C:\Program Files (x86)\Git\cmd\git.exe" (
        set "GIT_PATH=C:\Program Files (x86)\Git\cmd"
        set "PATH=%PATH%;%GIT_PATH%"
        echo Found Git at: C:\Program Files (x86)\Git\cmd\git.exe
        goto :git_found
    )
    echo Git not found. Please install Git or add it to PATH.
    echo.
    echo You can also run Git commands manually:
    echo   1. Open Command Prompt as Administrator
    echo   2. Navigate to: C:\Users\Reuven\Desktop\ML
    echo   3. Run the Git commands from GITHUB_SETUP.md
    pause
    exit /b 1
    :git_found
)

REM Get Git version
echo Git version:
git --version
echo.

REM Check if we're in a git repository
if exist .git (
    echo Git repository already initialized.
    git status
) else (
    echo Initializing new Git repository...
    git init
    echo Git repository initialized!
)

echo.
echo Adding files to Git...
git add .

echo.
echo Creating initial commit...
git commit -m "Initial commit: E-Commerce ML Recommendation System with optimized user categorization (58.8%% Silhouette Score)"

echo.
echo Checking remote repository...
git remote get-url origin >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Remote already configured.
    git remote get-url origin
) else (
    echo Adding remote repository...
    git remote add origin https://github.com/ReuvenRubin/e-commers-ML.git
    echo Remote added!
)

echo.
echo Setting branch to 'main'...
git branch -M main

echo.
echo ========================================
echo Ready to push to GitHub!
echo ========================================
echo.
echo To push to GitHub, run:
echo   git push -u origin main
echo.
echo If you get an error about unrelated histories, run:
echo   git pull origin main --allow-unrelated-histories
echo   git push -u origin main
echo.
pause

