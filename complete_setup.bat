@echo off
REM Complete Git Setup - Run this after setup_git.bat
REM השלמת הגדרת Git - הרץ את זה אחרי setup_git.bat

echo ========================================
echo Completing Git Setup
echo ========================================
echo.

cd /d "C:\Users\Reuven\Desktop\ML"

REM Configure Git user (if not already set)
echo Configuring Git user...
git config user.name "ReuvenRubin"
git config user.email "reuven542rub@gmail.com"
echo Git user configured.
echo.

REM Create commit
echo Creating commit...
git commit -m "Initial commit: E-Commerce ML Recommendation System with optimized user categorization (58.8%% Silhouette Score)"
echo.

REM Check if commit was successful
if %ERRORLEVEL% EQU 0 (
    echo Commit created successfully!
    echo.
    echo ========================================
    echo Ready to push to GitHub!
    echo ========================================
    echo.
    echo To push to GitHub, run:
    echo   git push -u origin main
    echo.
) else (
    echo Error creating commit. Please check the error message above.
    echo.
)

pause

