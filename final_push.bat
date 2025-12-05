@echo off
REM Final Push to GitHub - Complete Setup
REM דחיפה סופית ל-GitHub - הגדרה מלאה

echo ========================================
echo Final Push to GitHub
echo ========================================
echo.

cd /d "C:\Users\Reuven\Desktop\ML"

REM Add all new and modified files
echo Adding all files...
git add .
echo.

REM Commit changes
echo Creating commit...
git commit -m "Add GitHub setup scripts and update documentation"
echo.

REM Check if commit was successful
if %ERRORLEVEL% NEQ 0 (
    echo No new changes to commit, or commit already exists.
    echo.
) else (
    echo Commit created successfully!
    echo.
)

REM Push to GitHub
echo Pushing to GitHub...
echo (This may ask for your GitHub username and password/token)
echo.
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo SUCCESS! Project pushed to GitHub!
    echo ========================================
    echo.
    echo Repository: https://github.com/ReuvenRubin/e-commers-ML
    echo.
    echo You can now view your code on GitHub!
    echo.
) else (
    echo.
    echo ========================================
    echo Push failed. Possible reasons:
    echo ========================================
    echo.
    echo 1. Authentication required:
    echo    - Username: ReuvenRubin
    echo    - Password: Use Personal Access Token (not regular password)
    echo    - Create token at: https://github.com/settings/tokens
    echo.
    echo 2. Unrelated histories error:
    echo    Run: git pull origin main --allow-unrelated-histories
    echo    Then: git push -u origin main
    echo.
    echo 3. Repository already exists with different content:
    echo    You may need to force push (careful!):
    echo    git push -u origin main --force
    echo.
)

pause

