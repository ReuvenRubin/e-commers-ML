@echo off
REM Check Git Status
REM בדיקת סטטוס Git

cd /d "C:\Users\Reuven\Desktop\ML"

echo ========================================
echo Git Repository Status
echo ========================================
echo.

echo Checking Git status...
git status
echo.

echo Checking remote...
git remote -v
echo.

echo Checking last commit...
git log --oneline -1
echo.

echo Checking if we're ahead of remote...
git status -sb
echo.

echo ========================================
echo If you see "ahead of origin/main", run:
echo   git push -u origin main
echo ========================================
echo.

pause

