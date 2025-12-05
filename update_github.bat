@echo off
REM Quick update to GitHub
REM עדכון מהיר ל-GitHub

cd /d "C:\Users\Reuven\Desktop\ML"

echo Updating GitHub...
echo.

git add .
git commit -m "Remove unnecessary setup scripts"
git push

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Successfully updated GitHub!
) else (
    echo.
    echo Error occurred. Please check the message above.
)

pause

