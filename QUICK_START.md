# הוראות מהירות לעדכון ל-GitHub

## שלב 1: פתח Command Prompt חדש

1. לחץ `Windows + R`
2. הקלד `cmd` ולחץ Enter
3. **חשוב:** פתח Command Prompt חדש (לא PowerShell)

## שלב 2: הרץ את הסקריפטים

```cmd
cd C:\Users\Reuven\Desktop\ML
setup_git.bat
complete_setup.bat
push_to_github.bat
```

או להריץ הכל בבת אחת:
```cmd
cd C:\Users\Reuven\Desktop\ML
push_to_github.bat
```

או הרץ את הפקודות ידנית:

```cmd
cd C:\Users\Reuven\Desktop\ML
git init
git add .
git commit -m "Initial commit: E-Commerce ML Recommendation System"
git remote add origin https://github.com/ReuvenRubin/e-commers-ML.git
git branch -M main
git push -u origin main
```

## אם Git לא נמצא:

1. פתח Command Prompt חדש (לא PowerShell)
2. הרץ: `where git`
3. אם זה לא עובד, פתח מחדש את Command Prompt אחרי התקנת Git

## אם יש שגיאה על "unrelated histories":

```cmd
git pull origin main --allow-unrelated-histories
git push -u origin main
```

