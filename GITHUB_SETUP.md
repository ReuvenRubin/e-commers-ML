# הוראות עדכון ל-GitHub

## דרך מהירה (מומלץ) 🚀

**אם Git מותקן**, פשוט הרץ את אחד מהסקריפטים:

### Windows PowerShell:
```powershell
.\update_to_github.ps1
```

### Windows Command Prompt:
```cmd
update_to_github.bat
```

הסקריפטים יעשו הכל אוטומטית!

---

## דרך ידנית

### שלב 1: התקנת Git (אם לא מותקן)

1. הורד Git מ: https://git-scm.com/download/win
2. התקן את Git עם ההגדרות המומלצות
3. פתח מחדש את PowerShell/Terminal

## שלב 2: יצירת Repository מקומי

```bash
# נווט לתיקיית הפרויקט
cd C:\Users\Reuven\Desktop\ML

# אתחל repository חדש
git init

# הוסף את כל הקבצים
git add .

# צור commit ראשוני
git commit -m "Initial commit: E-Commerce ML Recommendation System with optimized user categorization"
```

## שלב 3: חיבור ל-GitHub Repository

```bash
# הוסף את ה-remote repository
git remote add origin https://github.com/ReuvenRubin/e-commers-ML.git

# בדוק שה-remote נוסף
git remote -v
```

## שלב 4: Push ל-GitHub

```bash
# שנה את שם ה-branch ל-main (אם צריך)
git branch -M main

# Push ל-GitHub
git push -u origin main
```

## עדכונים עתידיים

לאחר השינויים הראשונים, לעדכונים עתידיים:

```bash
# הוסף שינויים
git add .

# צור commit
git commit -m "תיאור השינויים"

# Push ל-GitHub
git push
```

## הערות חשובות

- **לא להעלות קבצים רגישים**: ודא ש-`.gitignore` כולל:
  - `__pycache__/`
  - `*.pyc`
  - `.env`
  - קבצי נתונים גדולים (אם יש)

- **אם יש שגיאות**: אם ה-repository ב-GitHub כבר קיים, ייתכן שתצטרך:
  ```bash
  git pull origin main --allow-unrelated-histories
  ```
  ואז:
  ```bash
  git push -u origin main
  ```

## מצב נוכחי של הפרויקט

- **User Categorization**: Silhouette Score ~58.8% (יעד: 88%+)
- **Optimizations**: 
  - Ultra-aggressive data enhancement (6.0x for top users)
  - High cluster counts (100-400)
  - Timeout mechanism (20 minutes)
  - Early stopping when 88%+ is reached

## קבצים עיקריים

- `src/phase1/ml_implementation.py` - User categorization algorithm
- `src/phase2/recommendation_system_ml.py` - Recommendation system
- `src/phase3/nlp_search_system.py` - NLP search system
- `requirements.txt` - Dependencies

