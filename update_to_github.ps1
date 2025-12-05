# PowerShell Script to Update Project to GitHub
# סקריפט לעדכון הפרויקט ל-GitHub

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GitHub Update Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Git is available
$gitPath = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitPath) {
    Write-Host "ERROR: Git not found in PATH!" -ForegroundColor Red
    Write-Host "Please add Git to your PATH or restart PowerShell after installing Git." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Trying to find Git in common locations..." -ForegroundColor Yellow
    
    # Try common Git installation paths
    $commonPaths = @(
        "C:\Program Files\Git\cmd\git.exe",
        "C:\Program Files (x86)\Git\cmd\git.exe",
        "$env:LOCALAPPDATA\Programs\Git\cmd\git.exe",
        "$env:ProgramFiles\Git\cmd\git.exe",
        "$env:ProgramFiles(x86)\Git\cmd\git.exe"
    )
    
    $found = $false
    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            $gitDir = Split-Path $path -Parent
            $env:Path = "$gitDir;$env:Path"
            Write-Host "Found Git at: $path" -ForegroundColor Green
            Write-Host "Added to PATH: $gitDir" -ForegroundColor Green
            $found = $true
            break
        }
    }
    
    if (-not $found) {
        Write-Host "Git not found in common locations." -ForegroundColor Red
        Write-Host ""
        Write-Host "Please try one of these options:" -ForegroundColor Yellow
        Write-Host "1. Restart PowerShell/Command Prompt after installing Git" -ForegroundColor Yellow
        Write-Host "2. Run Git commands manually (see git_commands.txt)" -ForegroundColor Yellow
        Write-Host "3. Use Git Bash or GitHub Desktop" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "You can also find where Git is installed by running:" -ForegroundColor Yellow
        Write-Host "  where git" -ForegroundColor White
        Write-Host "  (in Command Prompt, not PowerShell)" -ForegroundColor Gray
        exit 1
    }
}

# Get Git version
Write-Host "Git version:" -ForegroundColor Green
git --version
Write-Host ""

# Check if we're in a git repository
if (Test-Path .git) {
    Write-Host "Git repository already initialized." -ForegroundColor Green
    git status
} else {
    Write-Host "Initializing new Git repository..." -ForegroundColor Yellow
    git init
    Write-Host "Git repository initialized!" -ForegroundColor Green
}

Write-Host ""
Write-Host "Adding files to Git..." -ForegroundColor Yellow
git add .

Write-Host ""
Write-Host "Creating initial commit..." -ForegroundColor Yellow
$commitMessage = "Initial commit: E-Commerce ML Recommendation System

- User categorization with optimized clustering (58.8% Silhouette Score)
- Ultra-aggressive data enhancement (6.0x for top users)
- High cluster counts (100-400)
- Timeout mechanism (20 minutes)
- Early stopping when 88%+ is reached
- Hybrid recommendation system
- NLP search system"

git commit -m $commitMessage

Write-Host ""
Write-Host "Checking remote repository..." -ForegroundColor Yellow
$remote = git remote get-url origin -ErrorAction SilentlyContinue

if ($remote) {
    Write-Host "Remote already configured: $remote" -ForegroundColor Green
} else {
    Write-Host "Adding remote repository..." -ForegroundColor Yellow
    git remote add origin https://github.com/ReuvenRubin/e-commers-ML.git
    Write-Host "Remote added!" -ForegroundColor Green
}

Write-Host ""
Write-Host "Setting branch to 'main'..." -ForegroundColor Yellow
git branch -M main

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Ready to push to GitHub!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To push to GitHub, run:" -ForegroundColor Yellow
Write-Host "  git push -u origin main" -ForegroundColor White
Write-Host ""
Write-Host "If you get an error about unrelated histories, run:" -ForegroundColor Yellow
Write-Host "  git pull origin main --allow-unrelated-histories" -ForegroundColor White
Write-Host "  git push -u origin main" -ForegroundColor White
Write-Host ""

