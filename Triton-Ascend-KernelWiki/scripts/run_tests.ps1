param(
    [string]$Python = "$PSScriptRoot\..\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $ProjectRoot

function Run-Python {
    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

New-Item -ItemType Directory -Force -Path "reports\ut", "reports\st", "reports\combined" | Out-Null

Run-Python -m coverage erase
Run-Python -m coverage run --data-file=.coverage.ut --branch --source=src/triton_ascend_kb -m pytest tests/ut --junitxml=reports/ut/junit.xml
Run-Python -m coverage report --data-file=.coverage.ut --fail-under=85
Run-Python -m coverage json --data-file=.coverage.ut -o reports/ut/coverage.json
Run-Python -m coverage xml --data-file=.coverage.ut -o reports/ut/coverage.xml
Run-Python -m coverage html --data-file=.coverage.ut -d reports/ut/html

Run-Python -m coverage run --data-file=.coverage.st --branch --source=src/triton_ascend_kb -m pytest tests/st --junitxml=reports/st/junit.xml
Run-Python -m coverage report --data-file=.coverage.st --fail-under=80
Run-Python -m coverage json --data-file=.coverage.st -o reports/st/coverage.json
Run-Python -m coverage xml --data-file=.coverage.st -o reports/st/coverage.xml
Run-Python -m coverage html --data-file=.coverage.st -d reports/st/html

Run-Python -m coverage run --data-file=.coverage.combined --branch --source=src/triton_ascend_kb -m pytest tests/ut tests/st --junitxml=reports/combined/junit.xml
Run-Python -m coverage report --data-file=.coverage.combined --fail-under=90
Run-Python -m coverage json --data-file=.coverage.combined -o reports/combined/coverage.json
Run-Python -m coverage xml --data-file=.coverage.combined -o reports/combined/coverage.xml
Run-Python -m coverage html --data-file=.coverage.combined -d reports/combined/html
