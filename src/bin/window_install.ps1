# Navigate to the directory of the script
Set-Location $PSScriptRoot\..\..\

# Create directories
New-Item -ItemType Directory -Path .\data\spectral_image -Force
New-Item -ItemType Directory -Path .\data\spectral_image_1 -Force
New-Item -ItemType Directory -Path .\data\spectral_image_2 -Force
New-Item -ItemType Directory -Path .\RF_save -Force
New-Item -ItemType Directory -Path .\src\checkpoint -Force
New-Item -ItemType Directory -Path .\src\log\celery -Force
New-Item -ItemType Directory -Path .\src\run\celery -Force
New-Item -ItemType Directory -Path .\src\proj\db\db_new -Force
New-Item -ItemType Directory -Path .\src\data\img_col_data -Force
New-Item -ItemType Directory -Path .\src\data\saved_result -Force
New-Item -ItemType Directory -Path .\src\data\img_result_saved -Force
New-Item -ItemType Directory -Path .\model_saved\NN_save\using_mutual_information -Force

# Activate virtual environment
.\venv\Scripts\Activate

# Install requirements
pip install -r .\requirements.txt

# Install Celery
pip install -U Celery==5.3.6

# Deactivate virtual environment
deactivate
