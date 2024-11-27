# Create project directory
mkdir wave_network
cd wave_network

# Create directories
mkdir src data models

# Create required files
touch src/main.py src/model.py src/utils.py requirements.txt README.md

# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install torch numpy pandas scikit-learn matplotlib