# GitHub Setup & Repository Clone Tutorial

This tutorial will guide you through creating a GitHub account, setting up SSH keys, and cloning the Monte Carlo DCF Simulator repository.

---

## Part 1: Create a GitHub Account

### Step 1: Go to GitHub
1. Open your web browser and navigate to: **https://github.com**
2. Click the **"Sign up"** button in the top right corner

### Step 2: Create Your Account
1. Enter your **email address** (or use your existing email)
2. Create a **password** (must be at least 8 characters with a number and lowercase letter)
3. Choose a **username** (this will be your GitHub handle, e.g., `yourusername`)
4. Verify you're not a robot by completing the CAPTCHA
5. Click **"Create account"**

### Step 3: Verify Your Email
1. Check your email inbox for a verification email from GitHub
2. Click the verification link in the email
3. Complete any additional setup steps GitHub prompts

### Step 4: Choose Your Plan
1. GitHub will ask you to choose a plan
2. For this tutorial, select **"Free"** (the default option)
3. You can skip the personalization questions if you want

**✅ You now have a GitHub account!**

---

## Part 2: Generate an SSH Key

SSH keys allow you to securely connect to GitHub without entering your password every time.

### Step 1: Check for Existing SSH Keys
1. Open your **Terminal** (Mac/Linux) or **Git Bash** (Windows)
2. Run this command to check if you already have SSH keys:
   ```bash
   ls -al ~/.ssh
   ```
3. Look for files named `id_rsa` or `id_ed25519`. If they exist, you can use them or create a new one.

### Step 2: Generate a New SSH Key
1. In your terminal, run this command (replace `your_email@example.com` with your GitHub email):
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```
   
   **Note**: If your system doesn't support `ed25519`, use:
   ```bash
   ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
   ```

2. When prompted, press **Enter** to accept the default file location (`~/.ssh/id_ed25519`)

3. When prompted for a passphrase, you can either:
   - Press **Enter** twice to create a key without a passphrase (easier, but less secure)
   - Or enter a passphrase for extra security (you'll need to enter it each time you use the key)

### Step 3: Start the SSH Agent
1. Start the SSH agent:
   ```bash
   eval "$(ssh-agent -s)"
   ```
   
   You should see output like: `Agent pid 12345`

2. Add your SSH key to the agent:
   ```bash
   ssh-add ~/.ssh/id_ed25519
   ```
   
   (If you used RSA, use: `ssh-add ~/.ssh/id_rsa`)

### Step 4: Copy Your SSH Public Key
1. Display your public key:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
   
   (If you used RSA, use: `cat ~/.ssh/id_rsa.pub`)

2. **Copy the entire output** (it will look something like):
   ```
   ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIG... your_email@example.com
   ```

### Step 5: Add SSH Key to GitHub
1. Go to GitHub.com and **sign in**
2. Click your **profile picture** in the top right
3. Click **"Settings"**
4. In the left sidebar, click **"SSH and GPG keys"**
5. Click the green **"New SSH key"** button
6. Give it a **Title** (e.g., "My MacBook" or "My Computer")
7. In the **"Key"** field, **paste your public key** (the one you copied in Step 4)
8. Click **"Add SSH key"**
9. You may be prompted to enter your GitHub password to confirm

**✅ Your SSH key is now set up!**

### Step 6: Test Your SSH Connection
1. In your terminal, run:
   ```bash
   ssh -T git@github.com
   ```
2. You should see a message like:
   ```
   Hi username! You've successfully authenticated, but GitHub does not provide shell access.
   ```
3. If you see this, **you're all set!** If you see an error, check that you added the public key correctly.

---

## Part 3: Clone the Repository

### Step 1: Navigate to Your Desired Directory
1. Open your terminal
2. Navigate to where you want to store the project (e.g., Desktop, Documents, or a Projects folder):
   ```bash
   cd ~/Desktop
   ```
   or
   ```bash
   cd ~/Documents
   ```

### Step 2: Clone the Repository
1. Run the clone command:
   ```bash
   git clone git@github.com:DatOneOPGuy/MonteCarlo.git
   ```
2. You should see output like:
   ```
   Cloning into 'MonteCarlo'...
   remote: Enumerating objects: X, done.
   remote: Counting objects: 100% (X/X), done.
   remote: Compressing objects: 100% (X/X), done.
   remote: Total X (delta X), reused X (delta X), pack-reused X
   Receiving objects: 100% (X/X), done.
   Resolving deltas: 100% (X/X), done.
   ```

### Step 3: Navigate into the Project
```bash
cd MonteCarlo
```

**✅ The repository is now on your computer!**

---

## Part 4: Set Up and Run the Project

### Step 1: Check Python Installation
1. Check if Python 3.10+ is installed:
   ```bash
   python3 --version
   ```
2. You should see something like `Python 3.10.x` or higher. If not, install Python 3.10+ from [python.org](https://www.python.org/downloads/)

### Step 2: Create a Virtual Environment
1. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   ```
2. Activate the virtual environment:
   - **Mac/Linux**:
     ```bash
     source venv/bin/activate
     ```
   - **Windows** (Git Bash or PowerShell):
     ```bash
     venv\Scripts\activate
     ```
3. You should see `(venv)` at the beginning of your terminal prompt

### Step 3: Install Dependencies
1. Install the project dependencies:
   ```bash
   pip install -e .
   ```
   
   Or if you want dev dependencies too:
   ```bash
   pip install -e ".[dev]"
   ```

2. Wait for installation to complete (this may take a few minutes)

### Step 4: Run the Streamlit Application
1. Start the Streamlit app:
   ```bash
   streamlit run app/streamlit_app.py
   ```
2. Your browser should automatically open to `http://localhost:8501`
3. If it doesn't, open your browser and go to: `http://localhost:8501`

### Step 5: Use the Application
1. In the Streamlit app:
   - Check **"DCF Valuation Mode"** in the sidebar
   - Optionally enable **"Apple Project Preset"** for pre-configured Apple valuation parameters
   - Adjust simulation parameters (number of simulations, seed, etc.)
   - Click **"Run Simulation"**
   - View results, charts, and analytics

### Step 6: Stop the Application
- In your terminal, press **Ctrl+C** to stop the Streamlit server

---

## Troubleshooting

### SSH Key Issues
- **"Permission denied (publickey)"**: Make sure you added your public key to GitHub (not your private key)
- **"Could not resolve hostname github.com"**: Check your internet connection
- **"Host key verification failed"**: Run `ssh-keyscan github.com >> ~/.ssh/known_hosts`

### Git Clone Issues
- **"Repository not found"**: Make sure the repository is public or you have access to it
- **"Permission denied"**: Verify your SSH key is added to GitHub and test with `ssh -T git@github.com`

### Python/Installation Issues
- **"python3: command not found"**: Install Python 3.10+ from [python.org](https://www.python.org/downloads/)
- **"pip: command not found"**: Make sure Python is installed correctly and includes pip
- **"ModuleNotFoundError"**: Make sure you activated the virtual environment and installed dependencies

### Streamlit Issues
- **"streamlit: command not found"**: Make sure you activated the virtual environment and installed dependencies
- **Port already in use**: If port 8501 is busy, Streamlit will try the next available port (8502, 8503, etc.)

---

## Quick Reference Commands

```bash
# Navigate to project
cd ~/Desktop/MonteCarlo

# Activate virtual environment
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -e .

# Run the app
streamlit run app/streamlit_app.py

# Deactivate virtual environment (when done)
deactivate
```

---

## Next Steps

- Read `app/README.md` for detailed project documentation
- Check `app/APPLE_PROJECT_GUIDE.md` for Apple valuation instructions
- Review `app/METHODOLOGY_EXPLANATION.md` for methodology details
- Explore the code in `app/core/` to understand the implementation

**Happy simulating! 🎲📊**


