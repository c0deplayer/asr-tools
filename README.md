# ASR Benchmark

<div align="center">
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
  <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://img.shields.io/badge/Data%20License-CC_BY_NC_SA_4.0-blue" alt="Data License"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/Code%20License-MIT-blue" alt="Code License"></a>
</div>

## Overview

### Prerequisites

- Python 3.12+
- Required system packages:
  ```bash
  sudo apt-get install sox ffmpeg  # Ubuntu/Debian
  brew install sox ffmpeg          # macOS
  ```

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/[wip]
   cd [wip]
   ```

2. Install Python dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

   or using uv:

   ```bash
   uv sync
   ```

3. Configure your environment:
   - Copy `config/user/template.ini` to `config/user/user_config.ini`
   - Edit the file with your API keys and paths
