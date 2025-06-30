# HR Assistant Bot 

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Telegram API](https://img.shields.io/badge/Telegram%20Bot%20API-20.0%2B-blue)](https://core.telegram.org/bots/api)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Intelligent Recruitment Assistant

An AI-powered Telegram bot for automated candidate screening and job matching using Natural Language Processing (NLP) and semantic search technologies.

### ✨ Key Features
- **Smart CV analysis** with 50%+ relevance threshold
- **Multi-stage filtering** (age, citizenship)
- **Automated contact collection**
- **CSV database integration**

### 🛠 Tech Stack
- **Core**: Python 3.8+
- **NLP**: Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Framework**: `python-telegram-bot` v20+
- **Data**: Pandas/CSV

### Contacts
- **Разработчик | Developer:** Киселева Вероника | Kiseleva Veronica
- **Email:** kinilokis@gmail.com
- **Telegram:** @nikilichka
- **GitHub:** github.com/nikilichka

### 📜 License
MIT License

### 🚀 Quick Start
```bash
git clone https://github.com/yourusername/hr-assistant-bot.git
cd hr-assistant-bot
pip install -r requirements.txt
echo "TELEGRAM_TOKEN=your_bot_token" > .env
python bot.py
